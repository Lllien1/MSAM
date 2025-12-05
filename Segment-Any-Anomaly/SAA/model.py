import cv2
import numpy as np
import torch
from PIL import Image

# Grounding DINO（基于原始仓库做了轻量改动）
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# Segment Anything 模块
from SAM.segment_anything import build_sam, SamPredictor
# ImageNet 预训练特征提取器
from .modelinet import ModelINet


class Model(torch.nn.Module):
    def __init__(self,
                 ## DINO 模块
                 dino_config_file,
                 dino_checkpoint,

                 ## SAM 模块
                 sam_checkpoint,

                 ## 推理参数
                 box_threshold,
                 text_threshold,

                 ## 其它配置
                 out_size=256,
                 device='cuda',

                 ):
        '''
        初始化整套异常检测流水线，将 Grounding DINO、SAM 与 ImageNet 先验特征提取器串联起来。

        Args:
            dino_config_file: DINO 的配置文件路径。
            dino_checkpoint: DINO 的权重文件路径。
            sam_checkpoint: SAM 的权重文件路径。
            box_threshold: 文本框过滤阈值。
            text_threshold: 文本匹配阈值。
            out_size: 异常图的目标输出分辨率。
            device: 推理设备，例如 'cuda:0'。

        NOTE:
            1. 论文中属性提示 P^P 作用于区域 R，此实现中作用于检测框级区域 R^B。
            2. 该版本尚未加入 IoU 约束。
            3. 模块目前仅支持批大小为 1 的输入。
        '''
        super(Model, self).__init__()

        # 构建核心模型组件
        self.anomaly_region_generator = self.load_dino(dino_config_file, dino_checkpoint, device=device)
        self.anomaly_region_refiner = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.visual_saliency_extractor = ModelINet(device=device)

        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        # 推理阈值参数
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # 运行时状态
        self.out_size = out_size
        self.device = device
        self.is_sam_set = False

    def load_dino(self, model_config_path, model_checkpoint_path, device) -> torch.nn.Module:
        '''
        根据配置与权重加载 Grounding DINO 模型，并切换到 eval 模式。

        Args:
            model_config_path: 配置文件路径。
            model_checkpoint_path: 权重文件路径。
            device: 推理设备。

        Returns:
            torch.nn.Module: 已加载的 DINO 模型。
        '''
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()
        model = model.to(device)
        return model

    def get_grounding_output(self, image, caption, device="cpu") -> (torch.Tensor, torch.Tensor, str):
        '''
        调用 Grounding DINO，获取与文本提示匹配的候选框及其 logits。

        Args:
            image: 预处理后的图像张量。
            caption: 文本提示。
            device: 推理设备。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: 候选框、sigmoid logits 与标准化文本。
        '''
        caption = caption.lower()
        caption = caption.strip()

        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(device)

        with torch.no_grad():
            outputs = self.anomaly_region_generator(image[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        return boxes, logits, caption

    def set_ensemble_text_prompts(self, text_prompt_list: list, verbose=False) -> None:
        '''
        解析批量缺陷提示与过滤提示，为后续集成式 mask 生成做准备。

        Args:
            text_prompt_list: [(缺陷提示, 过滤提示)] 形式的列表。
            verbose: 是否打印解析结果。
        '''
        self.defect_prompt_list = [f[0] for f in text_prompt_list]
        self.filter_prompt_list = [f[1] for f in text_prompt_list]

        if verbose:
            print('used ensemble text prompts ===========')

            for d, t in zip(self.defect_prompt_list, self.filter_prompt_list):
                print(f'det prompts: {d}')
                print(f'filtered background: {t}')

            print('======================================')

    def set_property_text_prompts(self, property_prompts, verbose=False) -> None:
        '''
        从属性提示字符串中提取目标数量、面积比例等全局约束。

        Args:
            property_prompts: 包含多项属性的字符串。
            verbose: 是否打印解析后的参数。
        '''

        self.object_prompt = property_prompts.split(' ')[7]
        self.object_number = int(property_prompts.split(' ')[5])
        self.k_mask = int(property_prompts.split(' ')[12])
        self.defect_area_threshold = float(property_prompts.split(' ')[19])
        self.object_max_area = 1. / self.object_number
        self.object_min_area = 0.
        self.similar = property_prompts.split(' ')[6]

        if verbose:
            print(f'{self.object_prompt}, '
                  f'{self.object_number}, '
                  f'{self.k_mask}, '
                  f'{self.defect_area_threshold}, '
                  f'{self.object_max_area}, '
                  f'{self.object_min_area}')

    def ensemble_text_guided_mask_proposal(self, image, object_phrase_list, filtered_phrase_list,
                                           object_max_area, object_min_area,
                                           bbox_score_thr, text_score_thr):
        '''
        结合文本提示与属性过滤生成候选区域，并通过 SAM 细化为像素级掩码。

        Args:
            image: 原始 BGR 图像。
            object_phrase_list: 需要检测的文本提示列表。
            filtered_phrase_list: 需要剔除的背景提示列表。
            object_max_area: 允许的最大框面积比例。
            object_min_area: 允许的最小框面积比例。
            bbox_score_thr: DINO 框分数阈值。
            text_score_thr: 文本匹配阈值。

        Returns:
            Tuple[List[np.ndarray], np.ndarray, float]: 掩码集合、对应得分以及最大框面积。
        '''

        size = image.shape[:2]
        H, W = size[0], size[1]

        dino_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        dino_image, _ = self.transform(dino_image, None)  # 3, h, w

        if self.is_sam_set == False:
            self.anomaly_region_refiner.set_image(image)
            self.is_sam_set = True

        ensemble_boxes = []
        ensemble_logits = []
        ensemble_phrases = []

        max_box_area = 0.

        for object_phrase, filtered_phrase in zip(object_phrase_list, filtered_phrase_list):

            ########## 文本提示用于区域提案
            boxes, logits, object_phrase = self.text_guided_region_proposal(dino_image, object_phrase)

            ########## 属性提示用于区域过滤
            boxes_filtered, logits_filtered, pred_phrases = self.bbox_suppression(boxes, logits, object_phrase,
                                                                                  filtered_phrase,
                                                                                  bbox_score_thr, text_score_thr,
                                                                                  object_max_area, object_min_area)
            ## 若该轮没有候选框
            if boxes_filtered is not None:
                ensemble_boxes += [boxes_filtered]
                ensemble_logits += logits_filtered
                ensemble_phrases += pred_phrases

                boxes_area = boxes_filtered[:, 2] * boxes_filtered[:, 3]

                if boxes_area.max() > max_box_area:
                    max_box_area = boxes_area.max()

        if ensemble_boxes != []:
            ensemble_boxes = torch.cat(ensemble_boxes, dim=0)
            ensemble_logits = np.stack(ensemble_logits, axis=0)

            # 将检测框从归一化坐标还原到原图尺度
            for i in range(ensemble_boxes.size(0)):
                ensemble_boxes[i] = ensemble_boxes[i] * torch.Tensor([W, H, W, H]).to(self.device)
                ensemble_boxes[i][:2] -= ensemble_boxes[i][2:] / 2
                ensemble_boxes[i][2:] += ensemble_boxes[i][:2]

            # 将检测框通过 SAM 细化为掩码
            masks, logits = self.region_refine(ensemble_boxes, ensemble_logits, H, W)

        else:  # 如果没有候选框则返回空掩码
            masks = [np.zeros((H, W), dtype=bool)]
            logits = [0]
            max_box_area = 1

        return masks, logits, max_box_area

    def text_guided_region_proposal(self, dino_image, object_phrase):
        '''
        直接调用 Grounding DINO 输出文本提示对应的检测框。

        Args:
            dino_image: 预处理后的 PIL 图像张量。
            object_phrase: 目标文本提示。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: 框、logits 以及实际使用的文本。
        '''
        # 直接复用 Grounding DINO 的检测结果
        boxes, logits, caption = self.get_grounding_output(
            dino_image, object_phrase, device=self.device
        )

        return boxes, logits, caption

    def bbox_suppression(self, boxes, logits, object_phrase, filtered_phrase,
                         bbox_score_thr, text_score_thr,
                         object_max_area, object_min_area,
                         with_logits=True):
        '''
        综合分数、面积及文本过滤策略剔除不符合要求的候选框。

        Args:
            boxes: DINO 输出的归一化框。
            logits: 对应 logits。
            object_phrase: 目标提示。
            filtered_phrase: 背景过滤词。
            bbox_score_thr: 框分数阈值。
            text_score_thr: 文本匹配阈值。
            object_max_area: 最大面积比例。
            object_min_area: 最小面积比例。
            with_logits: 是否在短语后附加分数。

        Returns:
            Tuple[torch.Tensor, List[float], List[str]]: 通过筛选的框、分数与短语。
        '''

        # 根据多种策略筛选候选输出
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        boxes_area = boxes_filt[:, 2] * boxes_filt[:, 3]

        # 根据框分数与面积双重约束过滤候选框

        # 策略1：按照框分数阈值过滤
        box_score_mask = logits_filt.max(dim=1)[0] > bbox_score_thr

        # 策略2：限制最大面积
        box_max_area_mask = boxes_area < (object_max_area)

        # 策略3：限制最小面积
        box_min_area_mask = boxes_area > (object_min_area)

        filt_mask = torch.bitwise_and(box_score_mask, box_max_area_mask)
        filt_mask = torch.bitwise_and(filt_mask, box_min_area_mask)

        if torch.sum(filt_mask) == 0:  # 如果一次筛选都没有命中
            return None, None, None
        else:
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # 解析文本 token
        tokenlizer = self.anomaly_region_generator.tokenizer
        tokenized = tokenlizer(object_phrase)

        # 构建最终输出
        pred_phrases = []
        boxes_filtered = []
        logits_filtered = []
        for logit, box in zip(logits_filt, boxes_filt):
            # 策略4：按文本匹配阈值过滤
            pred_phrase = get_phrases_from_posmap(logit > text_score_thr, tokenized, tokenlizer)

            # 策略5：过滤背景类别
            if pred_phrase.count(filtered_phrase) > 0:  # 避免预测背景类别
                continue

            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

            boxes_filtered.append(box)
            logits_filtered.append(logit.max().item())

        if boxes_filtered == []:
            return None, None, None

        boxes_filtered = torch.stack(boxes_filtered, dim=0)

        return boxes_filtered, logits_filtered, pred_phrases

    def region_refine(self, boxes_filtered, logits_filtered, H, W):
        '''
        调用 SAM 对候选框进行细化，生成像素级掩码。

        Args:
            boxes_filtered: 通过筛选的检测框。
            logits_filtered: 对应得分。
            H: 原图高度。
            W: 原图宽度。

        Returns:
            Tuple[List[np.ndarray], List[float]]: 掩码与得分。
        '''
        if boxes_filtered == []:
            return [np.zeros((H, W), dtype=bool)], [0]

        transformed_boxes = self.anomaly_region_refiner.transform.apply_boxes_torch(boxes_filtered, (H, W)).to(
            self.device)

        masks, _, _ = self.anomaly_region_refiner.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks = masks.cpu().squeeze(1).numpy()

        return masks, logits_filtered

    def saliency_prompting(self, image, object_masks, defect_masks, defect_logits):
        '''
        基于显著性图对缺陷掩码重新打分，提升与对象差异大的区域权重。

        Args:
            image: 原始输入图像。
            object_masks: 对象掩码列表。
            defect_masks: 缺陷掩码列表。
            defect_logits: 缺陷置信度。

        Returns:
            Tuple[List[np.ndarray], np.ndarray, np.ndarray]: 掩码、重算得分与显著性图。
        '''

        ###### 自相似显著性计算
        similarity_map = self.visual_saliency_calculation(image, object_masks)

        ###### 根据显著性重新打分
        defect_masks, defect_rescores = self.rescore(defect_masks, defect_logits, similarity_map)

        return defect_masks, defect_rescores, similarity_map

    def single_object_similarity(self, image, object_masks):
        '''
        针对单对象场景，计算该对象与自身特征的差异度，生成显著性热力图。

        Args:
            image: 原始输入图像。
            object_masks: 单个对象掩码。

        Returns:
            np.ndarray: 与原图等尺寸的显著性热力图。
        '''
        # 仅关注对象区域特征，充分利用 GPU 版本推理

        # 计算整幅图的相似度开销巨大，因此统一缩放到较小分辨率
        self.visual_saliency_extractor.set_img_size(256)
        resize_image = cv2.resize(image, (256, 256))
        features, ratio_h, ratio_w = self.visual_saliency_extractor(resize_image)

        B, C, H, W = features.shape
        assert B == 1
        features_flattern = features.view(B * C, H * W)

        features_self_similarity = features_flattern.T @ features_flattern
        features_self_similarity = 0.5 * (1 - features_self_similarity)

        features_self_similarity = features_self_similarity.sort(dim=1, descending=True)[0]

        # 默认取前 400 个相似度值来稳定显著性估计
        features_self_similarity = torch.mean(features_self_similarity[:, :400], dim=1)
        heatMap2 = features_self_similarity.view(H, W).cpu().numpy()

        mask_anomaly_scores = cv2.resize(heatMap2, (image.shape[1], image.shape[0]))
        # 如需屏蔽对象外区域，可重新启用上方注释逻辑
        return mask_anomaly_scores

    def visual_saliency_calculation(self, image, object_masks):
        '''
        根据对象数量选择单实例或多实例策略，生成最终的显著性/相似度图。

        Args:
            image: 输入图像。
            object_masks: 对象掩码数组。

        Returns:
            np.ndarray: 显著性图。
        '''

        if self.object_number == 1:  # 单实例策略
            mask_area = np.sum(object_masks, axis=(1, 2))
            object_mask = object_masks[mask_area.argmax(), :, :]
            self_similarity_anomaly_map = self.single_object_similarity(image, object_mask)
            return self_similarity_anomaly_map

        else:  # 多实例策略
            resize_image = cv2.resize(image, (1024, 1024))
            features, ratio_h, ratio_w = self.visual_saliency_extractor(resize_image)

            feature_size = features.shape[2:]
            object_masks_clone = object_masks.copy()
            object_masks_clone = object_masks_clone.astype(np.int32)

            resize_object_masks = []
            for object_mask in object_masks_clone:
                resize_object_masks.append(cv2.resize(object_mask, feature_size, interpolation=cv2.INTER_NEAREST))

            mask_anomaly_scores = []

            for indx in range(len(resize_object_masks)):
                other_object_masks1 = resize_object_masks[:indx]
                other_object_masks2 = resize_object_masks[indx + 1:]
                other_object_masks = other_object_masks1 + other_object_masks2

                one_mask_feature, \
                one_feature_location, \
                other_mask_features = self.region_feature_extraction(
                    features,
                    resize_object_masks[indx],
                    other_object_masks
                )

                similarity = one_mask_feature @ other_mask_features.T  # (H*W, N)
                similarity = similarity.max(dim=1)[0]
                anomaly_score = 0.5 * (1. - similarity)
                anomaly_score = anomaly_score.cpu().numpy()

                mask_anomaly_score = np.zeros(feature_size)
                for location, score in zip(one_feature_location, anomaly_score):
                    mask_anomaly_score[location[0], location[1]] = score

                mask_anomaly_scores.append(mask_anomaly_score)

            mask_anomaly_scores = np.stack(mask_anomaly_scores, axis=0)
            mask_anomaly_scores = np.max(mask_anomaly_scores, axis=0)
            mask_anomaly_scores = cv2.resize(mask_anomaly_scores, (image.shape[1], image.shape[0]))

            return mask_anomaly_scores

    def region_feature_extraction(self, features, one_object_mask, other_object_masks):
        '''
        基于 ImageNet 预训练特征提取当前对象的像素特征，并汇集其它对象特征用于对比。

        Args:
            features: 视觉显著性网络输出的特征张量。
            one_object_mask: 当前对象的二值掩码。
            other_object_masks: 其余对象的掩码列表。

        Returns:
            Tuple[torch.Tensor, np.ndarray, torch.Tensor]: 当前对象特征、像素坐标以及其它对象特征。
        '''
        features_clone = features.clone()
        one_mask_feature = []
        one_feature_location = []
        for h in range(one_object_mask.shape[0]):
            for w in range(one_object_mask.shape[1]):
                if one_object_mask[h, w] > 0:
                    one_mask_feature += [features_clone[:, :, h, w].clone()]
                    one_feature_location += [np.array((h, w))]
                    features_clone[:, :, h, w] = 0.

        one_feature_location = np.stack(one_feature_location, axis=0)
        one_mask_feature = torch.cat(one_mask_feature, dim=0)

        B, C, H, W = features_clone.shape
        assert B == 1
        features_clone_flattern = features_clone.view(C, -1)

        other_mask_features = []
        for other_object_mask in other_object_masks:
            other_object_mask_flattern = other_object_mask.reshape(-1)
            other_mask_feature = features_clone_flattern[:, other_object_mask_flattern > 0]
            other_mask_features.append(other_mask_feature)

        other_mask_features = torch.cat(other_mask_features, dim=1).T

        return one_mask_feature, one_feature_location, other_mask_features

    def rescore(self, defect_masks, defect_logits, similarity_map):
        '''
        将显著性相似度映射到缺陷掩码得分，提升异质区域的权重。

        Args:
            defect_masks: 缺陷掩码列表。
            defect_logits: 原始置信度。
            similarity_map: 显著性图。

        Returns:
            Tuple[List[np.ndarray], np.ndarray]: 原掩码与新的置信度。
        '''
        defect_rescores = []
        for mask, logit in zip(defect_masks, defect_logits):
            if similarity_map[mask].size == 0:
                similarity_score = 1.
            else:
                similarity_score = np.exp(3 * similarity_map[mask].mean())

            refined_score = logit * similarity_score
            defect_rescores.append(refined_score)

        defect_rescores = np.stack(defect_rescores, axis=0)

        return defect_masks, defect_rescores

    def confidence_prompting(self, defect_masks, defect_scores, similarity_map):
        '''
        选取得分最高的若干掩码，进行加权融合得到最终异常热图。

        Args:
            defect_masks: 缺陷掩码集合。
            defect_scores: 对应置信度。
            similarity_map: 显著性图（此处主要用于附加信息）。

        Returns:
            np.ndarray: 归一化并重采样后的异常概率图。
        '''
        mask_indx = defect_scores.argsort()[-self.k_mask:]

        filtered_masks = []
        filtered_scores = []

        for indx in mask_indx:
            filtered_masks.append(defect_masks[indx])
            filtered_scores.append(defect_scores[indx])

        anomaly_map = np.zeros(defect_masks[0].shape)
        weight_map = np.ones(defect_masks[0].shape)

        for mask, logits in zip(filtered_masks, filtered_scores):
            anomaly_map += mask * logits
            weight_map += mask * 1.

        anomaly_map[weight_map > 0] /= weight_map[weight_map > 0]
        anomaly_map = cv2.resize(anomaly_map, (self.out_size, self.out_size))
        return anomaly_map

    def forward(self, image: np.ndarray):
        '''
        前向推理入口，串联对象检测、缺陷提案、显著性重加权与置信度融合。

        Args:
            image: BGR 格式的输入图像。

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: 异常热力图与包含显著性图的附加信息。
        '''
        ####### 对象提示（TGMP）用于定位主要对象
        object_masks, object_logits, object_area = self.ensemble_text_guided_mask_proposal(
            image,
            [self.object_prompt],
            ['PlaceHolder'],
            self.object_max_area,
            self.object_min_area,
            self.box_threshold,
            self.text_threshold
        )

        ###### 推理：根据对象面积动态设定缺陷面积阈值
        self.defect_max_area = object_area * self.defect_area_threshold
        self.defect_min_area = 0.

        ####### 语言提示与属性提示 $\mathcal{P}^L$、$\mathcal{P}^S$ 生成缺陷候选
        defect_masks, defect_logits, _ = self.ensemble_text_guided_mask_proposal(
            image,
            self.defect_prompt_list,
            self.filter_prompt_list,
            self.defect_max_area,
            self.defect_min_area,
            self.box_threshold,
            self.text_threshold
        )

        ###### 显著性提示 $\mathcal{P}^S$ 结合显著性图重排序
        defect_masks, defect_rescores, similarity_map = self.saliency_prompting(
            image,
            object_masks,
            defect_masks,
            defect_logits
        )

        ##### 置信度提示 $\mathcal{P}^C$ 选取得分最高的掩码
        anomaly_map = self.confidence_prompting(defect_masks, defect_rescores, similarity_map)

        self.is_sam_set = False

        appendix = {'similarity_map': similarity_map}

        return anomaly_map, appendix
