# -*- coding: utf-8 -*-
import argparse

# 本脚本负责评估 Segment Any Anomaly (SAA) 模型在不同数据集与类别上的检测性能，
# 核心流程包括：加载数据集与模型、执行推理、可视化结果、并输出 ROC/AUC/F1 等指标。

from tqdm import tqdm

import SAA as SegmentAnyAnomaly
from datasets import *
from utils.csv_utils import *
from utils.eval_utils import *
from utils.metrics import *
from utils.training_utils import *


def eval(
        # model-related
        model,
        train_data: DataLoader,
        test_data: DataLoader,

        # visual-related
        resolution,
        is_vis,

        # experimental parameters
        dataset,
        class_name,
        cal_pro,
        img_dir,
        k_shot,
        experiment_indx,
        device: str
):
    """
    评估主函数：遍历测试数据集、运行模型推理、汇总评价指标并保存可视化结果。
    由于少样本设定（k-shot）在本版本中默认关闭，因此 train_data 仅保留接口，未实际使用。
    """
    similarity_maps = []
    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []

    for (data, mask, label, name, img_type) in tqdm(test_data):
        # dataloader 返回的 data/mask/label/name 均为 batch 形式，需要逐元素展开
        for d, n, l, m in zip(data, name, label, mask):
            d = d.numpy()
            l = l.numpy()
            m = m.numpy()
            m[m > 0] = 1  # 归一化掩膜标签，确保非零都视为缺陷

            test_imgs += [d]
            names += [n]
            gt_list += [l]
            gt_mask_list += [m]

            score, appendix = model(d)  # SAA 模型返回分数与附加信息（相似度图）
            scores += [score]

            similarity_map = appendix['similarity_map']
            similarity_maps.append(similarity_map)

    # 为了便于批量绘图/指标统计，将原始推理结果统一插值到 eval_resolution
    test_imgs, scores, gt_mask_list = specify_resolution(
        test_imgs, scores, gt_mask_list,
        resolution=(resolution, resolution)
    )
    _, similarity_maps, _ = specify_resolution(
        test_imgs, similarity_maps, gt_mask_list,
        resolution=(resolution, resolution)
    )

    scores = normalize(scores)  # 缩放到 [0,1]，保证各图之间可比较
    similarity_maps = normalize(similarity_maps)

    np_scores = np.array(scores)
    img_scores = np_scores.reshape(np_scores.shape[0], -1).max(axis=1)  # 每张图用最大像素分数代表图像级异常程度

    if dataset in ['visa_challenge']:
        # ViSA Challenge 评测要求额外保存逐样本分数文件
        save_results(img_scores, scores, f'{img_dir}/..', f'{k_shot}shot', f'{experiment_indx}', names,
                     use_defect_type=True)

    if dataset in ['visa_challenge']:
        result_dict = {'i_roc': 0, 'p_roc': 0, 'p_pro': 0,
                       'i_f1': 0, 'i_thresh': 0, 'p_f1': 0, 'p_thresh': 0,
                       'r_f1': 0}
    else:
        gt_list = np.stack(gt_list, axis=0)
        # metric_cal 会返回图像级/像素级 ROC、AP 以及 PRO-AUC（可选）
        result_dict = metric_cal(np.array(scores), gt_list, gt_mask_list, cal_pro=cal_pro)

    if is_vis:
        # 选取若干样本输出原图、预测热力图与掩膜，方便人工检查
        plot_sample_cv2(
            names,
            test_imgs,
            {'SAA_plus': scores, 'Saliency': similarity_maps},
            gt_mask_list,
            save_folder=img_dir
        )

    return result_dict


def main(args):
    kwargs = vars(args)
    # argparse 返回 Namespace，这里统一转为 dict 方便在工具函数之间传递

    # prepare the experiment dir
    model_dir, img_dir, logger_dir, model_name, csv_path = get_dir_from_args(**kwargs)

    logger.info('==========running parameters=============')
    for k, v in kwargs.items():
        logger.info(f'{k}: {v}')
    logger.info('=========================================')

    # give some random seeds
    seeds = [111, 333, 999, 1111, 3333, 9999]
    kwargs['seed'] = seeds[kwargs['experiment_indx']]
    setup_seed(kwargs['seed'])

    # 根据 use_cpu / gpu_id 决定运行设备
    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"

    kwargs['device'] = device

    # get the train dataloader
    # 如果设置 k-shot>0，可在此读取少量支持集；当前默认不启用，返回 None
    if kwargs['k_shot'] > 0:
        train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)
    else:
        train_dataloader, train_dataset_inst = None, None

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    # get the model
    model = SegmentAnyAnomaly.Model(
        dino_config_file=kwargs['dino_config_file'],
        dino_checkpoint=kwargs['dino_checkpoint'],
        sam_checkpoint=kwargs['sam_checkpoint'],
        box_threshold=kwargs['box_threshold'],
        text_threshold=kwargs['text_threshold'],
        out_size=kwargs['eval_resolution'],
        device=kwargs['device'],
    )

    general_prompts = SegmentAnyAnomaly.build_general_prompts(kwargs['class_name'])
    manual_promts = SegmentAnyAnomaly.manul_prompts[kwargs['dataset']][kwargs['class_name']]

    textual_prompts = general_prompts + manual_promts

    model.set_ensemble_text_prompts(textual_prompts, verbose=False)  # 配置类别描述文本以指导 GroundingDINO

    property_text_prompts = SegmentAnyAnomaly.property_prompts[kwargs['dataset']][kwargs['class_name']]
    model.set_property_text_prompts(property_text_prompts, verbose=False)  # 进一步约束属性级提示

    model = model.to(device)

    # 执行完整评估流程：推理 -> 指标统计 -> 可视化
    metrics = eval(
        # model-related parameters
        model=model,
        train_data=train_dataloader,
        test_data=test_dataloader,

        # visual-related parameters
        resolution=kwargs['eval_resolution'],
        is_vis=True,

        # experimental parameters
        dataset=kwargs['dataset'],
        class_name=kwargs['class_name'],
        cal_pro=kwargs['cal_pro'],
        img_dir=img_dir,
        k_shot=kwargs['k_shot'],
        experiment_indx=kwargs['experiment_indx'],
        device=device
    )

    logger.info(f"\n")

    for k, v in metrics.items():
        logger.info(f"{kwargs['class_name']}======={k}: {v:.2f}")

    # 将结果追加写入 CSV，便于后续统计
    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    """解析命令行布尔字符串，如 'True'/'false'/1/0。"""
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    # data related parameters：指定要评估的数据集及类别，k-shot 仅为兼容接口
    parser.add_argument('--dataset', type=str, default='mvtec',
                        choices=['mvtec', 'visa_challenge', 'visa_public', 'ksdd2', 'mtd'])
    parser.add_argument('--class-name', type=str, default='metal_nut')
    parser.add_argument('--k-shot', type=int, default=0) # no effect... just set it to 0.

    # experiment related parameters：控制批量大小、输出目录、是否计算 PRO、一致化随机种子等
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--experiment_indx", type=int, default=0) # no effect... just set it to 0.
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", type=int, default=0)

    # method related parameters：模型结构超参与权重路径，可根据实际部署进行替换
    parser.add_argument('--eval-resolution', type=int, default=400)
    parser.add_argument("--dino_config_file", type=str,
                        default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                        help="path to config file")
    parser.add_argument(
        "--dino_checkpoint", type=str, default='weights/groundingdino_swint_ogc.pth', help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default='weights/sam_vit_h_4b8939.pth', help="path to checkpoint file"
    )

    parser.add_argument("--box_threshold", type=float, default=0.1, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.1, help="text threshold")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''  # 关闭 SSL 校验，避免在下载权重时因证书问题报错
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"  # 控制脚本只使用指定 GPU
    main(args)
