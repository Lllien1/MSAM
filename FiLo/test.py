import argparse
import logging
import os
import copy
import numpy as np
import torch
from PIL import Image
import time
import cv2
import json
import csv
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Grounding DINO
import models.GroundingDINO.groundingdino.datasets.transforms as T
from models.GroundingDINO.groundingdino.models import build_model
from models.GroundingDINO.groundingdino.util.slconfig import SLConfig
from models.GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)


import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import torchvision
import threading
import re

import torchvision.transforms as transforms
from tabulate import tabulate
from sklearn.metrics import (
    auc,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from sklearn.manifold import TSNE

import torch

import argparse
import copy
import torch.nn.functional as F
from datasets.mvtec_supervised import MVTecDataset
from datasets.visa_supervised import VisaDataset
import models.vv_open_clip as open_clip
from PIL import Image
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from matplotlib import pyplot as plt
from models.FiLo import FiLo

class DataLoaderX(torch.utils.data.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

status_abnormal_winclip = [
    "damaged {}",
    "broken {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]

anomaly_status_general = ["anomaly", "damage", "broken", "defect", "contamination"]

mvtec_anomaly_detail_gpt = {
    "carpet": "discoloration in a specific area,irregular patch or section with a different texture,frayed edges or unraveling fibers,burn mark or scorching",
    "grid": "crooked,cracks,excessive gaps,discoloration,deformation,missing,inconsistent spacing between grid elements,corrosion,visible signs,chipping",
    "leather": "scratches,discoloration,creases,uneven texture,tears,brittleness,damage,seams,heat damage,mold",
    "tile": "chipped,irregularities,discoloration,efflorescence,warping,missing,depressions,lippage,fungus,damage",
    "wood": "knots,warping,cracks along the grain,mold growth on the surface,staining from water damage,wood rot,woodworm holes,rough patches,protruding knots",
    "bottle": "cracked large,cracked small,dented large,dented small,leaking,discolored,deformed,missing cap,excessive condensation,unusual odor",
    "cable": "twisted,knotted cable strands,detached connectors,excessive stretching,dents,corrosion,scorching along the cable,exposed conductive material",
    "capsule": "irregular shape,discoloration coloring,crinkled,uneven seam,condensation inside the capsule,foreign particles,unusually soft or hard",
    "hazelnut": "fungal growth,unusual discoloration,rotten or foul odor emanating,insect infestation,wetness,misshapen shell,unusually thin,contaminants,unusual texture",
    "metal nut": "cracks,irregular threading,corrosion,missing,distortion,signs of discoloration,excessive wear on contact surfaces,inconsistent texture",
    "pill": "irregular shape,crumbling texture,excessive powder,Uneven coating,presence of air bubbles,disintegration,abnormal specks",
    "screw": "rust on the surface,bent,damaged threads,stripped threads,deformed top,coating damage,uneven grooves,inconsistent size",
    "toothbrush": "loose bristles,uneven bristle distribution,excessive shedding of bristles,staining on the bristles,abrasive texture,irregularities in the shape",
    "transistor": "burn marks,detached leads,signs of corrosion,irregularities in the shape,presence of cracks or fractures,signs of physical trauma,irregularities in the surface texture",
    "zipper": "bent,frayed,misaligned,excessive stiffness,corroded,detaches,loose,warped",
}

visa_anomaly_detail_gpt = {
    "candle": "cracks or fissures in the wax,Wax pooling unevenly around the wick,tunneling,incomplete wax melt pool,irregular or flickering flame,other,extra wax in candle,wax melded out of the candle",
    "capsules": "uneven capsule size,capsule shell appears brittle,excessively soft,dents,condensation,irregular seams or joints,specks",
    "cashew": "uneven coloring,fungal growth,presence of foreign objects,unusual texture,empty shells,signs of moisture,stuck together",
    "chewinggum": "consistency,presence of foreign objects,uneven coloring,excessive hardness,similar colour spot",
    "fryum": "irregular shape,unusual odor,uneven coloring,unusual texture,small scratches,different colour spot,fryum stuck together,other",
    "macaroni1": "uneven shape ,small scratches,small cracks,uneven coloring,signs of insect infestation,uneven texture,Unusual consistency",
    "macaroni2": "irregular shape,small scratches,presence of foreign particles,excessive moisture,Signs of infestation,small cracks,unusual texture",
    "pcb1": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb2": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb3": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb4": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pipe fryum": "uneven shape,presence of foreign objects,different colour spot,unusual odor,empty interior,unusual texture,similar colour spot,stuck together",
}

for cls_name in mvtec_anomaly_detail_gpt.keys():
    mvtec_anomaly_detail_gpt[cls_name] = (
        mvtec_anomaly_detail_gpt[cls_name].split(",")
    )

for cls_name in visa_anomaly_detail_gpt.keys():
    visa_anomaly_detail_gpt[cls_name] = (
        visa_anomaly_detail_gpt[cls_name].split(",")
    )

status_abnormal = {}

for cls_name in mvtec_anomaly_detail_gpt.keys():
    status_abnormal[cls_name] = ['abnormal {} ' + 'with {}'.format(x) for x in mvtec_anomaly_detail_gpt[cls_name]] 

for cls_name in visa_anomaly_detail_gpt.keys():
    status_abnormal[cls_name] = ['abnormal {} ' + 'with {}'.format(x) for x in visa_anomaly_detail_gpt[cls_name]]

mvtec_obj_list = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

visa_obj_list = [
    "candle",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "capsules",
]

location = {"top left": [(0, 0),(172, 172)],
                "top": [(173, 0),(344, 172)],
                "top right": [(345, 0), (517, 172)],
                "left": [(0, 173), (172, 344)],
                "center": [(173, 173), (344, 344)],
                "right": [(345, 173), (517, 344)],
                "bottom left": [(0, 345), (172, 517)],
                "bottom": [(173, 345), (344, 517)],
                "bottom right": [(345, 345), (517, 517)]}


def save_visualization(image_path, mask, boxes, cls_name, viz_dir, image_size):
    if not viz_dir:
        return
    os.makedirs(viz_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))

    mask_norm = mask - mask.min()
    mask_norm = mask_norm / (mask_norm.max() + 1e-6)
    heatmap = (plt.cm.jet(mask_norm)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    if torch.is_tensor(boxes):
        boxes_iter = boxes.cpu().numpy()
    else:
        boxes_iter = boxes
    for rect in boxes_iter:
        x1 = int(rect[0])
        y1 = int(rect[1])
        x2 = int(rect[2])
        y2 = int(rect[3])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    base = f"{cls_name}_{os.path.splitext(os.path.basename(image_path))[0]}"
    cv2.imwrite(os.path.join(viz_dir, f"{base}_orig.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(viz_dir, f"{base}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(viz_dir, f"{base}_mask.png"), (mask_norm * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(viz_dir, f"{base}_mask_bin.png"), ((mask_norm >= 0.5) * 255).astype(np.uint8))

def save_grounding_phrases(image_path, cls_name, boxes, phrases, viz_dir):
    if not viz_dir or not phrases:
        return
    base = f"{cls_name}_{os.path.splitext(os.path.basename(image_path))[0]}"
    records = []
    for rect, phrase in zip(boxes, phrases):
        rect_np = rect if isinstance(rect, (list, tuple)) else rect.tolist()
        records.append(
            {
                "phrase": phrase,
                "box": [float(x) for x in rect_np],
            }
        )
    json_path = os.path.join(viz_dir, f"{base}_phrases.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    transform = T.Compose(
        [
            T.RandomResize([image_size, image_size], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def cal_pro_score(obj, masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])

    return pro_auc


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint), strict=False)
    print(load_res)
    _ = model.eval()
    return model

# gaussion_filter = 

def get_grounding_output(
    model,
    image,
    caption,
    box_threshold,
    text_threshold,
    category,
    with_logits=True,
    device="cpu",
    area_thr=0.8,
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    boxes_area = boxes_filt[:, 2] * boxes_filt[:, 3]
    filt_mask = torch.bitwise_and(
        (logits_filt.max(dim=1)[0] > box_threshold), (boxes_area < area_thr)
    )

    if torch.sum(filt_mask) == 0:  # in case there are no matches
        filt_mask = torch.argmax(logits_filt.max(dim=1)[0])
        logits_filt = logits_filt[filt_mask].unsqueeze(0)  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask].unsqueeze(0)
    else:
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    # build pred
    pred_phrases = []
    boxes_filt_category = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        boxes_filt_category.append(box)
    boxes_filt_category = torch.stack(boxes_filt_category, dim=0)

    return boxes_filt_category, pred_phrases


def apply_boxes_torch(boxes, original_size):
    boxes = boxes.reshape(-1, 2, 2)
    old_h, old_w = original_size
    scale = image_size * 1.0 / max(original_size[0], original_size[1])
    newh, neww = original_size[0] * scale, original_size[1] * scale
    new_w = int(neww + 0.5)
    new_h = int(newh + 0.5)
    boxes = copy.deepcopy(boxes).to(torch.float)
    boxes[..., 0] = boxes[..., 0] * (new_w / old_w)
    boxes[..., 1] = boxes[..., 1] * (new_h / old_h)
    return boxes.reshape(-1, 4)


def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
    ax.text(x0, y0, label)


def check_elements_in_array(arr1, arr2):
    at = False
    for elem in arr1:
        if elem in arr2:
            at = True
            break
    return at


def cal_score(obj):
    table = []
    gt_px = []
    pr_px = []
    gt_sp = []
    pr_sp = []
    pr_sp_tmp = []
    table.append(obj)
    # print(results['cls_names'])
    for idxes in range(len(results["cls_names"])):
        if results["cls_names"][idxes] == obj:
            gt_px.append(results["imgs_masks"][idxes].squeeze(1).numpy())
            pr_px.append(results["anomaly_maps"][idxes])
            gt_sp.append(results["gt_sp"][idxes])
            pr_sp.append(results["pr_sp"][idxes])
    gt_px = np.array(gt_px)
    gt_sp = np.array(gt_sp)
    pr_px = np.array(pr_px)
    pr_sp = np.array(pr_sp)


    auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
    auroc_sp = roc_auc_score(gt_sp, pr_sp)
    ap_sp = average_precision_score(gt_sp, pr_sp)
    ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
    # f1_sp
    precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
    # f1_px
    precisions, recalls, thresholds = precision_recall_curve(
        gt_px.ravel(), pr_px.ravel()
    )
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
    # aupro
    if len(gt_px.shape) == 4:
        gt_px = gt_px.squeeze(1)
    if len(pr_px.shape) == 4:
        pr_px = pr_px.squeeze(1)
    aupro = cal_pro_score(obj, gt_px, pr_px)

    table.append(str(np.round(auroc_px * 100, decimals=1)))
    table.append(str(np.round(f1_px * 100, decimals=1)))
    table.append(str(np.round(ap_px * 100, decimals=1)))
    table.append(str(np.round(aupro * 100, decimals=1)))

    table.append(str(np.round(auroc_sp * 100, decimals=1)))
    table.append(str(np.round(f1_sp * 100, decimals=1)))
    table.append(str(np.round(ap_sp * 100, decimals=1)))


    table_ls.append(table)
    auroc_sp_ls.append(auroc_sp)
    auroc_px_ls.append(auroc_px)
    f1_sp_ls.append(f1_sp)
    f1_px_ls.append(f1_px)
    aupro_ls.append(aupro)
    ap_sp_ls.append(ap_sp)
    ap_px_ls.append(ap_px)




if __name__ == "__main__":

    parser = argparse.ArgumentParser("Test", add_help=True)
    parser.add_argument(
        "--groundingdino_config",
        type=str,
        default="./models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="path to config file",
    )
    parser.add_argument(
        "--grounded_checkpoint",
        type=str,
        default="./grounding_weight/grounding_mvtec.pth",
        help="path to checkpoint file",
    )

    parser.add_argument(
        "--clip_model", type=str, default="ViT-L-14-336", help="model used"
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="openai",
        help="pretrained weight used",
    )

    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument(
        "--features_list",
        type=int,
        nargs="+",
        default=[6, 12, 18, 24],
        help="features used",
    )

    parser.add_argument(
        "--dataset", type=str, default="mvtec", help="train dataset name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/path/to/data/mvtec",
        help="path to test dataset",
    )

    parser.add_argument(
        "--box_threshold", type=float, default=0.25, help="box threshold"
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25, help="text threshold"
    )
    parser.add_argument(
        "--area_threshold", type=float, default=0.7, help="defect area threshold"
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="running on cpu only!, default=False"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="default", help="ckpt_path"
    )

    parser.add_argument("--n_ctx", type=int, default=12, help="n_ctx")

    parser.add_argument(
        "--viz_dir",
        type=str,
        default="",
        help="directory to save visualization outputs (disabled if empty)",
    )
    parser.add_argument(
        "--textvec_dir",
        type=str,
        default="",
        help="directory to dump per-sample text vectors and t-SNE visualization",
    )

    args = parser.parse_args()


    # cfg
    groundingdino_config = (
        args.groundingdino_config
    )  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    area_threshold = args.area_threshold

    dataset_name = args.dataset
    dataset_dir = args.data_path
    device = args.device

    image_size = args.image_size
    save_path = args.ckpt_path.split("/")[-2]

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, f"{dataset_name}_log.txt")

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger("test")
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")


    positions_list = ['top left', 'top', 'top right', 'left', 'center', 'right', 'bottom left', 'bottom', 'bottom right']


    # load model
    model = load_model(groundingdino_config, grounded_checkpoint, device=device)  # DINO
    
    _, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, image_size, pretrained=args.clip_pretrained
    )

    # dataset
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    gaussion_filter = torchvision.transforms.GaussianBlur(3, 4.0)

    if dataset_name == "mvtec":
        test_data = MVTecDataset(
            root=dataset_dir,
            transform=preprocess,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    else:
        test_data = VisaDataset(
            root=dataset_dir,
            transform=preprocess,
            target_transform=transform,
            mode="test",
        )

    test_dataloader = DataLoaderX(
        test_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )
    
    obj_list = [x.replace("_", " ") for x in test_data.get_cls_names()]

    filo_model = FiLo(obj_list, args, device).to(device)

    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path)

    filo_model.load_state_dict(ckpt["filo"], strict=False)


    
    results = {}
    results["cls_names"] = []
    results["imgs_masks"] = []
    results["anomaly_maps"] = []
    results["gt_sp"] = []
    results["pr_sp"] = []

    total_infer_time = 0.0
    processed_images = 0
    textvec_vectors = []
    textvec_meta = []

    for items in tqdm(test_dataloader):
        start_time = time.perf_counter()
        image = items["img"].to(device)
        image_path = items["img_path"][0]
        # if 'bottle' not in image_path:
        #     continue
        cls_name = items["cls_name"][0]

        results["cls_names"].append(cls_name)

        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results["imgs_masks"].append(gt_mask)  # px
        label = int(items["anomaly"].item())
        results["gt_sp"].append(label)

        with torch.no_grad():
            # run grounding dino model
            if dataset_name == "mvtec":
                text_prompt =  " . ".join(anomaly_status_general + mvtec_anomaly_detail_gpt[cls_name])
            else:
                text_prompt =  " . ".join(anomaly_status_general + visa_anomaly_detail_gpt[cls_name])

            # print(text_prompt)
            _, image_dino = load_image(image_path)

            boxes_filt, pred_phrases = get_grounding_output(
                model, image_dino, text_prompt, box_threshold, text_threshold, category=cls_name, device='cuda', area_thr=area_threshold
            )
            
            # 删除不是异常的矩形框
            boxes_filt_copy = copy.deepcopy(boxes_filt)
            valid_boxes = []
            valid_phrases = []
            if dataset_name == "mvtec":
                for i in range(boxes_filt.size(0)):
                    if not check_elements_in_array(mvtec_anomaly_detail_gpt[cls_name] + anomaly_status_general, pred_phrases[i]):
                        pred_phrases[i] += "#$%"
                        # boxes_filt_copy = torch.cat([boxes_filt_copy[:i], boxes_filt_copy[i+1:]])
                        continue
                    boxes_filt_copy[i] = boxes_filt_copy[i] * torch.Tensor([image_size, image_size, image_size, image_size])
                    boxes_filt_copy[i][:2] -= boxes_filt_copy[i][2:] / 2
                    boxes_filt_copy[i][2:] += boxes_filt_copy[i][:2]
                    valid_boxes.append(boxes_filt_copy[i].cpu())
                    valid_phrases.append(pred_phrases[i])
                boxes_filt = boxes_filt_copy.cpu()
            else:
                for i in range(boxes_filt.size(0)):
                    if not check_elements_in_array(visa_anomaly_detail_gpt[cls_name] + anomaly_status_general, pred_phrases[i]):
                        pred_phrases[i] += "#$%"
                        # boxes_filt_copy = torch.cat([boxes_filt_copy[:i], boxes_filt_copy[i+1:]])
                        continue
                    boxes_filt_copy[i] = boxes_filt_copy[i] * torch.Tensor([image_size, image_size, image_size, image_size])
                    boxes_filt_copy[i][:2] -= boxes_filt_copy[i][2:] / 2
                    boxes_filt_copy[i][2:] += boxes_filt_copy[i][:2]
                    valid_boxes.append(boxes_filt_copy[i].cpu())
                    valid_phrases.append(pred_phrases[i])
                boxes_filt = boxes_filt_copy.cpu()

            if len(valid_boxes) > 0:
                boxes_selected = torch.stack(valid_boxes)
            else:
                boxes_selected = torch.empty((0, 4))

            position = []
            max_box = None
            max_pred = 0
            for box_tensor, phrase in zip(boxes_selected, valid_phrases):
                number = float(re.search(r'\((.*?)\)', phrase).group(1))
                if(number >= max_pred):
                    max_box = box_tensor
                    max_pred = number
            if max_box != None:
                center = (max_box[0] + max_box[2]) / 2, (max_box[1] + max_box[3]) / 2
            else:
                center = 259, 259
            for region, ((x1, y1), (x2, y2)) in location.items():
                if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                    position.append(region)
                    break


        with torch.no_grad():
            text_probs, anomaly_maps = filo_model(items, with_adapter=True, positions = position)

            for i in range(len(anomaly_maps)):
                anomaly_maps[i] = gaussion_filter(
                        (anomaly_maps[i][:, 1, :, :] - anomaly_maps[i][:, 0, :, :] + 1) / 2
                    )

            anomaly_map_ret = torch.mean(
                torch.stack(anomaly_maps, dim=0), dim=0
            ).unsqueeze(1)

            results["pr_sp"].append((text_probs.flatten()[1].item() + anomaly_map_ret.max().item()) / 2)

            anomaly_score = anomaly_map_ret

            anomaly_score_copy = anomaly_score.clone()
            for rect in boxes_selected:
                left_top_x = int(rect[0].item())
                left_top_y = int(rect[1].item())
                right_bottom_x = int(rect[2].item())
                right_bottom_y = int(rect[3].item())
                anomaly_score_copy[:, :, left_top_y:right_bottom_y, left_top_x:right_bottom_x] = 1

            anomaly_score = torch.where(anomaly_score_copy == 1, anomaly_score, anomaly_score * 0.7)


            mask_np = anomaly_score.detach().cpu().numpy().reshape(image_size, image_size)
            results["anomaly_maps"].append(mask_np)
            save_visualization(
                image_path,
                mask_np,
                boxes_selected,
                cls_name,
                args.viz_dir,
                image_size,
            )
            save_grounding_phrases(
                image_path,
                cls_name,
                boxes_selected,
                valid_phrases,
                args.viz_dir,
            )
            if args.textvec_dir:
                normal_vec = (
                    filo_model.last_normal_text_feature.squeeze().detach().cpu().numpy()
                )
                abnormal_vec = (
                    filo_model.last_abnormal_text_feature.squeeze().detach().cpu().numpy()
                )
                textvec_vectors.append(normal_vec)
                textvec_meta.append(
                    {
                        "cls": cls_name,
                        "image_path": image_path,
                        "label": label,
                        "type": "normal",
                    }
                )
                textvec_vectors.append(abnormal_vec)
                textvec_meta.append(
                    {
                        "cls": cls_name,
                        "image_path": image_path,
                        "label": label,
                        "type": "abnormal",
                    }
                )

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        total_infer_time += elapsed
        processed_images += 1
        fps = 1.0 / elapsed if elapsed > 0 else float("inf")
        logger.info(
            f"[Timing] {cls_name} - {os.path.basename(image_path)}: {elapsed:.3f}s ({fps:.1f} FPS)"
        )

    if processed_images > 0:
        avg = total_infer_time / processed_images
        avg_fps = 1.0 / avg if avg > 0 else float("inf")
        logger.info(
            f"[Timing] Average per-image time: {avg:.3f}s ({avg_fps:.1f} FPS)"
        )
    if args.textvec_dir and textvec_vectors:
        os.makedirs(args.textvec_dir, exist_ok=True)
        vectors_array = np.stack(textvec_vectors, axis=0)
        vec_path = os.path.join(args.textvec_dir, "text_vectors.npy")
        np.save(vec_path, vectors_array)
        meta_path = os.path.join(args.textvec_dir, "text_vectors_meta.csv")
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "cls", "image_path", "label", "type"])
            for idx, meta in enumerate(textvec_meta):
                writer.writerow(
                    [idx, meta["cls"], meta["image_path"], meta["label"], meta["type"]]
                )
        if vectors_array.shape[0] > 2:
            perplexity = min(30, vectors_array.shape[0] - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, init="random")
            tsne_coords = tsne.fit_transform(vectors_array)
            np.save(os.path.join(args.textvec_dir, "text_vectors_tsne.npy"), tsne_coords)
            plt.figure(figsize=(6, 5))
            type_to_style = {"normal": ("tab:blue", "o"), "abnormal": ("tab:red", "x")}
            for vec_type, (color, marker) in type_to_style.items():
                idxs = [i for i, meta in enumerate(textvec_meta) if meta["type"] == vec_type]
                if not idxs:
                    continue
                coords = tsne_coords[idxs]
                plt.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    label=vec_type,
                    s=20,
                    marker=marker,
                    c=color,
                    alpha=0.8,
                )
            labeled_pairs = set()
            for idx, meta in enumerate(textvec_meta):
                cls_name = meta["cls"]
                vec_type = meta["type"]
                pair_key = (cls_name, vec_type)
                if pair_key in labeled_pairs:
                    continue
                same_idxs = [
                    j
                    for j, m in enumerate(textvec_meta)
                    if m["cls"] == cls_name and m["type"] == vec_type
                ]
                if not same_idxs:
                    continue
                points = tsne_coords[same_idxs]
                centroid = points.mean(axis=0)
                plt.text(
                    centroid[0],
                    centroid[1],
                    cls_name,
                    color="black",
                    fontsize=8,
                    ha="center",
                    va="center",
                )
                labeled_pairs.add(pair_key)
            plt.legend()
            plt.title("Text Feature t-SNE")
            plt.tight_layout()
            plt.savefig(os.path.join(args.textvec_dir, "text_vectors_tsne.png"), dpi=300)
            plt.close()

    # metrics
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_ls = []
    ap_sp_ls = []
    ap_px_ls = []

    threads = [None] * 20
    idx = 0
    for obj in tqdm(obj_list):
 
        threads[idx] = threading.Thread(target=cal_score, args=(obj, ))
        threads[idx].start()
        idx += 1

    for i in range(idx):
        threads[i].join()

        

    # logger
    table_ls.append(
        [
            "mean",
            str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
            str(np.round(np.mean(f1_px_ls) * 100, decimals=1)),
            str(np.round(np.mean(ap_px_ls) * 100, decimals=1)),
            str(np.round(np.mean(aupro_ls) * 100, decimals=1)),
            str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
            str(np.round(np.mean(f1_sp_ls) * 100, decimals=1)),
            str(np.round(np.mean(ap_sp_ls) * 100, decimals=1)),
        ]
    )
    results = tabulate(
        table_ls,
        headers=[
            "objects",
            "auroc_px",
            "f1_px",
            "ap_px",
            "aupro",
            "auroc_sp",
            "f1_sp",
            "ap_sp",
        ],
        tablefmt="pipe",
    )
    logger.info("\n%s", results)
