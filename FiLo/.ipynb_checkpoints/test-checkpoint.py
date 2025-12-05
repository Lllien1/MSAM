import argparse
import logging
import os
import copy
import numpy as np
import torch
from PIL import Image

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Grounding DINO
import models.GroundingDINO.groundingdino.datasets.transforms as T
from models.GroundingDINO.groundingdino.models import build_model
from models.GroundingDINO.groundingdino.util.slconfig import SLConfig
from models.GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# ======================visual (PATCH) ======================
import cv2, os, csv, json
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms
import re

# —— 可调参数 ——
TAG_MAX_WIDTH_PX = 9999   # 设大一点基本不截断；想限制宽度可改回 240/300
TAG_FONT = cv2.FONT_HERSHEY_SIMPLEX
TAG_FONT_SCALE = 0.5
TAG_THICK = 1
TAG_BG = (80, 255, 80)    # BGR
TAG_FG = (0, 0, 0)
TAG_ALPHA = 0.65
DOWNWEIGHT_OUTSIDE_BOX = 0.85  # 可视化阶段对框外的温和抑制；设为 1.0 表示不抑制

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _overlay_mask_on_img(img_bgr, mask01, alpha=0.45):
    heat = (plt.cm.jet(mask01)[:, :, :3] * 255).astype(np.uint8)  # RGB
    heat = cv2.cvtColor(heat, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(img_bgr, 1.0, heat, alpha, 0)

def _xyxy_to_int(xyxy, W, H):
    x1,y1,x2,y2 = xyxy
    return [max(0,int(x1)), max(0,int(y1)), min(W-1,int(x2)), min(H-1,int(y2))]

def _text_size(text):
    (w, h), _ = cv2.getTextSize(text, TAG_FONT, TAG_FONT_SCALE, TAG_THICK)
    return w, h

def _wrap_text(text, max_width_px):
    """把长文本按像素宽度折成多行，不再用 '???'。最后一行若仍过长，用 '...' 截断。"""
    if max_width_px >= 9000:  # 视为不限制
        return [text]
    words = text.split(' ')
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip() if cur else w
        if _text_size(test)[0] <= max_width_px:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    # 最后一行若还是太宽，做安全截断并加 '...'
    if lines and _text_size(lines[-1])[0] > max_width_px:
        s = lines[-1]
        while s and _text_size(s + "...")[0] > max_width_px:
            s = s[:-1]
        lines[-1] = (s + "...") if s else "..."
    return lines

def _draw_tag(img, x, y, label, max_width_px=TAG_MAX_WIDTH_PX, bg=TAG_BG, fg=TAG_FG, alpha=TAG_ALPHA):
    lines = _wrap_text(label, max_width_px)
    # 计算整体背景框尺寸
    line_w = max(_text_size(t)[0] for t in lines) if lines else 0
    line_h = _text_size("A")[1]  # 行高
    pad = 4
    box_w = line_w + pad*2
    box_h = line_h*len(lines) + pad*(len(lines)+1)

    H, W = img.shape[:2]
    # 尽量把框画在目标框上方；不够空间就移到下方
    top = y - box_h - 2
    if top < 1:
        top = y + 2
    left = max(1, min(x, W - box_w - 1))

    # 画半透明背景
    roi = img[top:top+box_h, left:left+box_w].copy()
    bg_rect = np.full_like(roi, bg, dtype=np.uint8)
    cv2.addWeighted(bg_rect, alpha, roi, 1-alpha, 0, dst=roi)
    img[top:top+box_h, left:left+box_w] = roi

    # 逐行写字
    by = top + pad + line_h
    for t in lines:
        cv2.putText(img, t, (left + pad, by), TAG_FONT, TAG_FONT_SCALE, fg, TAG_THICK, cv2.LINE_AA)
        by += line_h + pad

def _pos_from_box(xyxy, W, H):
    x1,y1,x2,y2 = xyxy
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    col = ["left","center","right"][min(2, max(0, int(3*cx/W)))]
    row = ["top","middle","bottom"][min(2, max(0, int(3*cy/H)))]
    return f"{row} {col}"

def _save_viz_and_logs(sample_id, img_path, img_bgr, mask01, bin_mask,
                       boxes_xyxy, phrases, pos_list, text_topk, out_dir):
    _ensure_dir(out_dir)
    H, W = mask01.shape

    # 1) 灰度热力图 & 叠加
    cv2.imwrite(os.path.join(out_dir, f"{sample_id}_mask_gray.png"),
                (mask01 * 255).astype(np.uint8))
    overlay = _overlay_mask_on_img(img_bgr, mask01)

    # 画框 + 标签（折行，不再出现 ???）
    for (x1,y1,x2,y2), phr, pos in zip(boxes_xyxy, phrases, pos_list):
        x1,y1,x2,y2 = _xyxy_to_int((x1,y1,x2,y2), W, H)
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
        clean = phr.strip().replace("#$%","")
        label = f"{clean} | {pos}"
        _draw_tag(overlay, x1, y1, label, max_width_px=TAG_MAX_WIDTH_PX)

    cv2.imwrite(os.path.join(out_dir, f"{sample_id}_overlay.png"), overlay)

    # 2) 二值 mask
    cv2.imwrite(os.path.join(out_dir, f"{sample_id}_mask_bin.png"),
                (bin_mask * 255).astype(np.uint8))

    # 3) POS+框 元数据
    with open(os.path.join(out_dir, f"{sample_id}_pos_boxes.json"), "w") as f:
        json.dump(
            [{"box":[float(x1),float(y1),float(x2),float(y2)], "phrase":phr, "pos":pos}
             for (x1,y1,x2,y2), phr, pos in zip(boxes_xyxy, phrases, pos_list)],
            f, indent=2
        )

    # 4) 文本 Top-K（短语+置信度）
    with open(os.path.join(out_dir, f"{sample_id}_text_topk.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rank","text","confidence"])
        for i, (t, s) in enumerate(text_topk):
            w.writerow([i+1, t, float(s)])
            
            
def _norm_phrase(s: str) -> str:
    """
    把 GroundingDINO 产出的短语做“等价归一化”，用于文本维度的去重：
    - 去掉我们在pred里追加的括号分数 "(0.xx)"
    - 去掉 '#$%' 标记
    - 全部转小写，去多余空格与部分标点
    - 同义词/近义拼写做简单映射（可按需扩展）
    """
    if s is None:
        return ""
    s = s.strip()
    # 去掉我们追加在末尾的“(0.xx)”分数
    s = re.sub(r"\(\s*[-+]?\d*\.?\d+\s*\)\s*$", "", s)
    # 去掉无效标记
    s = s.replace("#$%", "")
    # 统一小写
    s = s.lower()
    # 去掉多余空白
    s = re.sub(r"\s+", " ", s)
    # 去掉常见标点（保留字母数字与空格）
    s = re.sub(r"[^0-9a-z\s]", "", s)

    # 可选：非常轻量的同义词归并（按你数据可继续扩展）
    aliases = {
        "anomaly": "anomaly",
        "defect": "defect",
        "damage": "damage",
        "broken": "broken",
        "contamination": "contamination",
        "discolored": "discoloration",  # discolored -> discoloration
        "misaligned": "misalignment",
        "bent": "bent",
        "cracked": "crack",
        "cracks": "crack",
        "hole": "hole",
        "holes": "hole",
    }
    tokens = s.split()
    tokens = [aliases.get(t, t) for t in tokens]
    return " ".join(tokens).strip()


def _iou(a, b) -> float:
    """
    计算两个框的 IoU。a, b 可以是形如 [x1,y1,x2,y2] 的列表/张量。
    """
    # 转浮点
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)

# ======================visual (PATCH END) ======================



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
    
    parser.add_argument("--viz_dir", type=str, default="./viz_out", help="where to save visualizations and dumps")

    args = parser.parse_args()


    # cfg
    groundingdino_config = (
        args.groundingdino_config
    )  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
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
    for items in tqdm(test_dataloader):
        image = items["img"].to(device)
        image_path = items["img_path"][0]
        # if 'bottle' not in image_path:
        #     continue
        cls_name = items["cls_name"][0]

        results["cls_names"].append(cls_name)

        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results["imgs_masks"].append(gt_mask)  # px
        results["gt_sp"].append(items["anomaly"].item())

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
            if dataset_name == "mvtec":
                for i in range(boxes_filt.size(0)):
                    if not check_elements_in_array(mvtec_anomaly_detail_gpt[cls_name] + anomaly_status_general, pred_phrases[i]):
                        pred_phrases[i] += "#$%"
                        # boxes_filt_copy = torch.cat([boxes_filt_copy[:i], boxes_filt_copy[i+1:]])
                        continue
                    boxes_filt_copy[i] = boxes_filt_copy[i] * torch.Tensor([image_size, image_size, image_size, image_size])
                    boxes_filt_copy[i][:2] -= boxes_filt_copy[i][2:] / 2
                    boxes_filt_copy[i][2:] += boxes_filt_copy[i][:2]
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
                boxes_filt = boxes_filt_copy.cpu()


            position = []
            max_box = None
            max_pred = 0
            for i in range(boxes_filt.size(0)):
                if '#$%' in pred_phrases[i]:
                    continue
                number = float(re.search(r'\((.*?)\)', pred_phrases[i]).group(1))
                if(number >= max_pred):
                    max_box = boxes_filt[i]
                    max_pred = number
            if max_box != None:
                center = (max_box[0] + max_box[2]) / 2, (max_box[1] + max_box[3]) / 2
            else:
                center = 259, 259
            for region, ((x1, y1), (x2, y2)) in location.items():
                if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                    position.append(region)
                    break
                    
            # ==== [VIZ prep begin] ====
            # 读原图并按模型 image_size 缩放成 BGR（和 GroundingDINO 一致）
            img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"read image failed: {image_path}")
            img_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            H, W = image_size, image_size
    
            # 过滤出“有效框”（你前面的逻辑里把无效的用 '#$%' 打标了）
            valid_boxes, valid_phrases = [], []
            for i in range(boxes_filt.size(0)):
                if '#$%' in pred_phrases[i]:
                    continue
                x1,y1,x2,y2 = boxes_filt[i].tolist()
                valid_boxes.append([x1,y1,x2,y2])
                valid_phrases.append(pred_phrases[i])
    
            # 给每个有效框生成离散 POS（九宫格）
            pos_list = [_pos_from_box(b, W, H) for b in valid_boxes]
            
            # -------- A-PATCH: 去重（NMS + 文本等价 + 高 IoU 合并） BEGIN --------
            # 1) 先从 valid_phrases 解析每个框的分数
            box_scores = []
            for s in valid_phrases:
                m = re.search(r'\(([-+]?[\d.]+)\)\s*$', s)
                box_scores.append(float(m.group(1)) if m else 0.0)
            
            boxes_t = torch.tensor(valid_boxes, dtype=torch.float32)
            scores_t = torch.tensor(box_scores, dtype=torch.float32)
            
            if boxes_t.numel() > 0:
                # 2) 几何 NMS（去掉高度重叠的重复框，保留分数高的）
                keep = nms(boxes_t, scores_t, iou_threshold=0.5)  # 阈值可调 0.3~0.6
                boxes_t = boxes_t[keep]
                scores_t = scores_t[keep]
                valid_phrases = [valid_phrases[i] for i in keep.tolist()]
            
                # 3) 文本维度再聚合：短语等价且 IoU 高的只保留分数最高一个
                final_boxes, final_phrases = [], []
                used = [False] * len(valid_phrases)
                for i in range(len(valid_phrases)):
                    if used[i]:
                        continue
                    pi = _norm_phrase(valid_phrases[i])
                    best_j = i
                    best_score = scores_t[i].item()
                    for j in range(i + 1, len(valid_phrases)):
                        if used[j]:
                            continue
                        if _norm_phrase(valid_phrases[j]) != pi:
                            continue
                        if _iou(boxes_t[i], boxes_t[j]) > 0.6:  # 文本同类且几何很近
                            if scores_t[j].item() > best_score:
                                best_j = j
                                best_score = scores_t[j].item()
                            used[j] = True
                    used[i] = True
                    final_boxes.append(boxes_t[best_j].tolist())
                    final_phrases.append(valid_phrases[best_j])
            
                valid_boxes = final_boxes
                valid_phrases = final_phrases
                # 框变了 → POS 需要重算
                pos_list = [_pos_from_box(b, W, H) for b in valid_boxes]
            
                # （可选）后续你用 boxes_filt 来融合 anomaly_score，
                # 如果希望融合也使用“去重后”的框，就把 boxes_filt 同步成去重结果：
                boxes_filt = torch.tensor(valid_boxes, dtype=torch.float32)
            # -------- A-PATCH: 去重（NMS + 文本等价 + 高 IoU 合并） END --------

    
            # 从 pred_phrases 里解析“(score)”作为文本置信度，并做 Top-K
            parsed = []
            for s in pred_phrases:
                try:
                    # s 里形如 "... (0.87)"，我们把最后一个括号里的数当置信度
                    conf = float(re.search(r'\(([-+]?[\d.]+)\)\s*$', s).group(1))
                except Exception:
                    conf = 0.0
                clean = s.replace("#$%", "").strip()
                parsed.append((clean, conf))
            # 只保留有效项
            parsed = [(t,c) for (t,c) in parsed if t and t not in ("#$%")]
            parsed.sort(key=lambda x: x[1], reverse=True)
            text_topk = parsed[:10]  # 需要几条就改几条
            # ==== [VIZ prep end] ====


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

#             anomaly_score_copy = anomaly_score.clone()
#             for rect in boxes_filt:
#                 left_top_x = int(rect[0].item())
#                 left_top_y = int(rect[1].item())
#                 right_bottom_x = int(rect[2].item())
#                 right_bottom_y = int(rect[3].item())
#                 anomaly_score_copy[:, :, left_top_y:right_bottom_y, left_top_x:right_bottom_x] = 1

#             anomaly_score = torch.where(anomaly_score_copy == 1, anomaly_score, anomaly_score * 0.7)
            
            # ============== 解决：为什么有框但 mask 很弱/没有 ==============
            # 先做 0~1 归一化，确保热力图动态范围稳定
            anomaly_score = anomaly_score - anomaly_score.min()
            den = (anomaly_score.max() - anomaly_score.min() + 1e-6)
            anomaly_score = anomaly_score / den
            
            # 可选：按框做温和抑制（可把 DOWNWEIGHT_OUTSIDE_BOX 设为 1.0 关闭）
            if DOWNWEIGHT_OUTSIDE_BOX < 0.999:
                anomaly_score_copy = torch.zeros_like(anomaly_score)
                for rect in boxes_filt:
                    x1 = int(rect[0].item()); y1 = int(rect[1].item())
                    x2 = int(rect[2].item()); y2 = int(rect[3].item())
                    anomaly_score_copy[:, :, y1:y2, x1:x2] = 1.0
                anomaly_score = anomaly_score * (DOWNWEIGHT_OUTSIDE_BOX) + anomaly_score * (1.0 - DOWNWEIGHT_OUTSIDE_BOX) * anomaly_score_copy
                
            # ============== 解决：为什么有框但 mask 很弱/没有 ==============


            results["anomaly_maps"].append(
                anomaly_score.detach().cpu().numpy().reshape(image_size, image_size)
            )
            # ==== [VIZ save begin] ====
            # 取当前样本的像素级异常图（0~1）
            mask01 = anomaly_score.detach().cpu().numpy().reshape(image_size, image_size)
            # 简单阈值（你也可以换为 Otsu 或你在评测里找到的最佳阈）
            bin_mask = (mask01 >= 0.5).astype(np.uint8)
    
            sample_id = f"{cls_name}_{os.path.splitext(os.path.basename(image_path))[0]}"
            _save_viz_and_logs(
                sample_id=sample_id,
                img_path=image_path,
                img_bgr=img_bgr,
                mask01=mask01,
                bin_mask=bin_mask,
                boxes_xyxy=valid_boxes,
                phrases=valid_phrases,
                pos_list=pos_list,
                text_topk=text_topk,
                out_dir=args.viz_dir
            )
            # ==== [VIZ save end] ====
            
            

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
