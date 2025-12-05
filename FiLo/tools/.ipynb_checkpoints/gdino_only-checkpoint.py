#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GroundingDINO-only inference (Scheme A: image-level metrics only)

Usage (VisA + meta):
  python tools/gdino_only.py \
    --dataset visa \
    --data_path "$FILO_ROOT/data/VisA_20220922" \
    --use_meta \
    --grounded_checkpoint "$FILO_ROOT/ckpt/grounding_train_on_mvtec.pth" \
    --groundingdino_config "$FILO_ROOT/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
    --device cuda \
    --viz_dir "$FILO_ROOT/viz_gdino_visa"

Environment (建议):
  export PYTHONPATH="$FILO_ROOT/models/GroundingDINO:$FILO_ROOT:${PYTHONPATH:-}"
"""

import os
import re
import cv2
import json
import copy
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# metrics (image-level only)
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# GroundingDINO imports
import models.GroundingDINO.groundingdino.datasets.transforms as T
from models.GroundingDINO.groundingdino.models import build_model
from models.GroundingDINO.groundingdino.util.slconfig import SLConfig
from models.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# --------------------------
# Utils
# --------------------------
def _ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _parse_conf_from_phrase(s: str) -> float:
    """
    从短语尾部提取括号分数: "... something (0.87)" -> 0.87
    失败返回 0.0
    """
    m = re.search(r'\(([-+]?[0-9]*\.?[0-9]+)\)\s*$', s or "")
    return float(m.group(1)) if m else 0.0

def _boxes_cxcywh_to_xyxy(boxes_cxcywh, wh):
    """ boxes: Nx4 in cxcywh (normalized); return Nx4 xyxy in pixels """
    W, H = wh
    xyxy = []
    for cx,cy,w,h in boxes_cxcywh:
        x1 = (cx - w/2.0) * W
        y1 = (cy - h/2.0) * H
        x2 = (cx + w/2.0) * W
        y2 = (cy + h/2.0) * H
        xyxy.append([float(x1), float(y1), float(x2), float(y2)])
    return xyxy

def _xyxy_to_int(b, W, H):
    x1,y1,x2,y2 = b
    return [max(0,int(x1)), max(0,int(y1)), min(W-1,int(x2)), min(H-1,int(y2))]

def _save_boxes_overlay(sample_id, img_bgr, boxes_xyxy, phrases, out_dir):
    _ensure_dir(out_dir)
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    for (x1,y1,x2,y2), phr in zip(boxes_xyxy, phrases):
        x1,y1,x2,y2 = _xyxy_to_int((x1,y1,x2,y2), W, H)
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
        # 文本太长就裁一下
        clean = phr.strip()
        if len(clean) > 40:
            clean = clean[:37] + "..."
        y_text = y1 - 6 if y1 - 6 > 10 else y1 + 15
        cv2.putText(overlay, clean, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(out_dir, f"{sample_id}_gdino_overlay.png"), overlay)
    with open(os.path.join(out_dir, f"{sample_id}_gdino_boxes.json"), "w") as f:
        json.dump([{"box":[float(x1),float(y1),float(x2),float(y2)], "phrase":phr}
                  for (x1,y1,x2,y2),phr in zip(boxes_xyxy, phrases)], f, indent=2)

def _build_prompt(cls_name: str, dataset: str) -> str:
    """
    根据数据集构造一个缺省 prompt；你可以按需改得更细。
    """
    cls_name = cls_name.replace("_", " ")
    if dataset.lower() == "mvtec":
        general = ["anomaly", "damage", "broken", "defect", "contamination"]
        detail = [f"abnormal {cls_name}"]
        txt = " . ".join(general + detail)
    else:
        # VisA 简版
        general = ["anomaly", "defect", "contamination", "crack", "scratch", "dent"]
        detail = [f"abnormal {cls_name}"]
        txt = " . ".join(general + detail)
    if not txt.endswith("."):
        txt += "."
    return txt

def _load_image_for_gdino(image_path, image_size):
    """Return (PIL-like) image in tensor format expected by GroundingDINO transforms."""
    from PIL import Image
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([image_size, image_size], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def _load_gdino(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    ckpt = torch_load(model_checkpoint_path, map_location="cpu")
    _ = model.load_state_dict(clean_state_dict(ckpt), strict=False)
    model.eval()
    return model

def torch_load(path, map_location="cpu"):
    import torch
    return torch.load(path, map_location=map_location)

def _grounding_infer(model, image_t, caption, box_thr, text_thr, area_thr, device="cuda"):
    """
    输入：model (GroundingDINO), image_t (3xHxW tensor), caption(str)。
    输出：过滤后 boxes_cxcywh (tensor Nx4, normalized), phrases(list[str])
    """
    import torch
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    model = model.to(device)
    image_t = image_t.to(device)

    with torch.no_grad():
        outputs = model(image_t[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid().cpu()[0]  # (nq, 256)
    boxes  = outputs["pred_boxes"].cpu()[0]             # (nq, 4) cxcywh (normalized)

    # 过滤：置信度阈值 + 面积阈值
    boxes_area = boxes[:,2] * boxes[:,3]
    keep = (logits.max(dim=1)[0] > box_thr) & (boxes_area < area_thr)

    if keep.sum().item() == 0:
        # 兜底：取分数最大的一个
        idx = torch.argmax(logits.max(dim=1)[0])
        logits_f = logits[idx:idx+1]
        boxes_f  = boxes[idx:idx+1]
    else:
        logits_f = logits[keep]
        boxes_f  = boxes[keep]

    # 构造短语
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    phrases = []
    out_boxes = []
    for lg, bx in zip(logits_f, boxes_f):
        phr = get_phrases_from_posmap(lg > text_thr, tokenized, tokenizer)
        phrases.append(phr + f"({str(lg.max().item())[:4]})")
        out_boxes.append(bx)
    if len(out_boxes) == 0:
        return boxes_f, []
    return (torch_stack(out_boxes, dim=0), phrases)

def torch_stack(seq, dim=0):
    import torch
    return torch.stack(seq, dim=dim)


# --------------------------
# Metrics (image-level only)
# --------------------------
def _report_img_level(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float32)
    if len(y_true) == 0:
        return None, None, None
    auroc = roc_auc_score(y_true, y_score)
    ap    = average_precision_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    f1 = (2*prec*rec) / (prec+rec+1e-12)
    f1_max = np.max(f1[np.isfinite(f1)]) if np.any(np.isfinite(f1)) else 0.0
    return auroc, f1_max, ap


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser("GroundingDINO-only inference", add_help=True)
    parser.add_argument("--groundingdino_config", type=str, required=True,
                        help="path to GroundingDINO config .py")
    parser.add_argument("--grounded_checkpoint", type=str, required=True,
                        help="path to GDINO checkpoint .pth / .safetensors")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--use_meta", action="store_true", help="use <data_path>/meta.json")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--area_threshold", type=float, default=0.7)
    parser.add_argument("--viz_dir", type=str, default="./viz_gdino")

    # 数据来源
    parser.add_argument("--image", type=str, default="", help="single image path (debug)")
    parser.add_argument("--dataset", type=str, default="visa", choices=["mvtec","visa"])
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--class_name", type=str, default="", help="optional class hint")
    parser.add_argument("--prompt", type=str, default="", help="manual prompt")

    args = parser.parse_args()

    # lazy import torch to avoid env noise before parsing
    import torch

    # 1) load model
    model = _load_gdino(args.groundingdino_config, args.grounded_checkpoint, args.device)

    # 2) collectors (image-level only)
    y_true_all, y_score_all = [], []
    per_cls_true = defaultdict(list)
    per_cls_score = defaultdict(list)

    # 3) data source
    items = []
    if args.use_meta:
        meta_fp = os.path.join(args.data_path, "meta.json")
        meta = json.load(open(meta_fp, "r"))
        print(f"[INFO] meta loaded: {len(meta)} records")
        for rec in meta:
            img_rel = rec["img_path"]
            cls_name = rec.get("cls_name", "unknown")
            gt = int(rec.get("anomaly", 0))
            items.append((os.path.join(args.data_path, img_rel), cls_name, gt))
    elif args.image:
        assert os.path.exists(args.image), f"not found: {args.image}"
        cls_name = args.class_name or "unknown"
        items.append((args.image, cls_name, 0))  # 无标签
    else:
        # 简单的文件夹遍历 (没有 GT 标签，不做评测)
        # VisA: <cls>/Data/Images/{Normal,Anomaly}/*.JPG
        # MVTec: <cls>/test/...
        if args.dataset.lower() == "visa":
            for cls in sorted(os.listdir(args.data_path)):
                cls_dir = os.path.join(args.data_path, cls, "Data", "Images")
                if not os.path.isdir(cls_dir):
                    continue
                for root, _, files in os.walk(cls_dir):
                    for fn in files:
                        if fn.lower().endswith((".png",".jpg",".jpeg","bmp","tif","tiff")):
                            items.append((os.path.join(root, fn), cls, None))
        else:
            # mvtec
            for cls in sorted(os.listdir(args.data_path)):
                test_root = os.path.join(args.data_path, cls, "test")
                if not os.path.isdir(test_root):
                    continue
                for root, _, files in os.walk(test_root):
                    for fn in files:
                        if fn.lower().endswith((".png",".jpg",".jpeg","bmp","tif","tiff")):
                            items.append((os.path.join(root, fn), cls, None))

    # 4) iterate
    pbar = tqdm(items, desc="GDINO(meta)" if args.use_meta else "GDINO")
    for img_path, cls_name, gt in pbar:
        # 4.1 read & resize (for saving overlay)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] read fail: {img_path}")
            continue
        img_bgr = cv2.resize(img_bgr, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)

        # 4.2 prepare tensor for GDINO
        _, img_t = _load_image_for_gdino(img_path, args.image_size)

        # 4.3 prompt
        prompt = args.prompt or _build_prompt(cls_name, args.dataset)

        # 4.4 infer
        boxes_cxcywh, phrases = _grounding_infer(
            model, img_t, prompt,
            args.box_threshold, args.text_threshold, args.area_threshold,
            device=args.device
        )
        # 4.5 save overlay
        boxes_xyxy = _boxes_cxcywh_to_xyxy(boxes_cxcywh.cpu().numpy(), (args.image_size, args.image_size))
        sample_id = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(args.viz_dir, args.dataset, cls_name.replace(" ", "_"))
        _save_boxes_overlay(sample_id, img_bgr, boxes_xyxy, phrases, out_dir)

        # 4.6 image-level score (max conf among phrases)
        img_score = max((_parse_conf_from_phrase(p) for p in phrases), default=0.0)

        # 4.7 collect metrics only when gt is available (i.e., use_meta)
        if gt is not None:
            y_true_all.append(int(gt))
            y_score_all.append(float(img_score))
            per_cls_true[cls_name].append(int(gt))
            per_cls_score[cls_name].append(float(img_score))

    # 5) report (image-level only)
    rows = []
    if len(y_true_all) > 0:
        # by class
        for cls in sorted(per_cls_true.keys()):
            auroc, f1, ap = _report_img_level(per_cls_true[cls], per_cls_score[cls])
            if auroc is None:
                continue
            rows.append([cls, f"{auroc*100:.1f}", f"{f1*100:.1f}", f"{ap*100:.1f}"])
        # mean
        auroc_m, f1_m, ap_m = _report_img_level(y_true_all, y_score_all)
        if auroc_m is not None:
            rows.append(["mean", f"{auroc_m*100:.1f}", f"{f1_m*100:.1f}", f"{ap_m*100:.1f}"])

        # pretty print
        try:
            from tabulate import tabulate
            print("\n[GroundingDINO-only] Image-level metrics:\n")
            print(tabulate(rows, headers=["objects","auroc_sp","f1_sp","ap_sp"], tablefmt="pipe"))
        except Exception:
            print("\n[GroundingDINO-only] Image-level metrics:")
            print("| objects | auroc_sp | f1_sp | ap_sp |")
            print("|:--------|---------:|------:|------:|")
            for r in rows:
                print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    else:
        print("\n[INFO] No GT found -> skip metrics (use --use_meta with meta.json to evaluate).")

    print(f"[OK] all done, outputs in: {args.viz_dir}")


if __name__ == "__main__":
    main()
