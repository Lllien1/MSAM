#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, csv, json, argparse
import numpy as np
import cv2
import torch
from PIL import Image


# ==== GroundingDINO 基础依赖（用工程里自带的） ====
import models.GroundingDINO.groundingdino.datasets.transforms as T
from models.GroundingDINO.groundingdino.models import build_model
from models.GroundingDINO.groundingdino.util.slconfig import SLConfig
from models.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# ---------- 小工具 ----------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def overlay_mask_on_img(img_bgr, mask01, alpha=0.45):
    import matplotlib.pyplot as plt
    heat = (plt.cm.jet(mask01)[:, :, :3] * 255).astype(np.uint8)  # RGB
    heat = cv2.cvtColor(heat, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(img_bgr, 1.0, heat, alpha, 0)

def xyxy_to_int(xyxy, W, H):
    x1,y1,x2,y2 = xyxy
    return [max(0,int(x1)), max(0,int(y1)), min(W-1,int(x2)), min(H-1,int(y2))]

def short_phrase(s, max_chars=32):
    # 去掉最后的 "(0.xx)" 分数标记，并裁长度
    s = re.sub(r"\(\s*[-+]?\d*\.?\d+\s*\)\s*$", "", s).strip()
    return (s[:max_chars] + "…") if len(s) > max_chars else s

def draw_tag(img, x, y, text, max_width_px=260, fg=(0,0,0), bg=(80,255,80), alpha=0.65):
    """在 (x,y) 处绘制带背景的可换行标签，避免 '???' 或重叠"""
    # 估算行宽：OpenCV 简单估计
    scale, thickness = 0.5, 1
    words = text.split()
    lines, cur = [], ""
    for w in words:
        to_try = (cur + " " + w).strip()
        (w_px,_), _ = cv2.getTextSize(to_try, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if w_px > max_width_px and cur:
            lines.append(cur); cur = w
        else:
            cur = to_try
    if cur: lines.append(cur)

    # 若顶端放不下就往下挪
    line_h = 14
    pad = 4
    h = pad*2 + line_h*len(lines)
    y0 = y - h - 4
    if y0 < 0:
        y0 = y + 4

    # 背景
    maxw = 0
    for ln in lines:
        (w_px,_), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        maxw = max(maxw, w_px)
    w = pad*2 + maxw
    bg_rect = img[y0:y0+h, x:x+w]
    if bg_rect.size != 0:
        overlay = bg_rect.copy()
        overlay[:] = bg
        cv2.addWeighted(overlay, alpha, bg_rect, 1-alpha, 0, bg_rect)

    # 文字
    ty = y0 + pad + line_h - 2
    for ln in lines:
        cv2.putText(img, ln, (x+pad, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)
        ty += line_h

def load_gdino(cfg_path, ckpt_path, device):
    args = SLConfig.fromfile(cfg_path)
    args.device = device
    model = build_model(args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    _ = model.load_state_dict(clean_state_dict(ckpt), strict=False)
    model.eval().to(device)
    return model

def load_image_for_gdino(image_path, image_size):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([image_size, image_size], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)  # 3, H, W
    return image_pil, image

def grounding_infer(model, image_tensor, caption, box_thr, text_thr, area_thr, device="cuda"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (Nq, 256)
    boxes  = outputs["pred_boxes"].cpu()[0]              # (Nq, 4)  cx,cy,w,h in [0,1]

    # 过滤：分数 + 面积（去掉超大框）
    mask = (logits.max(dim=1)[0] > box_thr) & ((boxes[:,2] * boxes[:,3]) < area_thr)
    if mask.sum() == 0:
        # 强制保留最高分的一条，防空
        idx = torch.argmax(logits.max(dim=1)[0])
        logits_f, boxes_f = logits[idx:idx+1], boxes[idx:idx+1]
    else:
        logits_f, boxes_f = logits[mask], boxes[mask]

    # 文本短语
    tokenizer = model.tokenizer
    tokenized  = tokenizer(caption)
    phrases = []
    for lg in logits_f:
        phr = get_phrases_from_posmap(lg > text_thr, tokenized, tokenizer)
        phr = phr + f"({str(lg.max().item())[:4]})"
        phrases.append(phr)

    return boxes_f, phrases  # 仍是归一化 cx,cy,w,h

def boxes_cxcywh_to_xyxy(boxes, img_wh):
    W, H = img_wh
    out = []
    for cx,cy,w,h in boxes.tolist():
        x1 = (cx - w/2.0) * W
        y1 = (cy - h/2.0) * H
        x2 = (cx + w/2.0) * W
        y2 = (cy + h/2.0) * H
        out.append([x1,y1,x2,y2])
    return out

# ---- 预置 Prompt：与 test.py 中一致的“通用+细粒度” ----
ANOMALY_GENERAL = ["anomaly", "damage", "broken", "defect", "contamination"]

MVT = {
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

VISA = {
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

def build_prompt(name, dataset):
    name = name.replace("_"," ")
    if dataset == "mvtec":
        details = MVT.get(name, "")
    else:
        details = VISA.get(name, "")
    words = ANOMALY_GENERAL + ([s.strip() for s in details.split(",") if s.strip()] if details else [])
    return " . ".join(words)

# ---------- 可视化/落盘 ----------
def save_boxes_overlay(sample_id, img_path, img_bgr, boxes_xyxy, phrases, out_dir):
    ensure_dir(out_dir)
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.copy()

    valid = []
    for (x1,y1,x2,y2), phr in zip(boxes_xyxy, phrases):
        x1,y1,x2,y2 = xyxy_to_int((x1,y1,x2,y2), W, H)
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
        label = short_phrase(phr, 28)
        draw_tag(overlay, x1, y1, label, max_width_px=240)
        valid.append({"box":[float(x1),float(y1),float(x2),float(y2)], "phrase":phr})

    cv2.imwrite(os.path.join(out_dir, f"{sample_id}_gdino_overlay.png"), overlay)
    with open(os.path.join(out_dir, f"{sample_id}_gdino_boxes.json"), "w") as f:
        json.dump(valid, f, indent=2)

    # 也另存文本 topk
    parsed = []
    for s in phrases:
        m = re.search(r"\(([-+]?\d*\.?\d+)\)\s*$", s)
        conf = float(m.group(1)) if m else 0.0
        parsed.append((short_phrase(s), conf))
    parsed.sort(key=lambda x: x[1], reverse=True)
    with open(os.path.join(out_dir, f"{sample_id}_gdino_text_topk.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rank","text","confidence"])
        for i,(t,c) in enumerate(parsed[:10]):
            w.writerow([i+1, t, float(c)])

# -------------------- 主逻辑 --------------------
def main():
    ap = argparse.ArgumentParser("GroundingDINO-only inference")
    ap.add_argument("--groundingdino_config", type=str,
                    default="./models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--grounded_checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--use_meta", action="store_true",
                    help="Use meta.json (data_path/meta.json) to enumerate test images instead of scanning <cls>/test dirs.")
    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--area_threshold", type=float, default=0.7)
    ap.add_argument("--viz_dir", type=str, default="./viz_gdino")
    # 两种模式：单图 / 数据集
    ap.add_argument("--image", type=str, default=None, help="单张图片路径")
    ap.add_argument("--dataset", type=str, choices=["mvtec","visa"], default=None)
    ap.add_argument("--data_path", type=str, default=None, help="数据根目录（包含各类别文件夹）")
    ap.add_argument("--class_name", type=str, default=None, help="指定类别名（可选）")
    ap.add_argument("--prompt", type=str, default=None, help="自定义文本（可选，优先级最高）")
    args = ap.parse_args()

    device = args.device
    model = load_gdino(args.groundingdino_config, args.grounded_checkpoint, device)

    # 单图模式
    if args.image is not None:
        img_path = args.image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError(img_path)
        img = cv2.resize(img, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
        _, img_t = load_image_for_gdino(img_path, args.image_size)

        # 文本
        if args.prompt:
            prompt = args.prompt
        elif args.class_name and args.dataset:
            prompt = build_prompt(args.class_name, args.dataset)
        else:
            # 最小化默认：只用通用异常词
            prompt = " . ".join(ANOMALY_GENERAL)

        boxes_cxcywh, phrases = grounding_infer(
            model, img_t, prompt,
            args.box_threshold, args.text_threshold, args.area_threshold,
            device=device
        )
        boxes_xyxy = boxes_cxcywh_to_xyxy(boxes_cxcywh, (args.image_size, args.image_size))
        sample_id = os.path.splitext(os.path.basename(img_path))[0]
        save_boxes_overlay(sample_id, img_path, img, boxes_xyxy, phrases, args.viz_dir)
        print(f"[OK] saved to {args.viz_dir}")
        return

    # 数据集模式（批量跑）
    if not (args.dataset and args.data_path):
        raise SystemExit("请指定 --image，或者指定 --dataset 与 --data_path")

    # 扫描类别子目录
    cls_dirs = sorted([d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path,d))])
    if args.class_name:
        cls_dirs = [c for c in cls_dirs if c == args.class_name or c.replace("_"," ") == args.class_name]

#     for cls in cls_dirs:
#         # 取 test 下的所有图
#         test_root = os.path.join(args.data_path, cls, "test")
#         if not os.path.exists(test_root):  # 有些数据布局不同，自行按需改
#             print(f"[WARN] skip {cls}: no test dir -> {test_root}")
#             continue
#         # 递归找图片
#         for root,_,files in os.walk(test_root):
#             for fn in files:
#                 if not fn.lower().endswith((".png",".jpg",".jpeg","bmp","tif","tiff")):
#                     continue
#                 img_path = os.path.join(root, fn)
#                 img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#                 if img is None: 
#                     print(f"[WARN] read fail: {img_path}")
#                     continue
#                 img = cv2.resize(img, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
#                 _, img_t = load_image_for_gdino(img_path, args.image_size)

#                 # prompt 优先级：--prompt > 数据集内置
#                 if args.prompt:
#                     prompt = args.prompt
#                 else:
#                     prompt = build_prompt(cls, args.dataset)

#                 boxes_cxcywh, phrases = grounding_infer(
#                     model, img_t, prompt,
#                     args.box_threshold, args.text_threshold, args.area_threshold,
#                     device=device
#                 )
#                 boxes_xyxy = boxes_cxcywh_to_xyxy(boxes_cxcywh, (args.image_size, args.image_size))
#                 sample_id = os.path.splitext(os.path.basename(img_path))[0]
#                 out_dir = os.path.join(args.viz_dir, args.dataset, cls)
#                 save_boxes_overlay(sample_id, img_path, img, boxes_xyxy, phrases, out_dir)
# === NEW: 用 meta.json（若存在）驱动推理；否则退回旧的“test 目录遍历” ===
        meta_path = os.path.join(args.data_path, "meta.json")
        use_meta = os.path.isfile(meta_path)
        
        if use_meta:
            print(f"[INFO] found meta.json -> {meta_path}, will iterate its 'test' split")
            with open(meta_path, "r") as f:
                meta = json.load(f)
            # 兼容两种结构：{"test":{cls:[...]}} 或直接 {cls:[...]}
            test_split = meta.get("test", meta)
            # 类名集合
            cls_names = sorted(test_split.keys())
        
            for cls in cls_names:
                entries = test_split[cls]
                out_dir = os.path.join(args.viz_dir, args.dataset, cls)
                os.makedirs(out_dir, exist_ok=True)
        
                for rec in entries:
                    img_rel = rec.get("img_path", "")
                    if not img_rel:
                        continue
                    # 路径既可能是相对 data_path，也可能是绝对路径，这里统一处理
                    img_path = img_rel if os.path.isabs(img_rel) else os.path.join(args.data_path, img_rel)
                    if not os.path.isfile(img_path):
                        print(f"[WARN] not found: {img_path}")
                        continue
        
                    # 读取与缩放一份可视化用的 BGR 图
                    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        print(f"[WARN] read fail: {img_path}")
                        continue
                    img_bgr = cv2.resize(img_bgr, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
        
                    # 给 GroundingDINO 的张量
                    _, img_t = load_image_for_gdino(img_path, args.image_size)
        
                    # 文本提示词：命令行 --prompt 优先，其次用内置（按你的 gdino_only 原逻辑）
                    prompt = args.prompt if getattr(args, "prompt", None) else build_prompt(cls, args.dataset)
        
                    boxes_cxcywh, phrases = grounding_infer(
                        model, img_t, prompt,
                        args.box_threshold, args.text_threshold, args.area_threshold,
                        device=device
                    )
                    boxes_xyxy = boxes_cxcywh_to_xyxy(boxes_cxcywh, (args.image_size, args.image_size))
        
                    sample_id = os.path.splitext(os.path.basename(img_path))[0]
                    save_boxes_overlay(sample_id, img_path, img_bgr, boxes_xyxy, phrases, out_dir)
        
            print(f"[OK] all done, outputs in: {args.viz_dir}")
        
        else:
            # ——保留你原来的“test 目录遍历”逻辑作为后备——
            for cls in cls_dirs:
                test_root = os.path.join(args.data_path, cls, "test")
                if not os.path.exists(test_root):
                    print(f"[WARN] skip {cls}: no test dir -> {test_root}")
                    continue
                for root, _, files in os.walk(test_root):
                    for fn in files:
                        if not fn.lower().endswith((".png",".jpg",".jpeg","bmp","tif","tiff")):
                            continue
                        img_path = os.path.join(root, fn)
                        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        if img_bgr is None:
                            print(f"[WARN] read fail: {img_path}")
                            continue
                        img_bgr = cv2.resize(img_bgr, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
                        _, img_t = load_image_for_gdino(img_path, args.image_size)
        
                        prompt = args.prompt if getattr(args, "prompt", None) else build_prompt(cls, args.dataset)
        
                        boxes_cxcywh, phrases = grounding_infer(
                            model, img_t, prompt,
                            args.box_threshold, args.text_threshold, args.area_threshold,
                            device=device
                        )
                        boxes_xyxy = boxes_cxcywh_to_xyxy(boxes_cxcywh, (args.image_size, args.image_size))
                        sample_id = os.path.splitext(os.path.basename(img_path))[0]
                        out_dir = os.path.join(args.viz_dir, args.dataset, cls)
                        save_boxes_overlay(sample_id, img_path, img_bgr, boxes_xyxy, phrases, out_dir)
            print(f"[OK] all done, outputs in: {args.viz_dir}")


if __name__ == "__main__":
    main()
