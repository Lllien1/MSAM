#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, re
from glob import glob

def norm_rel(root, p):
    return os.path.relpath(p, root).replace("\\", "/")

def num_stem(fn):
    # 提取文件名中的数字部分（000、0123 等）
    m = re.search(r'(\d+)', os.path.basename(fn))
    return m.group(1) if m else None

def build_meta(data_root, out_json, split_for_all="test"):
    """
    data_root: /root/autodl-tmp/FiLo/data/VisA_20220922
    会扫描每个类的 Normal / Anomaly 图片，并尝试为 Anomaly 配对 mask。
    输出为扁平 list[dict]，字段：img_path, mask_path, cls_name, split, anomaly
    """
    classes = sorted([d for d in os.listdir(data_root) 
                      if os.path.isdir(os.path.join(data_root,d)) and d not in ("split_csv",)])
    out = []
    miss_mask = 0
    total_ano = 0

    for cls in classes:
        img_norm_dir = os.path.join(data_root, cls, "Data", "Images", "Normal")
        img_ano_dir  = os.path.join(data_root, cls, "Data", "Images", "Anomaly")
        msk_ano_dir  = os.path.join(data_root, cls, "Data", "Masks",  "Anomaly")

        # Normal 图片（无 mask）
        for pat in ("*.jpg","*.JPG","*.jpeg","*.png","*.bmp","*.tif","*.tiff"):
            for img in glob(os.path.join(img_norm_dir, pat)):
                out.append({
                    "img_path": norm_rel(data_root, img),
                    "mask_path": "",
                    "cls_name": cls,
                    "split": split_for_all,  # 没有官方划分时，统一标为 test
                    "anomaly": 0
                })

        # Anomaly 图片（尝试匹配 mask/<num>.png）
        for pat in ("*.jpg","*.JPG","*.jpeg","*.png","*.bmp","*.tif","*.tiff"):
            for img in glob(os.path.join(img_ano_dir, pat)):
                total_ano += 1
                stem = num_stem(img)
                mask_rel = ""
                if stem:
                    cand = os.path.join(msk_ano_dir, f"{stem}.png")
                    if os.path.exists(cand):
                        mask_rel = norm_rel(data_root, cand)
                    else:
                        miss_mask += 1
                else:
                    miss_mask += 1

                out.append({
                    "img_path": norm_rel(data_root, img),
                    "mask_path": mask_rel,   # 配不到就留空
                    "cls_name": cls,
                    "split": split_for_all,
                    "anomaly": 1
                })

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote {out_json} with {len(out)} records. anomaly images: {total_ano}, missing masks: {miss_mask}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="e.g. /root/autodl-tmp/FiLo/data/VisA_20220922")
    ap.add_argument("--out_json", default="meta_gdino.json", help="output json filename under data_root")
    ap.add_argument("--split", default="test", help="value for split field when dataset has no official split")
    args = ap.parse_args()

    out_json = os.path.join(args.data_root, args.out_json)
    build_meta(args.data_root, out_json, split_for_all=args.split)
