import argparse
import copy
import logging
import os
import re
import json

import cv2
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from tabulate import tabulate
from tqdm import tqdm

import models.vv_open_clip as open_clip
from datasets.mvtec_supervised import MVTecDataset
from datasets.visa_supervised import VisaDataset
from models.FiLo import FiLo
from models.GroundingDINO.groundingdino.datasets import transforms as T
from models.GroundingDINO.groundingdino.models import build_model
from models.GroundingDINO.groundingdino.util.slconfig import SLConfig
from models.GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)


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

for cls in mvtec_anomaly_detail_gpt:
    mvtec_anomaly_detail_gpt[cls] = mvtec_anomaly_detail_gpt[cls].split(",")
for cls in visa_anomaly_detail_gpt:
    visa_anomaly_detail_gpt[cls] = visa_anomaly_detail_gpt[cls].split(",")

status_abnormal = {}
for cls in mvtec_anomaly_detail_gpt:
    status_abnormal[cls] = [
        "abnormal {} with {}".format("{}", detail) for detail in mvtec_anomaly_detail_gpt[cls]
    ] + status_abnormal_winclip
for cls in visa_anomaly_detail_gpt:
    status_abnormal[cls] = [
        "abnormal {} with {}".format("{}", detail) for detail in visa_anomaly_detail_gpt[cls]
    ] + status_abnormal_winclip

positions_list = [
    "top left",
    "top",
    "top right",
    "left",
    "center",
    "right",
    "bottom left",
    "bottom",
    "bottom right",
]

location = {
    "top left": [(0, 0), (172, 172)],
    "top": [(173, 0), (344, 172)],
    "top right": [(345, 0), (517, 172)],
    "left": [(0, 173), (172, 344)],
    "center": [(173, 173), (344, 344)],
    "right": [(345, 173), (517, 344)],
    "bottom left": [(0, 345), (172, 517)],
    "bottom": [(173, 345), (344, 517)],
    "bottom right": [(345, 345), (517, 517)],
}

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
    heatmap = (plt.cm.jet(mask_norm)[:, :, :3] * 255).astype("uint8")
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    if torch.is_tensor(boxes):
        boxes_iter = boxes.cpu().numpy()
    else:
        boxes_iter = boxes
    for rect in boxes_iter:
        if len(rect) != 4:
            continue
        x1, y1, x2, y2 = map(int, rect)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    base = f"{cls_name}_{os.path.splitext(os.path.basename(image_path))[0]}"
    cv2.imwrite(os.path.join(viz_dir, f"{base}_orig.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(viz_dir, f"{base}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(viz_dir, f"{base}_mask.png"), (mask_norm * 255).astype("uint8"))
    cv2.imwrite(os.path.join(viz_dir, f"{base}_mask_bin.png"), ((mask_norm >= 0.5) * 255).astype("uint8"))


def save_grounding_phrases(image_path, cls_name, boxes, phrases, viz_dir):
    if not viz_dir or not phrases:
        return
    os.makedirs(viz_dir, exist_ok=True)
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().tolist()
    records = []
    for rect, phr in zip(boxes, phrases):
        if len(rect) != 4:
            continue
        records.append({"phrase": phr, "box": [float(x) for x in rect]})
    base = f"{cls_name}_{os.path.splitext(os.path.basename(image_path))[0]}"
    json_path = os.path.join(viz_dir, f"{base}_phrases.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_image(image_path, image_size):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([image_size, image_size], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image


def cal_pro_score(obj, masks, amaps, max_step=200, expect_fpr=0.3):
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            from skimage import measure

            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
    pros, fprs = np.array(pros), np.array(fprs)
    idxes = fprs < expect_fpr
    if not np.any(idxes):
        return 0.0
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min() + 1e-6)
    return auc(fprs, pros[idxes])


def load_dino(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint), strict=False)
    model.eval()
    return model


def get_grounding_output(
    model, image, caption, box_threshold, text_threshold, device="cpu", area_thr=0.8
):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    boxes_area = boxes_filt[:, 2] * boxes_filt[:, 3]
    filt_mask = torch.bitwise_and(
        (logits_filt.max(dim=1)[0] > box_threshold), (boxes_area < area_thr)
    )
    if torch.sum(filt_mask) == 0:
        idx = torch.argmax(logits_filt.max(dim=1)[0])
        logits_filt = logits_filt[idx].unsqueeze(0)
        boxes_filt = boxes_filt[idx].unsqueeze(0)
    else:
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    pred_phrases = []
    boxes_filt_category = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        boxes_filt_category.append(box)
    boxes_filt_category = torch.stack(boxes_filt_category, dim=0)
    return boxes_filt_category, pred_phrases


def check_elements_in_array(arr1, arr2):
    return any(elem in arr2 for elem in arr1)


def parse_args():
    parser = argparse.ArgumentParser("FiLo Few-shot Test", add_help=True)
    parser.add_argument("--dataset", type=str, choices=["mvtec", "visa"], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--groundingdino_config", type=str, required=True)
    parser.add_argument("--grounded_checkpoint", type=str, required=True)
    parser.add_argument("--cls_name", type=str, default="", help="evaluate single class if provided")

    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--area_threshold", type=float, default=0.7)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--clip_model", type=str, default="ViT-L-14-336")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max_pro_steps",
        type=int,
        default=50,
        help="number of thresholds for PRO calculation (smaller is faster; default 50)",
    )
    parser.add_argument(
        "--viz_dir",
        type=str,
        default="",
        help="directory to save visualization outputs (disabled if empty)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
    logger = logging.getLogger("fewshot_test")

    model_dino = load_dino(args.groundingdino_config, args.grounded_checkpoint, device=device)

    _, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, args.image_size, pretrained=args.clip_pretrained
    )
    target_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )
    gaussion_filter = torchvision.transforms.GaussianBlur(3, 4.0)

    if args.dataset == "mvtec":
        test_data = MVTecDataset(
            root=args.data_path,
            transform=preprocess,
            target_transform=target_transform,
            aug_rate=-1,
            mode="test",
        )
        detail_dict = mvtec_anomaly_detail_gpt
    else:
        test_data = VisaDataset(
            root=args.data_path,
            transform=preprocess,
            target_transform=target_transform,
            mode="test",
        )
        detail_dict = visa_anomaly_detail_gpt

    target_cls = args.cls_name.replace("_", " ") if args.cls_name else None
    if target_cls:
        obj_list = [target_cls]
    else:
        obj_list = [x.replace("_", " ") for x in test_data.get_cls_names()]

    filo_model = FiLo(obj_list, args, device).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    filo_model.load_state_dict(ckpt["filo"], strict=False)

    loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    results = {"cls_names": [], "imgs_masks": [], "anomaly_maps": [], "gt_sp": [], "pr_sp": []}

    for items in tqdm(loader):
        cls_name = items["cls_name"][0]
        if target_cls and cls_name != target_cls:
            continue

        image_path = items["img_path"][0]
        results["cls_names"].append(cls_name)

        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results["imgs_masks"].append(gt_mask)
        results["gt_sp"].append(items["anomaly"].item())

        if args.dataset == "mvtec":
            text_prompt = " . ".join(anomaly_status_general + mvtec_anomaly_detail_gpt[cls_name])
        else:
            text_prompt = " . ".join(anomaly_status_general + visa_anomaly_detail_gpt[cls_name])

        image_dino = load_image(image_path, args.image_size)
        boxes_filt, pred_phrases = get_grounding_output(
            model_dino,
            image_dino,
            text_prompt,
            args.box_threshold,
            args.text_threshold,
            device=device,
            area_thr=args.area_threshold,
        )

        boxes_filt_copy = copy.deepcopy(boxes_filt)
        valid_boxes = []
        valid_phrases = []
        if args.dataset == "mvtec":
            vocab = mvtec_anomaly_detail_gpt[cls_name] + anomaly_status_general
        else:
            vocab = visa_anomaly_detail_gpt[cls_name] + anomaly_status_general
        for i in range(boxes_filt.size(0)):
            if not check_elements_in_array(vocab, pred_phrases[i]):
                continue
            box = boxes_filt_copy[i] * torch.Tensor(
                [args.image_size, args.image_size, args.image_size, args.image_size]
            )
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            valid_boxes.append(box.cpu())
            valid_phrases.append(pred_phrases[i])
        if valid_boxes:
            boxes_selected = torch.stack(valid_boxes)
        else:
            boxes_selected = torch.empty((0, 4))

        position = []
        max_box = None
        max_pred = 0
        for rect, phrase in zip(boxes_selected, pred_phrases):
            if not isinstance(rect, torch.Tensor):
                continue
            match = re.search(r"\((.*?)\)", phrase)
            if not match:
                continue
            number = float(match.group(1))
            if number >= max_pred:
                max_pred = number
                max_box = rect
        if max_box is not None:
            center = ((max_box[0] + max_box[2]) / 2, (max_box[1] + max_box[3]) / 2)
            for region, ((x1, y1), (x2, y2)) in location.items():
                if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                    position.append(region)
                    break

        with torch.no_grad():
            text_probs, anomaly_maps = filo_model(items, with_adapter=True, positions=position)
            for i in range(len(anomaly_maps)):
                anomaly_maps[i] = gaussion_filter(
                    (anomaly_maps[i][:, 1, :, :] - anomaly_maps[i][:, 0, :, :] + 1) / 2
                )
            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=0), dim=0).unsqueeze(1)
            results["pr_sp"].append(
                (text_probs.flatten()[1].item() + anomaly_map.max().item()) / 2
            )

            if boxes_selected.numel() > 0:
                score_copy = anomaly_map.clone()
                for rect in boxes_selected:
                    x1, y1, x2, y2 = map(int, rect.tolist())
                    score_copy[:, :, y1:y2, x1:x2] = 1
                anomaly_map = torch.where(score_copy == 1, anomaly_map, anomaly_map * 0.7)

            results["anomaly_maps"].append(
                anomaly_map.detach().cpu().numpy().reshape(args.image_size, args.image_size)
            )
            mask_np = results["anomaly_maps"][-1]
            save_visualization(
                image_path,
                mask_np,
                boxes_selected,
                cls_name,
                args.viz_dir,
                args.image_size,
            )
            save_grounding_phrases(
                image_path,
                cls_name,
                boxes_selected,
                valid_phrases,
                args.viz_dir,
            )

    if not results["cls_names"]:
        logger.warning("No samples processed; check cls_name or dataset path.")
        return

    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_ls = []
    ap_sp_ls = []
    ap_px_ls = []

    def cal_score(obj):
        idxes = [i for i, cls in enumerate(results["cls_names"]) if cls == obj]
        if not idxes:
            return
        gt_px = np.array([results["imgs_masks"][i].squeeze(1).numpy() for i in idxes])
        pr_px = np.array([results["anomaly_maps"][i] for i in idxes])
        gt_sp = np.array([results["gt_sp"][i] for i in idxes])
        pr_sp = np.array([results["pr_sp"][i] for i in idxes])

        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        ap_sp = average_precision_score(gt_sp, pr_sp)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        precisions, recalls, _ = precision_recall_curve(gt_sp, pr_sp)
        f1_sp = np.max((2 * precisions * recalls) / (precisions + recalls + 1e-6))
        precisions_px, recalls_px, _ = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_px = np.max((2 * precisions_px * recalls_px) / (precisions_px + recalls_px + 1e-6))
        aupro = cal_pro_score(obj, gt_px, pr_px, max_step=args.max_pro_steps)

        table_ls.append(
            [
                obj,
                f"{auroc_px*100:.1f}",
                f"{f1_px*100:.1f}",
                f"{ap_px*100:.1f}",
                f"{auroc_sp*100:.1f}",
                f"{f1_sp*100:.1f}",
                f"{ap_sp*100:.1f}",
                f"{aupro*100:.1f}",
            ]
        )
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)
        f1_sp_ls.append(f1_sp)
        f1_px_ls.append(f1_px)
        aupro_ls.append(aupro)
        ap_sp_ls.append(ap_sp)
        ap_px_ls.append(ap_px)

    for cls in obj_list:
        cal_score(cls)

    headers = [
        "objects",
        "auroc_px",
        "f1_px",
        "ap_px",
        "auroc_sp",
        "f1_sp",
        "ap_sp",
        "aupro",
    ]
    table_ls.append(
        [
            "mean",
            f"{np.mean(auroc_px_ls)*100:.1f}",
            f"{np.mean(f1_px_ls)*100:.1f}",
            f"{np.mean(ap_px_ls)*100:.1f}",
            f"{np.mean(auroc_sp_ls)*100:.1f}",
            f"{np.mean(f1_sp_ls)*100:.1f}",
            f"{np.mean(ap_sp_ls)*100:.1f}",
            f"{np.mean(aupro_ls)*100:.1f}",
        ]
    )
    logger.info("\n%s", tabulate(table_ls, headers=headers, tablefmt="pipe"))


if __name__ == "__main__":
    main()
