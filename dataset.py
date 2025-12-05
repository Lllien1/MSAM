import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from utils.defect_definitions import mvtec_short_keywords

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _default_transforms(image_size: int = 1008) -> Tuple[Callable, Callable]:
    """Default image/mask transforms aligned with SAM3 1008px input."""
    img_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ]
    )
    return img_tf, mask_tf


@dataclass
class SampleEntry:
    img_path: str
    mask_path: str
    cls_name: str
    anomaly: int
    specie_name: str


class MVTecMetaDataset(Dataset):
    """Dataset that mirrors FiLo `mvtec_supervised.py` meta.json sampling (train/train_all/test).

    - meta.json structure: meta['train'/'test'][cls] is a list of dicts with keys
      {'img_path','mask_path','cls_name','specie_name','anomaly'}.
    - train: k-shot per class_name (obj_name) from train split; train_all: k-shot per class across train split;
      test: full test split per meta.json.
    - aug_rate: probability to synthesize a 2x2 mosaic from random defects in test set of the same class.
    - Returns: (image_tensor, mask_tensor, prompt_list, is_anomaly, cls_name)
    """

    def __init__(
        self,
        root: str,
        meta_path: Optional[str] = None,
        mode: str = "test",
        k_shot: int = 0,
        obj_name: Optional[str] = None,
        aug_rate: float = 0.0,
        prompt_dict: Optional[Dict[str, List[str]]] = None,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.aug_rate = aug_rate
        self.prompt_dict = prompt_dict or mvtec_short_keywords
        img_tf, mask_tf = _default_transforms()
        self.image_transform = image_transform or img_tf
        self.mask_transform = mask_transform or mask_tf

        meta_file = meta_path or os.path.join(root, "meta.json")
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_info = json.load(f)

        if mode == "train_all":
            split_meta = meta_info["train"]
            cls_names = list(split_meta.keys())
        elif mode == "train":
            split_meta = meta_info["train"]
            cls_names = [obj_name] if obj_name is not None else list(split_meta.keys())
        else:
            split_meta = meta_info[mode]
            cls_names = list(split_meta.keys())

        self.entries: List[SampleEntry] = []
        for cls in cls_names:
            data_list = split_meta[cls]
            if mode in ("train", "train_all") and k_shot > 0:
                indices = torch.randint(0, len(data_list), (k_shot,))
                chosen = [data_list[i] for i in indices]
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    with open(os.path.join(save_dir, "k_shot.txt"), "a", encoding="utf-8") as f:
                        for d in chosen:
                            f.write(d["img_path"] + "\n")
            else:
                chosen = data_list
            for d in chosen:
                self.entries.append(
                    SampleEntry(
                        img_path=d["img_path"],
                        mask_path=d["mask_path"],
                        cls_name=d["cls_name"],
                        anomaly=int(d["anomaly"]),
                        specie_name=d.get("specie_name", d["cls_name"]),
                    )
                )

        # cache class-wise test paths for mosaic augmentation
        self.test_cache = split_meta if "test" in meta_info else meta_info.get("test", {})

    def __len__(self) -> int:
        return len(self.entries)

    def _combine_img(self, cls_name: str) -> Tuple[Image.Image, Image.Image]:
        """Mimic mvtec_supervised combine_img: 2x2 mosaic from random test defects."""
        img_paths_root = os.path.join(self.root, cls_name, "test")
        img_ls, mask_ls = [], []
        defects = os.listdir(img_paths_root)
        for _ in range(4):
            defect = random.choice(defects)
            files = os.listdir(os.path.join(img_paths_root, defect))
            random_file = random.choice(files)
            img_path = os.path.join(img_paths_root, defect, random_file)
            mask_path = os.path.join(
                self.root, cls_name, "ground_truth", defect, random_file[:3] + "_mask.png"
            )
            img = Image.open(img_path).convert("RGB")
            img_ls.append(img)
            if defect == "good":
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
            else:
                mask_arr = np.array(Image.open(mask_path).convert("L")) > 0
                img_mask = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L")
            mask_ls.append(img_mask)

        w, h = img_ls[0].size
        result_image = Image.new("RGB", (2 * w, 2 * h))
        result_mask = Image.new("L", (2 * w, 2 * h))
        for i, (img, msk) in enumerate(zip(img_ls, mask_ls)):
            row, col = divmod(i, 2)
            x, y = col * w, row * h
            result_image.paste(img, (x, y))
            result_mask.paste(msk, (x, y))
        return result_image, result_mask

    def __getitem__(self, idx: int):
        data = self.entries[idx]
        img_path = os.path.join(self.root, data.img_path)
        mask_path = os.path.join(self.root, data.mask_path) if data.mask_path else None
        cls_name = data.cls_name
        is_anomaly = data.anomaly != 0

        try:
            if random.random() < self.aug_rate:
                img, img_mask = self._combine_img(cls_name)
            else:
                img = Image.open(img_path).convert("RGB")
                if not is_anomaly or mask_path is None or not os.path.exists(mask_path):
                    img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
                    is_anomaly = False
                else:
                    mask_arr = np.array(Image.open(mask_path).convert("L")) > 0
                    img_mask = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L")

            img = self.image_transform(img)
            img_mask = self.mask_transform(img_mask)
        except (OSError, ValueError) as e:
            # log skipped file for troubleshooting
            print(f"[WARN] Skip corrupted sample idx={idx} img={img_path} mask={mask_path} err={e}")
            # fallback to next sample to avoid worker crash on truncated images
            return self.__getitem__((idx + 1) % len(self.entries))

        # Build prompt list
        if is_anomaly:
            prompt_list = [cls_name] + self.prompt_dict.get(cls_name, [])
        else:
            prompt_list = ["normal", "clean", cls_name]

        return img, img_mask, prompt_list, is_anomaly, cls_name
