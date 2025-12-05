import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from torch.utils.tensorboard import SummaryWriter

from datasets.mvtec_supervised import MVTecDataset
from datasets.visa_supervised import VisaDataset
from utils.loss import FocalLoss, BinaryDiceLoss
import models.vv_open_clip as open_clip
from models.FiLo import FiLo


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def build_fewshot_dataset(args, preprocess, target_transform):
    cls_name_raw = args.cls_name
    mode = "train" if cls_name_raw else "train_all"
    common_kwargs = dict(
        root=args.train_data_path,
        transform=preprocess,
        target_transform=target_transform,
        k_shot=args.k_shot,
        save_dir=args.save_path,
    )
    if args.dataset == "mvtec":
        return MVTecDataset(
            aug_rate=args.aug_rate,
            mode=mode,
            obj_name=cls_name_raw if cls_name_raw else None,
            **common_kwargs,
        )
    if args.dataset == "visa":
        return VisaDataset(
            mode=mode,
            obj_name=cls_name_raw if cls_name_raw else None,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def parse_args():
    parser = argparse.ArgumentParser("FiLo Few-shot Train", add_help=True)
    parser.add_argument("--dataset", type=str, choices=["mvtec", "visa"], required=True)
    parser.add_argument(
        "--cls_name",
        type=str,
        required=False,
        default="",
        help="target class name; leave empty to sample k_shot for every class",
    )
    parser.add_argument("--k_shot", type=int, default=5, help="number of training samples to draw")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./ckpt_fewshot")

    parser.add_argument("--clip_model", type=str, default="ViT-L-14-336")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument(
        "--features_list", type=int, nargs="+", default=[6, 12, 18, 24]
    )
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--aug_rate", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--decoder_learning_rate", type=float, default=1e-4)
    parser.add_argument("--adapter_learning_rate", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--adapter_epoch", type=int, default=2)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="early-stop patience (epochs without improvement)",
    )
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default="",
        help="path to pretrained FiLo checkpoint (optional, e.g., filo_train_on_visa.pth)",
    )
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_path, "runs"))
    global_step = 0
    best_stage1 = float("inf")
    patience1 = 0
    best_stage2 = float("inf")
    patience2 = 0

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

    train_dataset = build_fewshot_dataset(args, preprocess, target_transform)
    train_loader = DataLoaderX(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    obj_list = [name.replace("_", " ") for name in train_dataset.get_cls_names()]
    device = args.device
    filo_model = FiLo(obj_list, args, device).to(device)
    if args.init_ckpt:
        if os.path.isfile(args.init_ckpt):
            ckpt = torch.load(args.init_ckpt, map_location=device)
            state_dict = ckpt.get("filo", ckpt)
            print(f"Loading initial checkpoint from {args.init_ckpt}")
            filo_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[WARN] init_ckpt {args.init_ckpt} not found, training from scratch.")

    main_part_param_groups = [
        {"params": filo_model.decoder_cov.parameters(), "lr": args.decoder_learning_rate},
        {"params": filo_model.decoder_linear.parameters(), "lr": args.decoder_learning_rate},
        {"params": filo_model.normal_prompt_learner.parameters(), "lr": args.learning_rate},
        {"params": filo_model.abnormal_prompt_learner.parameters(), "lr": args.learning_rate},
    ]
    optimizer_main = torch.optim.AdamW(main_part_param_groups, betas=(0.5, 0.999))

    adapter_param_groups = [
        {"params": filo_model.adapter.parameters(), "lr": args.adapter_learning_rate},
    ]
    optimizer_adapter = torch.optim.AdamW(adapter_param_groups, betas=(0.5, 0.999))

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    # stage 1: decoder + prompt
    for epoch in range(args.epoch):
        loss_list = []
        for items in tqdm(train_loader, desc=f"main epoch {epoch+1}/{args.epoch}"):
            text_probs, anomaly_maps = filo_model(items, with_adapter=False)
            gt = items["img_mask"].squeeze().to(device)
            gt[gt > 0.5], gt[gt <= 0.5] = 1, 0

            loss = 0
            for amap in anomaly_maps:
                loss += loss_focal(amap, gt)
                loss += loss_dice(amap[:, 1, :, :], gt)
                loss += loss_dice(amap[:, 0, :, :], 1 - gt)

            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()
            loss_list.append(loss.item())
            global_step += 1
            writer.add_scalar("stage1/loss_iter", loss.item(), global_step)
        print(f"[Stage1] epoch {epoch+1}/{args.epoch} loss={np.mean(loss_list):.4f}")
        writer.add_scalar("stage1/loss_epoch", np.mean(loss_list), epoch + 1)
        if np.mean(loss_list) < best_stage1 - 1e-6:
            best_stage1 = np.mean(loss_list)
            patience1 = 0
        else:
            patience1 += 1
        if patience1 >= args.patience:
            print(f"[Stage1] early stop triggered at epoch {epoch+1}")
            break

    # stage 2: adapter only
    for epoch in range(args.adapter_epoch):
        loss_list = []
        for items in tqdm(train_loader, desc=f"adapter epoch {epoch+1}/{args.adapter_epoch}"):
            label = items["anomaly"][0].to(device)
            text_probs, _ = filo_model(items, only_train_adapter=True, with_adapter=True)
            logits = text_probs[:, 0, ...] / 0.07
            loss = F.cross_entropy(logits.squeeze(), label)

            optimizer_adapter.zero_grad()
            loss.backward()
            optimizer_adapter.step()
            loss_list.append(loss.item())
            global_step += 1
            pred = logits.argmax(dim=1)
            acc = (pred == label).float().mean().item()
            writer.add_scalar("stage2/loss_iter", loss.item(), global_step)
            writer.add_scalar("stage2/acc_iter", acc, global_step)
        mean_loss = np.mean(loss_list)
        print(f"[Adapter] epoch {epoch+1}/{args.adapter_epoch} loss={mean_loss:.4f}")
        writer.add_scalar("stage2/loss_epoch", mean_loss, epoch + 1)
        if mean_loss < best_stage2 - 1e-6:
            best_stage2 = mean_loss
            patience2 = 0
        else:
            patience2 += 1
        if patience2 >= args.patience:
            print(f"[Adapter] early stop triggered at epoch {epoch+1}")
            break

    tag = args.cls_name if args.cls_name else "allcls"
    save_name = f"{tag}_k{args.k_shot}_filo_train_on_{args.dataset}"
    ckpt_path = os.path.join(args.save_path, f"{save_name}.pth")
    torch.save({"filo": filo_model.state_dict()}, ckpt_path)
    print(f"Saved few-shot checkpoint to {ckpt_path}")
    writer.close()


if __name__ == "__main__":
    main()
