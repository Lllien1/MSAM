# ModelFramework

> Soldier-Officer Window self-Attention（SOWA）仓库的模型结构与 API 速查。本文聚焦 HFWA 模型（配置 `configs/model/sowa_hfwa.yaml`），分别拆解数据、Hydra 配置、LightningModule 以及视觉/文本/融合模块，帮助你快速定位可修改的层级。

## 1. 端到端流程总览

| 阶段 | 位置 | 要点 |
| --- | --- | --- |
| Hydra 启动 | `src/train.py` / `src/eval.py` | 通过 `hydra.main` 读取根配置（`configs/train.yaml` / `configs/eval.yaml`），按 CLI 覆盖组合 `data/`, `model/`, `callbacks/`, `logger/`, `trainer/`, `prompt/` 等组。 |
| 数据模块 | `src/data/anomaly_clip_datamodule.py` | `AnomalyCLIPDataModule` 负责实例化数据集、`kshot_dataloader`，并暴露 Lightning 期望的 `train/val/test` dataloader。 |
| 模型构建 | `src/models/anomaly_clip_module.AnomalyCLIPModule` | Hydra `model=sowa_hfwa` 时，LightningModule 内部组合 `AnomalyCLIP` 主干 + 多损失 + 指标。 |
| 训练/评估 | Lightning `Trainer` | `training_step`/`validation_step`/`test_step` 统一调用 `self.net(images, cls_name)`，并根据配置启用 Few-shot、可视化回调等。 |
| 自动化脚本 | `scripts/*.sh` | 例如 `scripts/infer_auto.sh` 复用 `src/eval.py` CLI，方便批量运行。 |

## 2. 配置与数据

### 2.1 Hydra 配置栈

- `configs/train.yaml` / `configs/eval.yaml`：声明默认数据/模型/Trainer/Logger/Callbacks，并允许 `experiment=*`、`prompt=*`、`local=*` 等 override。
- 数据配置 `configs/data/*.yaml`：包含数据根目录、批大小、`num_workers`、是否启用 `kshot_dataset` 等。
- 模型配置 `configs/model/*.yaml`：指向 `src/models/anomaly_clip_module.AnomalyCLIPModule`，并细化 `net`, `optimizer`, `scheduler`, `loss`, `k_shot`, `filter` 等参数。`sowa_hfwa.yaml` 是仓库默认的 HFWA 组合。
- Prompt 组 `configs/prompt/*.yaml`：集中管理 `class_names`、`state_template`（normal/anomaly 模板）、是否启用 Coop 等。
- Trainer/Callbacks/Logger 组：Lightning 原生参数，均可通过 `+trainer.fast_dev_run=true` 等 CLI 覆盖。

### 2.2 数据模块与数据集

1. **Meta 文件**：`src/data/components/anomal_dataset.py` 与 `kshot_dataset.py` 都要求数据目录下存在 `meta.json`，结构大致为：
   ```json
   {
     "train": {"cls": [{"img_path": "...", "mask_path": "...", "anomaly": 0}, ...]},
     "test": { ... }
   }
   ```
2. **Dataset 基类**：`BaseDataset` 支持 `preload` 或按需加载，`_load_image/_load_mask` 统一处理 PIL 图像和二值 mask（空缺时返回全零 mask，分辨率来自 `default_size`）。
3. **Visa/MVTec Dataset**：
   - `VisaDataset`：在 `__getitem__` 中返回 `{image, image_mask, cls_name, anomaly, image_path}`，供 LightningModule 直接使用。
   - `MVTecDataset`：可启用 `aug_rate`，以 2×2 拼图的方式在 `_combine_img` 中合成新样本。
4. **Few-shot 数据**：`VisaKShotDataset` / `MVTecKShotDataset` 继承 `KShotDataset`，在 `k_shot` 模式下由 `AnomalyCLIPDataModule.kshot_dataloader()` 读取，格式为 `{'image': Tensor[k, C, H, W], 'mask': Tensor[k, 1, H, W], 'cls_name': str}`。
5. **图像与 mask 变换**：`src/data/components/transform.py` 中的 `ImageTransform` / `MaskTransform` 负责 Resize→CenterCrop→ToTensor→Normalize，默认与 OpenAI CLIP 的均值/方差对齐。

## 3. HFWA 模型主干（AnomalyCLIP）

`src/models/components/anomaly_clip.AnomalyCLIP` 是核心容器，配置字段来源于 Hydra：

- `arch`: OpenAI CLIP backbone（默认 `ViT-L/14@336px`）。
- `image_size`: 决定预处理与最终 anomaly map 的输出尺寸（HFWA 默认 336）。
- `feature_map_idx`: 要 hook 的 Transformer block 下标。对 ViT-L/14 来说 0~23 共 24 层，此处 `[5, 11, 17, 23]` 对应论文中 Soldier→Officer 的四个 Stage（H1-H4）。
- `adapter`: 指向 `BasicLayer`（`_partial_` 注入），用于在每个 hook 层上执行 Frozen Window Self-Attention。
- `fusion`: 目前使用 `src/models/components/cross_modal.DotProductFusion`。
- `state_template`: Coop Prompt 的 normal/anomaly 模板，例如 `["{}", "a brand new {}"]` / `["damaged {}", "broken {}"]`。

### 3.1 视觉分支：VisionMapEncoder + HFWA

文件：`src/models/components/vision_encoder.py`

1. `VisionMapEncoder` 继承 `BaseEncoder`，在初始化时调用 `register_feature_maps`，对 `clip_model.visual.transformer.resblocks[idx]` 注册 `forward_hook`。
2. `forward` 流程：
   - `encode` 直接调用 `clip_model.encode_image(images)` 得到 CLS 全局特征。
   - `feature_maps = self.get_feature_maps()` 返回每个索引处的 `[L+1, B, C]` 张量（第 0 个 token 为 CLS）。
   - `patches`: 将 hook 输出 permute 为 `[B, L, C]` 并乘 `visual.proj`（CLIP 的线性投影），得到“未适配”的 patch token。
   - `patch_features`: 将 hook 输出送入 Adapter（见 3.2），输出同尺寸 patch token，再通过 `visual.proj` 对齐到 CLIP 语义空间。
3. 返回值：`(image_feature, patches, patch_features)`，供 `AnomalyCLIP.forward` 计算 logits 与多层 anomaly map。

### 3.2 Adapter（BasicLayer）细节

文件：`src/models/components/adapter.py`

- `BasicLayer` 由若干 `SwinTransformerBlock` 堆叠，每个 Block 内部包含：
  1. `WindowAttention`：基于输入 `attn`（直接拷贝自 CLIP 对应 block 的 self-attention）构建窗口注意力。HFWA 采用 `value_only=True`，即只用 value 向量进行基于余弦的注意力，适合冻结 query/key 的设定。
  2. `window_partition` / `window_reverse`：将 `[B, H×W, C]` 的 patch token 切分成 `(window_size, window_size)`，再在 SW-MSA 中做平移。
  3. MLP (`Linear`)：可通过 `hidden_features` 指定额外层数；默认等同于单层 `nn.Linear(dim, out_features)`。
- `configs/model/sowa_hfwa.yaml` 中的默认参数：
  - `input_resolution: [24, 24]`（对应 336 / 14）。
  - `window_size: 12`，`depth: 1`（可改成 >1 以启用 shift-window）。
  - `num_heads: 8`，`value_only: true`，`attn_drop: 0.1`。
  - `share_weight: true` —— 四个 stage 复用同一个 Adapter；若设为 `false` 或直接在配置里给 `adapter` 一个列表即可实现层间差异化。

| Stage（论文） | `feature_map_idx` | token 分辨率（ViT-L/14@336） | 默认 Adapter | 说明 |
| --- | --- | --- | --- | --- |
| H1 Soldier | 5 | 24×24 | BasicLayer（共享） | 捕捉高频纹理，主要响应浅层缺陷。 |
| H2 Soldier | 11 | 24×24 | 同上 | 深一点的语义纹理，仍在“Soldier”域。 |
| H3 Officer | 17 | 24×24 | 同上 | 更抽象的语义块。 |
| H4 Officer | 23 | 24×24 | 同上 | 最深层，与 CLS token 语义接近，用于补足宏观异常。 |

> 若需要在 H1/H2 使用更小 window、在 H3/H4 使用 shift-window，可在配置中为 `feature_map_idx` 提供等长的 Adapter 列表，例如：`adapter: [adapter_soldier, adapter_soldier, adapter_officer, adapter_officer]`。

### 3.3 文本分支：AnomalyPromptLearner + TextMapEncoder

- `AnomalyPromptLearner`（`src/models/components/coop.py`）实现 DualCoOp：
  - 为每个类别维护正/负两套 learnable context（`ctx_pos`, `ctx_neg`），初始长度 `prompt_length=12`。
  - `state_template` 控制提示句尾部（如 normal: `"{}"`, anomaly: `"damaged {}"`），生成的完整句子为 `[CLS] ctx... template(class)`。
  - `forward()` 返回 `(prompts, tokenized_prompts)`，shape 为 `(2 * M * K, context_length, dim)` / `(2 * M * K, context_length)`。
- `TextMapEncoder`（`src/models/components/text_encoder.py`）使用与视觉相同的 hook 机制：
  - `feature_map_idx` 默认 `[-1]`（只取最后一层），可根据需要抓取 transformer 中间层。
  - 返回 `(cls_feature, feature_maps)`，其中 `cls_feature` shape `(2 * M * K, C)`，`feature_maps` 列表可用于分析文本 token 的分层特征。

### 3.4 多模态融合：FusionModule + DotProductFusion

文件：`src/models/components/cross_modal.py`

- `FusionModule` 负责根据 batch 内的 `cls_name` 选择对应的文本向量：
  1. `pool_text_features`：将 `(2 * M * K, C)` reshape → `(M, 2, K, C)` 并对模板 K 求均值，得到 `(M, 2, C)` 的 normal / anomaly 表示。
  2. `get_batch_text_features`：按 `cls_names` 取出 `(B, 2, C)`。
  3. 将 `(B, 2, C)` 与视觉特征送入 `fusion_method`（默认为 `DotProductFusion`）。
- `DotProductFusion` 会对文本/视觉特征分别做 L2 归一化，然后进行：
  - CLS：`(B, C) @ (B, 2, C)^T → (B, 2)`。
  - Patch：`einsum('blc,bkc->blk') → (B, L, 2)`。

### 3.5 前向输出与推理

`AnomalyCLIP.forward(images, cls_name)` 返回六个对象：
1. `image_features`: `[B, C]` CLS 向量。
2. `text_features`: `[2 * M * K, C]`。
3. `patches`: `len(feature_map_idx)` 个 `List[Tensor[B, L, C]]`，为未适配的 patch token。
4. `patch_feature_maps`: 同长度列表，包含 HFWA 适配后的 token。
5. `anomaly_maps`: 每个 stage 生成 `(B, 2, H, W)`，完成 `softmax` 后可直接监督。
6. `text_probs`: `(B, 2)`，用于图像级 normal/anomaly 分类。

## 4. Lightning 模块与训练/推理循环

文件：`src/models/anomaly_clip_module.py`

- **初始化**：
  - `self.net = AnomalyCLIP(...)`，其余超参通过 `self.save_hyperparameters` 记录。
  - 损失函数：`CrossEntropyLoss`（图像级）、`FocalLoss` + `BinaryDiceLoss`（像素级 normal/anomaly 通道）。
  - 指标：`torchmetrics.Accuracy` + `MulticlassAUROC`（图像级），像素级指标在测试后集中计算。
- **训练/验证步骤**：
  - `model_step` 解包 batch，直接调用 `self.forward(images, cls_name)`。
  - `training_step` 聚合损失并记录 `train/loss_*`；`validation_step` 同理但仅在 `enable_validation=true` 时执行。
  - `on_train_batch_start` 记录当前学习率；`on_validation_epoch_end` 维护 best acc/AUROC。
- **测试阶段**：
  - `test_step` 额外收集 `pr_sp`、`anomaly_maps`、`image_mask`，并在 `filter=true` 时对像素分数应用 `scipy.ndimage.gaussian_filter(sigma=4)`。
  - 结束时 `on_test_epoch_end` 统计 image/pixel AUROC、AP、F1、AUPRO，并用 `tabulate` 打印表格。
- **优化器/调度器**：由 Hydra 注入（AdamW + ReduceLROnPlateau）。`configure_optimizers` 自动读取 `self.hparams.optimizer/scheduler`。
- **编译/AMP**：`compile=true` 时在 `setup('fit')` 调用 `torch.compile`；BF16/AMP 由 Trainer 配置。

### 4.1 Few-shot 记忆流程

1. 配置层：将 `model.k_shot` 设为 `true`，并确保数据配置提供 `dataset.kshot`。
2. 运行时：`on_test_start` 调用 `self.trainer.datamodule.kshot_dataloader()`，对每个类别运行 `kshot_step`，缓存 `self.mem_features[cls_name] = List[Tensor[k, L, C]]`。
3. `kshot_anomaly`：
   - 将支持集 patch 展开为 `(k * L, C)`，与当前 batch patch 做余弦距离。
   - 取最小距离作为 anomaly score，恢复为 `(1, 1, H, H)` 并插值到 `image_size`。
4. `test_step` 中若启用 Few-shot，会把主干 `anomaly_map` 与 `kshot_anomaly_map` 相加，再参与指标与可视化。

## 5. 回调、日志与脚本

- **回调**：
  - `configs/callbacks/*.yaml` 选择 Lightning 内建的 `ModelCheckpoint`, `EarlyStopping`, `RichModelSummary`, `RichProgressBar`。
  - `src/models/components/callback.AnomalyVisualizationCallback`：`visualize=true` 时保存 `(原图, GT, heatmap)`，路径由配置 `callbacks.visualization.dirpath` 提供。
- **日志**：`configs/logger/*.yaml` 支持 CSV、W&B、TensorBoard；`logger.wandb.offline=true` 可离线保存。
- **脚本**：`scripts/infer_auto.sh` / `infer_data.sh` / `schedule.sh` 直接调用 `python src/eval.py ...`，可作为集群任务模板。

## 6. 模块 API 速查

| 模块 / 函数 | 文件 | 关键参数 | 返回 / 备注 |
| --- | --- | --- | --- |
| `AnomalyCLIP(images, cls_name)` | `src/models/components/anomaly_clip.py` | `images: Tensor[B,3,H,W]`, `cls_name: List[str]` | `(image_features, text_features, raw_patches, adapted_patches, anomaly_maps, text_probs)` |
| `VisionMapEncoder(inputs)` | `src/models/components/vision_encoder.py` | `adapter`, `feature_map_idx`, `share_weight` | `(image_feature, raw_patch_tokens, hfwa_patch_tokens)` |
| `BasicLayer(attn, input_resolution, ...)` | `src/models/components/adapter.py` | `window_size`, `depth`, `num_heads`, `value_only`, `drop/attn_drop` | 复用 CLIP Self-Attention 做窗口化适配，返回 `[B, L, C]` |
| `AnomalyPromptLearner()` | `src/models/components/coop.py` | `class_names`, `prompt_length`, `state_template` | `(prompts, tokenized_prompts)`，用于 `TextMapEncoder` |
| `FusionModule(text_feat, vision_feat, cls_names)` | `src/models/components/cross_modal.py` | `fusion_method`, `embedding_dim` | `(B, 2)` 或 `(B, L, 2)` logits |
| `AnomalyCLIPModule.training_step(batch, batch_idx)` | `src/models/anomaly_clip_module.py` | `loss_ce/focal/dice`，`filter`, `k_shot` | 计算多任务损失并记录指标 |
| `AnomalyCLIPModule.kshot_step(batch)` | 同上 | 接受 `{'image': Tensor[k,...], 'cls_name': str}` | `Dict[cls_name -> List[patch_tokens]]` |
| `AnomalyVisualizationCallback.on_test_batch_end(...)` | `src/models/components/callback.py` | `visualize=true` | 将每个 batch 的输出保存到 `dirpath` |

## 7. 常见改动入口

1. **修改 HFWA 层**：在 `configs/model/sowa_hfwa.yaml` 调整 `feature_map_idx`、`adapter` 参数，或添加自定义 Adapter（实现 `nn.Module`，输入 `[B, L, C]`）。
2. **替换 Prompt 方案**：切换到 `PromptEncoder`（纯模板）或 `PromptTextEncoder`（零微调）时，只需在模型配置里改写 `prompt_learner`/`tokenizer` 指向。
3. **新数据集**：实现 `BaseDataset` 子类并在 `configs/data/*.yaml` 中替换相应路径/数据工厂；Few-shot 则实现 `KShotDataset` 子类。
4. **指标或可视化**：`src/utils/metrics.py`、`src/utils/vision_utils.py` 是集中修改点，新增指标后记得在 `on_test_epoch_end` 中读取。

通过以上拆解，可以快速定位 HFWA 模型中 Soldier/Officer 层、Window Adapter、Prompt 学习器以及 Lightning 插件化部分，便于定制新的注意力结构或引入不同的少样本记忆机制。***
