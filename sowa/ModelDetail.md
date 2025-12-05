# ModelDetail

面向 SOWA 仓库的模型架构速查。本文档基于论文 “SOWA: Adapting Hierarchical Frozen Window Self-Attention to Visual-Language Models for Better Anomaly Detection”，同时结合仓库实现，说明各模块与代码位置，并补充 HFWA/FWA 对外 API，便于在其他模型中复用。

## 论文模块对照

### 1. Hierarchical Frozen Window Self-Attention (HFWA)
- **论文要点**：4.1 节将 CLIP ViT 按 H1-H4 四个 stage 组织，分别关注点/线/面/复合异常，并在每层后串联 FWA 适配器（Soldier: 浅层，Officer: 深层）。
- **仓库实现**：
  - `configs/model/sowa_hfwa.yaml` 通过 `feature_map_idx: [5, 11, 17, 23]` 指定四层 Transformer block，并用 `share_weight: true` 共享适配器权重。
  - `src/models/components/anomaly_clip.py` 中 `AnomalyCLIP` 组合 `VisionMapEncoder`、`AnomalyPromptLearner`、`FusionModule` 构建 HFWA 主干。
  - `VisionMapEncoder` (`src/models/components/vision_encoder.py`) 继承 `BaseEncoder` 自动注册 hook，`forward` 返回 `(cls_token, 原始 patch_features, 适配后 patch_feature_maps)` 三组张量，供分类/分割共享。
- **开发提示**：调整 `feature_map_idx` 或 `adapter` 超参即可增删 stage 或替换 Soldier/Officer 结构，CLIP 主干保持冻结以便快速迁移。

### 2. Frozen Window Self-Attention (FWA) 适配器
- **论文要点**：4.3 节提出 WindowPartition → 窗口内自注意力 → WindowReverse 流水线，并引入 Value-Value (VV) 注意力抑制无关背景，然后使用线性层对多层结果加权。
- **仓库实现**：
  - `src/models/components/adapter.py` 内 `window_partition`/`window_reverse`、`WindowAttention`、`SwinTransformerBlock`、`BasicLayer` 共同组成 FWA；`WindowAttention` 的 `value_only=True` 即 VV 注意力。
  - `configs/model/sowa_hfwa.yaml` 将 `BasicLayer` 以 `_partial_` 形式注入 `VisionMapEncoder`，可在配置中热切换 `input_resolution`、`window_size`、`num_heads`、`drop_path` 等参数。
- **开发提示**：若 Soldier 与 Officer 需要不同窗口/深度，可在配置中声明多组 adapter 并关闭 `share_weight`，即实例化多份 FWA。

### 3. Dual Learnable Prompts (DualCoOp)
- **论文要点**：4.2 节通过式 (3) 定义 `pn` (normal) 与 `pa` (abnormal) 两套可学习 token，模板使用 “abnormal [cls] / normal [cls]”，在 object-agnostic 场景下统一替换为 “object”。
- **仓库实现**：
  - `src/models/components/coop.py` 的 `AnomalyPromptLearner` 维护 `ctx_pos`/`ctx_neg` 可学习参数，`state_template` 由 `configs/model/sowa_hfwa.yaml` 提供。
  - `AnomalyCLIP.get_text_features` 调用 `prompt_learner()` 与 `TextMapEncoder`(`src/models/components/text_encoder.py`) 生成 2 × M × K 维文本特征，并交由 `FusionModule` 针对 batch 的 `cls_name` 聚合。
- **开发提示**：需要覆盖多类别或新的提示语义时，只需在配置中更新 `class_names`、`state_template` 或替换 `prompt_learner`，无需修改主干代码。

### 4. 视觉-语言对齐与多损失联合
- **论文要点**：式 (5) 中以 λ1-λ3 联合 Dice、Focal、BCE，缓解像素类别不平衡并强化边界。
- **仓库实现**：
  - `src/models/anomaly_clip_module.py` 在 `__init__` 中注入 `loss["cross_entropy"]`、`loss["focal"]`、`loss["dice"]`；`training_step` / `validation_step` 同时累计图像级与像素级损失。
  - `configs/model/sowa_hfwa.yaml` 的 `loss` 小节声明 `torch.nn.CrossEntropyLoss`、`FocalLoss`、`BinaryDiceLoss`。
  - `src/models/components/cross_modal.py` 的 `FusionModule` + `DotProductFusion` 负责文本/视觉特征的归一化点积，输出分类 logits 与像素概率，对应 4.4 节的视觉-语言对齐。

### 5. Few-shot Memory Bank 推理
- **论文要点**：4.5 节与附录 A 先用正常样本构建记忆库，再与测试图像多层特征比对生成 K-shot anomaly map，并与主分支结果融合。
- **仓库实现**：
  - `AnomalyCLIPModule.kshot_step`、`kshot_anomaly`、`on_test_start`、`test_step` (`src/models/anomaly_clip_module.py`) 结合 `src/data/components/kshot_dataset.py` 完成 Few-shot：预编码参考集 → 计算 patch 级余弦距离 → 与主分支得分求和。
  - `configs/model/sowa_hfwa.yaml` 的 `k_shot` 标志控制评测阶段是否启用记忆库。
- **开发提示**：如需自定义检索策略，可在 `kshot_anomaly` 中替换距离度量或融合方式，训练流程无需变动。

### 6. 其他关键衔接件
- `src/models/anomaly_clip_module.py` 测试阶段可选 `gaussian_filter` 平滑像素得分，对应论文中的噪声抑制。
- Hydra 顶层配置 (`configs/train.yaml`) 统一组织 data/model/callbacks/logger/trainer，确保任何模块都能在配置层替换。

## 论文架构（代码执行顺序）

1. **Hydra 初始化**：`python src/train.py model=sowa_hfwa` 实例化 `AnomalyCLIPModule`，注入优化器、调度器、损失与功能开关。
2. **视觉分支**：`AnomalyCLIP.forward` 先调用 `VisionMapEncoder`，CLIP ViT 各层输出由 hook 捕获并送入 `BasicLayer`(FWA)，再乘以 `clip_model.visual.proj` 对齐维度。
3. **文本分支**：`AnomalyPromptLearner` 生成 normal/abnormal 提示，`TextMapEncoder` 输出 token 级特征，`FusionModule` 按 batch 内 `cls_name` 聚合成 `(B,2,C)`。
4. **跨模态融合**：`FusionModule` → `DotProductFusion` 分别对 CLS token 与各层 patch map 做点积，得到图像级 logits 与像素级 anomaly map，并通过上采样 + softmax 输出概率图。
5. **多损失训练**：CE（图像级）+ Focal/Dice（像素级）共同反传，Lightning 记录 Accuracy、AUROC；`filter` 标志控制推理阶段是否平滑。
6. **Few-shot 扩展**：若 `k_shot=True`，`on_test_start` 预先遍历参考集 dataloader 构建 `mem_features`，`test_step` 将主分支 map 与 `kshot_anomaly` 结果逐图融合，再评估 AUROC/AP/AUPRO。
7. **结果汇总**：`on_test_epoch_end` 使用 `src/utils/metrics.py` 计算图像级/像素级指标，并通过 `tabulate` 输出表格，便于外部系统读取。

## HFWA / FWA API 规范

以下接口定义可直接复用于其他模型，所有形状遵循 PyTorch 约定，默认 dtype 为 `torch.float32`。

### 1. FWA (Frozen Window Attention) Adapter

| 接口 | 功能摘要 | 关键参数 | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| `window_partition(x, window_size)` / `window_reverse(...)` | 在 `[B,H,W,C]` 与 `(num_windows*B, window_size, window_size, C)` 之间分块/还原 | `window_size: int` | `[B,H,W,C]` 或窗口张量 | 对应分块/还原结果 |
| `WindowAttention(attn, window_size, num_heads, *, value_only=False, ...)` | 复制 CLIP resblock 的注意力权重，在窗口内执行注意力；`value_only=True` 即 VV 注意力 | `attn`, `window_size: tuple`, `num_heads`, `value_only`, `cpb_dim`, `attn_drop`, `proj_drop` | `[num_windows*B, N, C]`, `N = window_size^2` | `[num_windows*B, N, C]` |
| `SwinTransformerBlock(attn, input_resolution, window_size, num_heads, shift_size=0, ...)` | WindowAttention + MLP，支持平移窗口 | `input_resolution`, `window_size`, `shift_size`, `drop`, `attn_drop`, `drop_path` | `[B, L, C]` | `[B, L, C]` |
| `BasicLayer(attn, input_resolution, window_size, depth, num_heads, *, value_only=True, ...)` | 多层 `SwinTransformerBlock` 串联，构成论文定义的 FWA 适配器 | `depth`, `hidden_features`, `out_features` | `[B, L, C]` | `[B, L, out_features]` |

**嵌入步骤**
1. 从目标 ViT/CLIP 层提取原始 `attn`，使用 `functools.partial(BasicLayer, ...)` 预设窗口、头数、VV 等超参。
2. 将 `[B,L,C]` patch tokens 传入 `adapter(tokens)`，即可得到增强特征；若不同层需要不同配置，可实例化多份 adapter。
3. 需要恢复标准 QKV 注意力时，将 `value_only=False`。

### 2. HFWA (Hierarchical FWA) 编码器

| 接口 | 功能摘要 | 关键参数 | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| `VisionMapEncoder(clip_model, adapter, feature_map_idx, share_weight=False)` | 在 CLIP ViT 的 `feature_map_idx` 注册 hook，并对每层输出应用 FWA 适配器 | `clip_model`, `adapter`, `feature_map_idx: List[int]`, `share_weight` | `images: [B,3,H,W]` | `(cls_feature, raw_patch_maps, adapted_patch_maps)` |
| `AnomalyCLIP.forward(images, cls_name)` | HFWA + 文本提示 + 融合的完整前向，可在其他系统中复用其视觉部分 | `images`, `cls_name` | 图像批次 | `(image_features, text_features, patches, patch_feature_maps, anomaly_maps, text_probs)` |

**示例**
```python
from functools import partial
from src.models.components.adapter import BasicLayer
from src.models.components.vision_encoder import VisionMapEncoder

adapter_cfg = partial(
    BasicLayer,
    input_resolution=(24, 24),
    window_size=12,
    depth=1,
    num_heads=8,
    value_only=True,
    attn_drop=0.1,
)

vision_encoder = VisionMapEncoder(
    clip_model=clip_model,
    adapter=adapter_cfg,
    feature_map_idx=[5, 11, 17, 23],
    share_weight=True,
)

cls_feature, raw_maps, hfwa_maps = vision_encoder(images)
# hfwa_maps 可直接送入自定义融合/检测头
```

**注意事项**
- `feature_map_idx` 必须与 `input_resolution` 匹配，计算方式示例：ViT-L/14@336px → `(336/14) = 24`，对应 `[5,11,17,23]` 四层。
- `share_weight=False` 时，`VisionMapEncoder` 为每个层索引深拷贝一份 adapter，便于 Soldier/Officer 使用不同窗口大小或深度。
- `hfwa_maps` 仍是 patch token 空间，需要结合 `FusionModule` 或自定义 decoder 才能得到像素级概率。

### 3. 技术文档输出要求
1. 在项目 README 或设计文档中声明依赖项（CLIP 主干版本、PyTorch/Lightning、timm、hydra 等）。
2. 使用表格记录 `feature_map_idx`、`window_size`、`value_only`、`hidden_features` 等超参，方便复现。
3. 明确接口契约：输入/输出张量形状、dtype、是否允许梯度回传、显存需求。
4. 描述可扩展点：如替换跨模态融合策略、引入自定义 Few-shot 记忆、或开放 CLIP 主干微调的策略和注意事项。

按照以上 API 规范，即可在其他模型中快速嵌入 HFWA/FWA，保持 Soldier-Officer 分层注意力的语义，同时确保实现具有可维护性与可测试性。
