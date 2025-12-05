# SAM3 Perception Encoder vs FiLo OpenCLIP

## Key deltas
- **Image encoder footprint**: SAM3 uses a 32-layer ViT-Det backbone with mixed global/window attention and RoPE on 1008 px inputs, then a dual-neck FPN to 4 scales (256-d). FiLo keeps OpenCLIP ViT-L/14-336 (24 global-attention layers, no neck) and exposes four 1024-d patch-token taps.
- **Positional handling**: SAM3 combines tiled absolute positions (pretrained at 336) with optional RoPE inside windowed blocks; FiLo only uses learned absolute positions with a persistent CLS token.
- **Attention pattern**: SAM3 alternates window attention (size 24) with global blocks at indices 7/15/23/31; FiLo runs full global self-attention uniformly.
- **Text pipeline**: SAM3 text tower is a 24-layer, width-1024 causal Transformer (context 32) whose tokens are linearly resized to 256-d for fusion. FiLo’s OpenCLIP text tower is 12 layers, width 768 (context 77) with an EOT pooling + projection to 768-d.

## Image encoder stack

| Component | SAM3 Perception Encoder (paths) | FiLo OpenCLIP (paths) | Delta |
| --- | --- | --- | --- |
| Input & patch embed | ViT patch embed, kernel/stride 14 → 72×72 grid for 1008×1008 input, bias-less conv, LN-pre (`sam3/model_builder.py:_create_vit_backbone`, `sam3/model/vitdet.py`) | Conv patch embed 14×14 on 336×336 → 24×24 grid, standard biasless conv, LN-pre (`FiLo/models/vv_open_clip/model_configs/ViT-L-14-336.json`, `vv_open_clip/transformer.py`) | SAM3 trains/evals at 3× larger resolution; tiling pretrain abs-pos |
| Positional enc | Learned abs-pos initialized at pretrain size 336, tiled to 1008; RoPE enabled in blocks; CLS not retained (`vitdet.py`) | Learned abs-pos added once to CLS+patch tokens; no RoPE; CLS kept throughout (`vv_open_clip/transformer.py`) | SAM3 mixes absolute + rotary; FiLo only absolute |
| Block depth & width | 32 Transformer blocks, dim 1024, heads 16, MLP ratio 4.625, drop-path 0.1 (`_create_vit_backbone`) | 24 Transformer blocks, dim 1024, heads 16, MLP ratio 4.0, no drop-path set (`VisionTransformer`) | SAM3 is deeper with slightly wider MLP |
| Attention type per block | Window attention (size 24) except global blocks at indices 7, 15, 23, 31; uses RoPE inside windows (`vitdet.Block`) | Full global self-attention in every layer; no windowing | SAM3 alternates local/global receptive fields; FiLo stays global |
| Normalization & residual layout | LN before blocks; optional layer scale per block; LN-post disabled; no CLS in output; outputs feature maps reshaped to H×W | LN-pre; CLS preserved; outputs sequence tokens; minor “attention surgery” on last blocks to clone attention weights (`VisionTransformer.forward`) | SAM3 produces pure grid tokens; FiLo keeps CLS and reshapes only when consumers request |
| Multi-scale neck | Sam3DualViTDetNeck upsamples/downsamples final stage to 4 scales (×4, ×2, ×1, ×0.5) with conv-transpose/conv + positional enc, all to 256-d (`sam3/model/necks.py`) | None; FiLo directly uses four intermediate patch-token sets (layers 6/12/18/24) at 1024-d without FPN (`FiLo/models/FiLo.py` calls `encode_image(out_layers=[6,12,18,24])`) | SAM3 builds an FPN; FiLo returns raw multi-layer tokens |

## Text encoder stack

| Component | SAM3 Text Encoder (paths) | FiLo OpenCLIP Text Encoder (paths) | Delta |
| --- | --- | --- | --- |
| Tokenizer & context | BPE vocab 49,408, context length 32 (`sam3/assets/bpe_simple_vocab_16e6.txt.gz`, `model_builder.py:_create_text_encoder`) | BPE vocab 49,408, context length 77 (`vv_open_clip/model_configs/ViT-L-14-336.json`) | SAM3 shortens context to 32 for tracking/segmentation prompts |
| Embedding width / heads / layers | Width 1024, 16 heads, 24 layers, causal mask; LN-post optional (enabled), returns tokens (`sam3/model/text_encoder_ve.py:164-238`) | Width 768, 12 heads, 12 layers, causal mask; LN-post; returns pooled EOT vector (`vv_open_clip/transformer.py:540-620`, `model.py:211-241`) | SAM3 is 2× deeper and wider |
| Positional strategy | Learned positional embedding length 32; no CLS; pooling type “none” so tokens preserved (`text_encoder_ve.TextTransformer`) | Learned positional embedding length 77; optional CLS support; uses EOT argmax pooling | FiLo collapses to a single sentence embedding; SAM3 keeps full token map |
| Projection / output dim | Linear resizer 1024 → 256 to match detection transformer; outputs attention mask + resized tokens (`VETextEncoder.forward`) | Text projection matrix 768 → 768 (model embed_dim); outputs normalized vector (`CustomCLIP.encode_text`) | SAM3 downsizes for fusion; FiLo stays at contrastive dimension |

## Source references
- SAM3 image encoder: `sam3/model_builder.py` (`_create_vit_backbone`, `_create_vit_neck`), `sam3/model/vitdet.py`, `sam3/model/necks.py`.
- SAM3 text encoder: `sam3/model_builder.py:_create_text_encoder`, `sam3/model/text_encoder_ve.py`.
- FiLo OpenCLIP image/text: `FiLo/models/vv_open_clip/model_configs/ViT-L-14-336.json`, `FiLo/models/vv_open_clip/transformer.py`, `FiLo/models/vv_open_clip/model.py`.
# SAM3 感知编码器 vs FiLo OpenCLIP

## 主要差异
- **图像编码器规模**：SAM3 采用 32 层 ViT-Det 主干，混合全局/窗口注意力并在 1008px 输入上启用 RoPE，随后用双颈 FPN 生成 4 个尺度的 256 维特征。FiLo 使用 OpenCLIP ViT-L/14-336（24 层全局注意力，无颈），直接暴露 4 份 1024 维 patch token。
- **位置编码**：SAM3 将预训练于 336 的绝对位置嵌入平铺到 1008，并在窗口块内使用 RoPE；FiLo 只使用学习的绝对位置并保留 CLS。
- **注意力模式**：SAM3 在第 7/15/23/31 层使用全局注意力，其余为 24×24 窗口注意力；FiLo 每层均为全局自注意力。
- **文本管线**：SAM3 文本塔为 24 层、宽度 1024、上下文 32 的因果 Transformer，输出 token 并线性缩放到 256 维以便融合。FiLo 的 OpenCLIP 文本塔为 12 层、宽度 768、上下文 77，使用 EOT 池化投影到 768 维。

## 图像编码器堆栈

| 组件 | SAM3 感知编码器（路径） | FiLo OpenCLIP（路径） | 差异 |
| --- | --- | --- | --- |
| 输入与补丁嵌入 | ViT 补丁嵌入，卷积核/步长 14 → 1008×1008 得到 72×72 网格，无偏置卷积，LN-pre（`sam3/model_builder.py:_create_vit_backbone`, `sam3/model/vitdet.py`） | 14×14 卷积补丁嵌入，336×336 → 24×24 网格，标准无偏置卷积，LN-pre（`FiLo/models/vv_open_clip/model_configs/ViT-L-14-336.json`, `vv_open_clip/transformer.py`） | SAM3 训练/推理分辨率是 FiLo 的 3×；平铺预训练绝对位置 |
| 位置编码 | 预训练尺寸 336 的学习绝对位置，平铺到 1008；窗口块内启用 RoPE；不保留 CLS（`vitdet.py`） | 学习绝对位置一次性加到 CLS+patch token；无 RoPE；CLS 保留（`vv_open_clip/transformer.py`） | SAM3 同时用绝对+旋转；FiLo 仅绝对 |
| 深度与宽度 | 32 层，维度 1024，16 头，MLP 比 4.625，drop-path 0.1（`_create_vit_backbone`） | 24 层，维度 1024，16 头，MLP 比 4.0，未设置 drop-path（`VisionTransformer`） | SAM3 更深且 MLP 更宽 |
| 每层注意力类型 | 7/15/23/31 层全局注意力，其余窗口注意力（24×24），窗口内用 RoPE（`vitdet.Block`） | 全局自注意力贯穿所有层；无窗口 | SAM3 交替局部/全局感受野；FiLo 全局 |
| 归一化与残差布局 | 块前 LN，可选 LayerScale；LN-post 关闭；输出不含 CLS，只保留网格特征 | LN-pre；CLS 保留；输出序列 token；末端对注意力做 “surgery” 克隆权重（`VisionTransformer.forward`） | SAM3 输出纯网格；FiLo 保留 CLS 并按需重排 |
| 多尺度颈部 | Sam3DualViTDetNeck 将最终阶段上采样/下采样到 4 个尺度（×4/×2/×1/×0.5），转置卷积/卷积 + 位置编码，通道 256（`sam3/model/necks.py`） | 无颈；FiLo 直接返回四个中间层 1024 维 patch token（`FiLo/models/FiLo.py` 调用 `encode_image(out_layers=[6,12,18,24])`） | SAM3 构建 FPN；FiLo 直接用多层 token |

## 文本编码器堆栈

| 组件 | SAM3 文本编码器（路径） | FiLo OpenCLIP 文本编码器（路径） | 差异 |
| --- | --- | --- | --- |
| 分词与上下文 | BPE 词表 49,408，上下文长度 32（`sam3/assets/bpe_simple_vocab_16e6.txt.gz`, `model_builder.py:_create_text_encoder`） | BPE 词表 49,408，上下文长度 77（`vv_open_clip/model_configs/ViT-L-14-336.json`） | SAM3 缩短上下文以适配跟踪/分割 prompt |
| 宽度/头数/层数 | 宽 1024，16 头，24 层，因果掩码；可选 LN-post（启用），返回 tokens（`sam3/model/text_encoder_ve.py:164-238`） | 宽 768，12 头，12 层，因果掩码；LN-post；返回 EOT 池化向量（`vv_open_clip/transformer.py:540-620`, `model.py:211-241`） | SAM3 更深更宽 |
| 位置策略 | 学习的位置嵌入长度 32；无 CLS；池化类型 “none”，保留全部 token（`text_encoder_ve.TextTransformer`） | 学习的位置嵌入长度 77；可选 CLS；使用 EOT argmax 池化 | FiLo 收敛为单句向量；SAM3 保留 token map |
| 投影 / 输出维度 | 线性缩放 1024 → 256 与检测 Transformer 对齐；输出注意力掩码+缩放后 tokens（`VETextEncoder.forward`） | 文本投影矩阵 768 → 768（模型对比维度）；输出归一化向量（`CustomCLIP.encode_text`） | SAM3 下采样便于融合；FiLo 保持对比学习维度 |

## 参考文件
- SAM3 图像编码器：`sam3/model_builder.py`（`_create_vit_backbone`, `_create_vit_neck`）、`sam3/model/vitdet.py`、`sam3/model/necks.py`。
- SAM3 文本编码器：`sam3/model_builder.py:_create_text_encoder`、`sam3/model/text_encoder_ve.py`。
- FiLo OpenCLIP 图像/文本：`FiLo/models/vv_open_clip/model_configs/ViT-L-14-336.json`、`FiLo/models/vv_open_clip/transformer.py`、`FiLo/models/vv_open_clip/model.py`。
