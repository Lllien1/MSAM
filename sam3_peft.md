# SAM 3 工业异常微调方案（PEFT 优化版）

目标：在 MVTec-AD 上将 SAM 3 适配工业缺陷检测，最小改动但保证适配力。核心改进：明确 LoRA 绑定位置、对齐文本维度/掩码、增强提示语义、扩大可学范围并稳健训练与验证。

## 1) 总体思路
- **图像侧**：冻结主干，向 ViT-Det 的注意力 `qkv` 注入 LoRA（覆盖所有 window/global block）。可选在 `TransformerEncoderFusion` 的 cross-attn 再加 LoRA 提升文-图对齐。
- **文本侧**：使用冻结的 SAM3 TextEncoder 得到 1024 维 token，短词列表 → tokenizer → TextEncoder → 取 pooled/EOT，均值成“异常原型”，再拼 learnable ctx；统一过 `resizer(1024→256)`，扩展 attention mask。
- **解码侧**：Mask Decoder 全量训练；Neck/Fusion/Text resizer 解冻小学习率，避免表示瓶颈。
- **提示设计**：异常 = 类别词 + 缺陷短词均值 + learnable ctx；正常 = “normal/clean” + 类别词；可选保留单词 token 避免过度均值。

## 2) 关键实现细节
- **LoRA 绑定**
  - 目标模块：`sam3/model/vitdet.py` 的 `Block.attn.qkv`（需要自定义 target_modules，因为没有 q_proj/v_proj 命名）；可选 `TransformerEncoderFusion` 内的 cross-attn。
  - Rank 建议 16；检查绑定后参数量>0 以确认生效。
- **Prompt Learner**
  - 输入：短词列表（异常：缺陷词+类别；正常：normal/clean+类别）。
  - Text 编码：SAM3 tokenizer + TextEncoder (width=1024, layers=24, heads=16, ctx_len=32) → pooled/EOT (每词) → 均值 → 原型向量 `(1,1024)`.
  - Learnable ctx：`n_ctx=4`，形状 `(n_ctx,1024)`；拼接为 `(1,n_ctx+1,1024)`；再过 `resizer` → `(1,n_ctx+1,256)`。
  - Mask 同步：将 prompt 长度并入 `text_attention_mask`（可见为 False），防止 padding 被关注。
- **多尺度/视频兼容**
  - 若为多尺度 prompt，可为每个视觉尺度复用同一原型，或为尺度设独立 ctx；在 `encoder_extra_kwargs` 传入，cross-attn 时选择对应 prompt。
  - 视频/追踪时缓存 prompt，避免每帧重建。

## 3) 训练与超参
- **优化分组**
  - Group A：LoRA + Prompt ctx + 文字侧线性（如独立的 1024→256 投影），lr≈1e-3。
  - Group B：Mask Decoder + Neck + TransformerEncoderFusion + text resizer，lr≈1e-4~1e-5。
  - Weight decay 按需（prompt ctx 可设 0）。
- **损失**：`Total = α * Focal + β * Dice`，起始 α=5, β=1，监控正负样本平衡后再调；可加小权重 BCE 辅助稳定。
- **数据与 Prompt**
  - 参考 `mvtec_short_keywords`（保持短词），为正常样本显式加入正向词（normal/clean）。
  - 若 context_length 不够，需扩展 positional embedding 或截断 prompt。

## 4) 验证与调试清单
- 维度检查：Prompt 输出 `(B, n_ctx+1, 256)`，attention mask 同步长度；前向无 shape 错误。
- LoRA 生效：打印可训练参数，确保 qkv LoRA 权重存在且梯度非零。
- 快速验证：小 batch 正/负混合跑前向，监控 loss 不为常数；可用 5% 数据做 sanity check。
- 评估指标：按类报告 mIoU/F1，关注正常类别的漏检；若过拟合，调低 α 或增大正样本权重。

## 5) 待生成/修改的代码模块
- `utils/defect_definitions.py`：短词字典（保持原版）。
  - 异常提示：`[class] + defect_words`；正常提示：`["normal","clean", class]`。
- `dataset.py`：`MVTecSAMDataset` 返回 `(image, mask, prompt_list, is_anomaly, class_name)`；prompt_list 为短词列表。
- `model_components.py`：
  - `AveragedPromptLearner`：实现 Encode → Mean → Concat ctx → Resizer → Mask 更新。
  - `apply_lora_to_sam`：自定义 target_modules 指向 `Block.attn.qkv`（及可选 Fusion cross-attn）。
- `model_wrapper.py`：`FineTuneSAM3` 串联 Image Encoder(+LoRA)、Prompt Learner、Decoder，暴露前向接口。
- `train.py`：分组优化器、loss 组合、日志与验证脚本。

## 6) 风险与缓解
- LoRA 未命中：自定义匹配 qkv；训练前打印匹配结果。
- Prompt 过短/被截断：检查 tokenizer 长度与 positional embedding；必要时扩长 ctx_len。
- 过拟合/不收敛：调整 α/β，适度解冻 Neck/Fusion，小步长；正负样本平衡采样。

参考：SAM3 论文 Model Details、`sam3/model_builder.py`、`sam3/model/vitdet.py`、`sam3/model/text_encoder_ve.py`。该规范可直接指导后续代码修改与实现。  
