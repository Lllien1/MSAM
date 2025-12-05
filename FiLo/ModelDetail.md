# ModelDetail —— FiLo 模型架构速查

## 1. 设计目标与总体思路
- 论文《FiLo: Zero-shot anomaly detection by fine-grained description and high-quality localization》（PDF 第1–2页）提出 **Fine-Grained Description (FG-Des)** 与 **High-Quality Localization (HQ-Loc)** 两大组件，旨在同时提升工业品零样本异常检测的识别精度与定位质量。
- 仓库实现中，FG-Des 通过自适应 Prompt + LLM 生成的异常描述来强化 CLIP 文本条件；HQ-Loc 则结合 Grounding DINO 粗定位、位置增强提示与多尺度/多形状的跨模态交互 (MMCI) 来抑制背景噪声并精细化像素级热力图（参见 `models/FiLo.py` 与 `test.py`）。

## 2. 论文模块与代码映射
### 2.1 FG-Des：细粒度描述
1. **异常词典构建**：论文利用 LLM 生成的类别级 defect 描述在代码中以字典形式硬编码（`models/FiLo.py:33-135`），并融合 WinCLIP 的通用模板（`status_abnormal_winclip`）。这些字符串即 FG-Des 中的 per-class prompts。
2. **Prompt 学习器**
   - `PromptLearner_normal`（`models/FiLo.py:156-247`）与 `PromptLearner_abnormal`（`models/FiLo.py:260-373`）为每个类别维护可学习的上下文 token，并用 `meta_net` 依据图像特征自适应调整 prompt。
   - 异常 Prompt 额外拼接位置短语（如 “at top left”），呼应 HQ-Loc 的位置增强提示。
3. **文本编码与特征归一**：`TextEncoder`（`models/FiLo.py:389-418`）直接复用 CLIP Transformer，输出与视觉特征同维度的文本向量并做 L2 正规化，完成 FG-Des 在代码层面的闭环。
4. **意义对应**：以上模块实现了论文中“可学习模板 + 细粒度描述”思想，确保不同类别/缺陷都能拥有独立语义锚点。

### 2.2 HQ-Loc：高质量定位
1. **Grounding DINO 粗定位**：`test.py:201-347` 定义了 `load_model`、`get_grounding_output` 等封装，利用文本描述（FG-Des 产物 + 通用异常词）生成候选框，过滤非异常区域（`test.py:480-545`）。该步骤对应论文 HQ-Loc 中“预定位”部分。
2. **位置增强的 Prompt 选择**：根据 Grounding DINO 的最大得分框中心，`test.py:533-559` 会定位 3×3 网格区域并将其作为 `FiLo.forward(..., positions=position)` 的约束；模型内部在 `FiLo.forward` 中筛选对应位置的异常 Prompt 子集（`models/FiLo.py:600-626`），契合论文描述的 location-enhanced prompts。
3. **多尺度多形状跨模态交互（MMCI）**：
   - `LinearLayer` 与 `CovLayer`（`models/FiLo.py:424-521`）分别对来自不同视觉层的 patch tokens 施加线性投影与 1×1/3×3/5×5/7×7/1×5/5×1 卷积组合，实现论文所述 multi-scale & multi-shape 融合。
   - 为了获得多层 patch tokens，定制版 OpenCLIP 在 `models/vv_open_clip/model.py:212-235` 中扩展了 `encode_image` 以返回 `self.visual(image, out_layers)`。
   - 在 `FiLo.forward` 中，这些特征与文本特征做内积得到像素概率图，并通过 `torch.softmax` + 上采样合成为 anomaly map 列表（`models/FiLo.py:636-701`）。
4. **分数融合**：最终会同时利用图像级文档概率 (`text_probs`) 与像素图统计（加框增强）作为 HQ-Loc 输出（`test.py:559-612`）。

### 2.3 Adapter 与双阶段训练
- `Adapter`（`models/FiLo.py:437-450`）为图像特征提供轻量残差调优层，以便在有限样本下快速对齐语义。
- `train.py` 首先训练解码器 + Prompt（`train.py:189-217`），再锁定主体仅优化 Adapter（`train.py:221-260`），对应论文附录中提到的轻量适配策略。
- 损失函数来自 `utils/loss.py`：FocalLoss + BinaryDiceLoss，分别约束双通道逻辑与像素分割质量，契合 HQ-Loc 对精细定位的需求。

## 3. 模块顺序（开发/封装接口速查）
1. **数据加载**：`datasets/mvtec_supervised.py` 与 `datasets/visa_supervised.py` 负责输出 `{'img','img_mask','cls_name','anomaly'...}`，并在 MVTec 模式下支持多图拼接增强（`combine_img`）。
2. **图像编码**：FiLo 在 `FiLo.forward` 内调用定制 OpenCLIP，输出 `[batch, tokens, dim]` 的主干特征与多层 patch tokens（`models/FiLo.py:546-573`）。
3. **生成 Prompt**：根据类别与 Grounding DINO 返回的位置，Normal/Abnormal Prompt Learner 生成批量文本序列，随后由 `TextEncoder` 得到文本向量（`models/FiLo.py:574-626`）。
4. **跨模态匹配与 MMCI**：`decoder_linear` + `decoder_cov` 处理不同比例 patch，内积后插值到整幅图，形成多尺度 anomaly maps（`models/FiLo.py:636-701`）。
5. **定位增强**：`test.py:511-612` 将 Grounding DINO 框映射到热力图上做放大/抑制，并与 `text_probs` 融合得到最终图像级分数。该结果亦写入 `results` 并被 `cal_score` 计算多种指标（`test.py:347-476` + `612-703`）。
6. **日志与可视化**：测试脚本会自动输出 `tabulate` 生成的表格并在 `ckpt/<dataset>_log.txt` 中记录所有参数与指标，方便对照论文表格。

## 4. 训练与推理要点
- **参数**：`train.py` 暴露 CLIP backbone、采样尺寸、特征层列表、学习率（主干/解码器/Adapter 独立设置）等关键超参，便于研究者尝试新组合。
- **损失与采样**：像素损失对 normal/abnormal 双通道都做 Dice 约束，需保证 `img_mask` 在 dataloader 中被二值化（代码中已处理）。
- **推理依赖**：需要提前准备 Grounding DINO 配置与 checkpoint，并确保 PDF 中列举的 prompt 词典与 `status_abnormal` 保持同步；否则 FG-Des/HQ-Loc 语义可能脱节。
- **性能验证**：脚本默认计算 AUROC_px/F1_px/AP_px/AUPRO + 图像级指标，指标计算通过多线程完成，若增加新类别需同步更新 `obj_list` 逻辑。

## 5. 扩展建议
1. **替换/扩充 Prompt**：可直接在 `status_normal`、`status_abnormal` 或字典中添加新语句以覆盖更多语义，再利用 `PromptLearner` 维持可学习上下文。
2. **自定义定位策略**：如果 Grounding DINO 不适合某些场景，可在 `test.py` 里替换 `get_grounding_output` 或调整 `location` 网格划分，以保持 HQ-Loc 的“位置约束”思想。
3. **改进 MMCI**：`CovLayer`/`LinearLayer` 模块支持任意 kernel 配置，适合尝试空洞卷积、注意力块等结构，只需保持 `tokens` 形状一致即可。
4. **新数据集接入**：沿用 `datasets/*` 的返回格式即可无缝接入 FiLo 训练/推理流程；若无 pixel mask，可在 `train.py` 中关闭像素损失或改为伪标签。

> 通过上述映射，开发者可以快速理解论文提出的 FG-Des/HQ-Loc 如何在仓库中落地，并据此进行改进、封装或接口扩展。
