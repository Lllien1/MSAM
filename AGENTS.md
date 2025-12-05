# AGENTS — FiLo/SOWA/SAA 融合路线图

## 目标回顾
- **主体框架**仍以 `FiLo` (`FiLo/models/FiLo.py`, `FiLo/train.py`, `FiLo/test.py`) 的 FG-Des + HQ-Loc 流程为骨架，沿用其多尺度文本-视觉匹配与 MMCI 解码。
- **视觉主干替换**：用 `sowa` 中的 **Hierarchical Frozen Window Self-Attention (HFWA)** 组件（`sowa/src/models/components/vision_encoder.py`, `.../adapter.py`, `configs/model/sowa_hfwa.yaml`）替代 FiLo 中 `open_clip.encode_image(..., features_list)` 到 `decoder_linear/decoder_cov` 的这段路径。
- **混合提示约束**：把 `Segment-Any-Anomaly/SAA` 的 Property Prompt、Saliency Prompt、Confidence Prompt（`Segment-Any-Anomaly/SAA/model.py`）引入 HQ-Loc 阶段，用于重排/裁剪 MMCI 输出的像素热力图，从而在统一框架下形成全新的模型。

## 现有模块速记
### FiLo（`./FiLo`）
- **OpenCLIP 主干**：`FiLo/models/FiLo.py:533-576` 使用 `open_clip.create_model_and_transforms(args.clip_model='ViT-L-14-336')`，默认输入 336×336，`features_list=[6,12,18,24]`，返回 `[B,L,1024]` 的 patch tokens 与 CLS (`FiLo/train.py:69-144`, `test.py:441-538`)。
- **MMCI 解码**：`decoder_linear = LinearLayer(1024, 768, 4)` + `decoder_cov = CovLayer(1024, 768, 4)`（`FiLo/models/FiLo.py:424-521`）要求 4 组 1024 维 patch tokens（偶数层给 `CovLayer`，奇数层给 `LinearLayer`），并输出 768 维多尺度热力图后再 upsample（`FiLo/models/FiLo.py:636-701`）。
- **HQ-Loc**：`test.py:201-612` 结合 Grounding DINO 候选框、位置增强 prompt，最后把 `anomaly_maps` 投射到全图并依据框进行抑制/增强。

### HFWA（`./sowa`）
- **配置**：`configs/model/sowa_hfwa.yaml` 固定 `arch: ViT-L/14@336px`、`feature_map_idx: [5, 11, 17, 23]`、`window_size:12`，与 FiLo 的采样尺寸一致但 hook 索引为 block 级（非 FiLo 自定义的 `out_layers`）。
- **VisionMapEncoder**：`sowa/src/models/components/vision_encoder.py:11-79` 通过 hook 捕获指定 resblock 输出，得到 `feature_maps (B, L, 1024)`，并在 `self.proj`（CLIP 投影矩阵，`1024 -> 768`）后返回 `patches`；HFWA `adapter`（`.../adapter.BasicLayer`、基于 `WindowAttention`/`SwinTransformerBlock`）对每个层的 Soldier/Officer 窗口做 value-only attention，输出 `patch_features`（同样在投影后为 768 维）。
- **Lightning 模块**：`sowa/src/models/components/anomaly_clip.py` 将 `VisionMapEncoder` 输出的多尺度特征与 `AnomalyPromptLearner`、`FusionModule` 结合，整体维度约定为 768。

### Property/Saliency/Confidence Prompts（`./Segment-Any-Anomaly/SAA`）
- **Property Prompt**：`model.py:128-343` 中的 `set_property_text_prompts` + `bbox_suppression` 根据 `hybrid_prompts.py`/`prompts/*.py` 提供的 `object_count / area_ratio / k_mask` 规则对 Grounding DINO 的候选框做硬过滤（即论文中的 $\mathcal{P}^P$）。
- **Saliency Prompt**：`model.py:375-533` 依赖 `ModelINet` (`modelinet.py:11-181`) 生成 1024×1024 显著性/自相似图，再在 `rescore` 中重新加权缺陷掩码得分（论文 $\mathcal{P}^S$）。
- **Confidence Prompt**：`model.py:559-589` 选 top-`k_mask` 掩码做置信度融合输出最终热图（$\mathcal{P}^C$）。

## 融合蓝图（按执行顺序）
1. **统一 CLIP 主干实例**  
   - 仍使用 `open_clip.create_model_and_transforms('ViT-L-14-336')`，但只保留一次权重加载，并在 FiLo 初始化时同时构造 HFWA `VisionMapEncoder`。  
   - 事项：FiLo 依赖 `encode_image(image, out_layers)` 扩展，此接口目前在 `models/vv_open_clip/model.py:212` 通过 `out_layers` 返回多层 patch tokens。HFWA 则基于 resblock hook。需要一个桥接包装器，既能喂给 HFWA adapter，也能返回 FiLo 期望的 4 份未投影（1024 维） patch tokens。

2. **HFWA → MMCI 接口对齐**  
   - 计划：让 HFWA 输出两路特征  
     1. `raw_tokens`: 在 `VisionMapEncoder.get_feature_maps()` 后、投影前拷贝一份 1024 维 patch tokens，对齐 FiLo `decoder_linear/decoder_cov` 输入。  
     2. `hfwa_tokens`: HFWA adapter（窗口注意力）处理后的 1024 维增强特征；必要时在 FiLo 中新增开关，让 MMCI 可消化增强/原始特征或级联它们。  
   - 需注意：SOWA 默认在返回前执行 `@ self.proj`（输出 768）。若要保持 1024 维，需在新封装里跳过 `proj`（或在 FiLo 侧把 `decoder_*` 的 `dim_in` 改为 768 并对齐所有卷积权重）。**建议**：保留 FiLo 1024 维接口，在 HFWA 分支中多返回“pre-projection tokens”，避免大规模改 decoder 结构。

3. **MMCI 输出与属性提示联动**  
   - 目前 FiLo 的 HQ-Loc 在 `test.py:511-612` 用 Grounding DINO 框做加权。我们可以：  
     - 先复用 FiLo 的 DINO 流程生成 `boxes/logits`。  
     - 将这些候选喂入 `bbox_suppression` (`Segment-Any-Anomaly/SAA/model.py:267-343`)，把 $\mathcal{P}^P$ 的 `object_count`, `area_ratio`, `k_mask` 等规则转成一个 `BoxFilter`，然后把过滤后的框再传给 FiLo/mmci 以便抠出合法区域。  
     - 同时把 `defect_area_threshold = object_area * ratio` 逻辑引入 FiLo heatmap 裁剪（等价于对 MMCI 结果做面积上限约束）。

4. **引入显著性重加权（Saliency Prompt）**  
   - 接口：在 FiLo 生成每层 `anomaly_maps` 后，将其转换成 SAA 所需的 `defect_masks`（B×K×H×W）。然后调用 `saliency_prompting` (`model.py:375-533`)：  
     - `ModelINet` (`modelinet.py:11-181`) 会把原图 resize→1024 长边、提取多尺度特征并生成 `similarity_map`。  
     - 使用该 `similarity_map` 对 MMCI 热力图进行 `rescore`（指数放大/抑制），提升与显著区域的对齐度。  
   - 需新增桥接层，把 FiLo 的图像预处理（336 resize+normalize）同步给 ModelINet，保证 saliency 使用原分辨率图像。

5. **置信度融合输出（Confidence Prompt）**  
   - 采用 `confidence_prompting` (`model.py:559-589`) 逻辑，对 saliency 调整后的 MMCI 热力图进行排序，取 `k_mask`（由属性提示给出）做加权平均，并保留 `similarity_map` 作为诊断输出。  
   - 最终 FiLo 的 `anomaly_maps` 列表将被替换为：`raw_mmci`（诊断）、`saliency_rescored`、`confidence_fused`。其中 `confidence_fused` 才参与阈值化/指标计算。

6. **配置/Prompt 管理**  
   - `Segment-Any-Anomaly/SAA/prompts/*.py` & `hybrid_prompts.py` 维护数据集/类别到属性提示的映射。需要在 FiLo 的 `args` 中新增入口（或读取已有 `meta.json`）以选择对应 prompt 配置。  
   - 对于 FG-Des 已经存在的 per-class phrases，可将 SAA 的 `manual_prompts`/`property_prompts` 写入同一配置模块，避免三套词典各自维护。

7. **训练 & 推理流程**  
   - 训练阶段：HFWA 需要在前向中运行，但仍沿用 FiLo 两阶段训练（`train.py:187-260`）。必要时把 HFWA adapter 参数加入第一阶段可训练列表。  
   - 推理阶段：`test.py` 应在 Grounding DINO → Property Prompt → FiLo(HFWA) → Saliency/Confidence 之间串联，最后输出带显著性/置信度附加图，以便对照论文的贡献点。

## 兼容性/风险提示
- **维度不一致**：FiLo decoder 目前硬编码 `dim_in=1024`；HFWA 若只返回 768 维（经 `proj`）将导致崩溃。需要确保在 HFWA 包装器里导出 1024 维 tokens 或同步调整 MMCI。
- **Hook 索引差异**：FiLo 的 `features_list` 依据 `encode_image(..., out_layers)` 中的 patch 拼接顺序；HFWA 的 `[5,11,17,23]` 直接访问 `visual.transformer.resblocks`。需建立 mapping（例如把 `features_list` 改写为 HFWA 的索引，或提供配置项，以保证两者都覆盖同样数量/尺度的特征）。
- **显著性模型输入尺寸**：ModelINet 以 1024 为长边（`modelinet.py:63-114`），与 FiLo 336 预处理不同。为避免插值误差，应保留原分辨率图像副本供显著性/属性提示使用。
- **Prompt 版本管理**：FiLo 的 `status_abnormal` 直接写在代码里，SAA 的 prompt 则拆到多个 py 文件。建议在新框架中集中管理（例如 `config/prompts/*.yaml`），否则极易漂移。
- **推理性能**：SAA 的 saliency/confidence 目前默认单张图处理；集成后要关注 batch >1 的行为（FiLo 支持批处理）。必要时在桥接器中添加循环或 `torch.vmap`。

## 下一步建议
1. 编写 `open_clip` 包装器（或新 `hfwa_clip.py`）输出 `(cls, hfwa_tokens, raw_tokens)`，并在 `FiLo` 构造函数中注入该模块替换原先的 `encode_image(..., features_list)`。
2. 在 `test.py` 新增 `PropertyPromptController`，封装 `Segment-Any-Anomaly/SAA` 的属性/置信度逻辑，确保可以从命令行切换数据集的 prompt 配置。
3. 为 saliency/confidence 整理独立的 `Regularizer` 类，明确输入/输出 contract（`torch.Tensor` 形状、阈值设定），并添加单元测试验证与原 SAA 行为一致。
4. 更新项目文档（README 或新的 `docs/fi_sowa.md`）说明整合后的完整推理链路以及依赖（例如需额外安装 `timm`, `opencv-python-headless`，引用自 `Segment-Any-Anomaly`）。

通过本 AGENTS 文档，后续即可围绕“FiLo 主体 + HFWA 特征编码 + Property/Saliency/Confidence 提示约束”这一目标逐步重构与实现新的统一模型。 
