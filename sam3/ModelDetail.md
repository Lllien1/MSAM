# SAM3 模型速查（ModelDetail）
面向在仓库内快速定位论文模块与实现的开发者。

## 总览
- 任务：开放词表的检测/分割/跟踪，支持文本、点/框/掩码提示。
- 入口：`build_sam3_image_model()` 与 `build_sam3_video_model()`（`sam3/model_builder.py`）。
- 文本侧：VE tokenizer + Transformer 文本编码器（`sam3/model/tokenizer_ve.py`, `sam3/model/text_encoder_ve.py`）。
- 视觉侧：ViT 主干 + FPN/双颈（`sam3/model/vitdet.py`, `sam3/model/necks.py`），RoPE 支持。
- 融合与推理：VL 组合 `SAM3VLBackbone`，提示几何编码 `SequenceGeometryEncoder`，Transformer 编/解码器 + presence token（`sam3/model/encoder.py`, `sam3/model/decoder.py`），分割头 `UniversalSegmentationHead`（`sam3/model/maskformer_segmentation.py`）。
- 视频：检测器复用图像模型，外接掩码记忆式跟踪器（`sam3/model/sam3_video_inference.py`, `sam3/model/sam3_tracking_predictor.py`, `sam3/model/memory.py`）。

## 论文模块 ⇔ 代码定位
- **文本编码（开放词表概念理解）**  
  - 论文：文本编码器 + BPE tokenizer。  
  - 代码：`SimpleTokenizer` 读取 `assets/bpe_simple_vocab_16e6.txt.gz`；`VETextEncoder` 堆叠 24 层自注意力（宽度 1024，头数 16），输出 `language_features`/`language_mask`（`sam3/model/tokenizer_ve.py`, `sam3/model/text_encoder_ve.py`）。
- **视觉主干（统一图像/视频特征）**  
  - 论文：大分辨率 ViT，带绝对位置与 RoPE、全局块。  
  - 代码：`ViT`（32 层、embed_dim=1024、patch=14、RoPE）、`Sam3DualViTDetNeck` 生成多尺度特征并可选 SAM2 颈支持实例交互（`sam3/model/vitdet.py`, `sam3/model/necks.py`）。
- **视觉-语言对齐骨干**  
  - 论文：视觉与文本并行编码，后续多模态融合。  
  - 代码：`SAM3VLBackbone` 同时前向视觉与文本，输出图像特征、位置编码与文本 memory/mask（`sam3/model/vl_combiner.py`）。
- **几何提示编码（点/框/掩码/序列）**  
  - 论文：统一提示表示，加入位置编码与池化。  
  - 代码：`SequenceGeometryEncoder`（3 层 `TransformerEncoderLayer`，支持点/框直接投影与池化，位置编码 `PositionEmbeddingSine`；`sam3/model/geometry_encoders.py`, `sam3/model/position_encoding.py`）。
- **多模态 Transformer 编码器**  
  - 论文：文本池化后与视觉特征交叉注意力，对提示做早期融合。  
  - 代码：`TransformerEncoderFusion`（6 层，8 头，cross/self attention，支持 act checkpoint；`sam3/model/encoder.py`），封装于 `TransformerWrapper`。
- **检测/分割解码器（含 Presence Token）**  
  - 论文：解码器联合 box refine、presence token 预测“概念存在性”，O2O/O2M 查询、DAC。  
  - 代码：`TransformerDecoder`（6 层 `TransformerDecoderLayer`，200 查询，`dac=True`，`presence_token=True`，可选文本 cross-attn；`sam3/model/decoder.py`）。Presence token 通过 `presence_token_head` 产生 logits；下游与类/掩码分数联合。
- **掩码/分数生成**  
  - 论文：上采样掩码头、点乘评分。  
  - 代码：`UniversalSegmentationHead` + `PixelDecoder`（3 级上采样、跨提示注意力）；`DotProductScoring`/`MLP` 生成概念与掩码匹配得分（`sam3/model/maskformer_segmentation.py`, `sam3/model/model_misc.py`）。
- **实例交互（SAM1 任务）**  
  - 论文：交互式点/框修正。  
  - 代码：`SAM3InteractiveImagePredictor` 包装跟踪器做单帧交互（`sam3/model/sam1_task_predictor.py`），可通过 `enable_inst_interactivity=True` 挂载。
- **视频检测-跟踪解耦**  
  - 论文：检测器+掩码记忆跟踪器，时间消歧。  
  - 代码：检测器 `Sam3ImageOnVideoMultiGPU`（图像模型多 GPU 版，早期融合）；跟踪器 `Sam3TrackerPredictor` 使用 `SimpleMaskEncoder`（downsample + fuser）、RoPE 自/交注意力 (`TransformerDecoderLayerv2`)、`TransformerEncoderCrossAttention`；交互逻辑封装于 `Sam3VideoInferenceWithInstanceInteractivity`（热启动、NMS、关联阈值等；`sam3/model/sam3_video_inference.py`, `sam3/model/sam3_tracking_predictor.py`, `sam3/model/memory.py`）。
- **性能与内核**  
  - 论文：高效推理/大分辨率。  
  - 代码：自定义 Triton/FA kernels (`sam3/perflib/*`)、RoPEAttention (`sam3/sam/transformer.py`)，TensorFloat-32 自动开启（`sam3/model_builder.py`）。

## 架构执行顺序（图像 / 文本提示）
1) 输入预处理：`Sam3Processor` 将 PIL/ndarray 转 tensor，归一化；文本组装为字符串列表（`sam3/model/sam3_image_processor.py`）。  
2) 文本侧：`SimpleTokenizer` → `VETextEncoder` 输出 `language_features` (B, L, 256) 与 mask。  
3) 视觉侧：图像 → `ViT` 干 → `Sam3DualViTDetNeck` FPN，多尺度特征 + 位置编码。  
4) 几何提示：点/框/掩码 → `SequenceGeometryEncoder` 生成 prompt token。  
5) 融合编码：`SAM3VLBackbone` 输出视觉/文本特征；`TransformerEncoderFusion` 对文本池化并与视觉 cross-attn，支持提示早期融合。  
6) 解码：`TransformerDecoder`（O2O 查询，DAC，text cross-attn，可选 presence token）；输出 box 参考、隐状态、presence logits/feats。  
7) 掩码与打分：`UniversalSegmentationHead` 上采样掩码；`DotProductScoring` 计算概念-掩码得分，presence logit 与类别/掩码概率联合形成最终分数。  
8) 后处理：`Sam3Processor` 做阈值、NMS (`sam3/perflib/nms.py`)，输出 `masks/boxes/scores/logits`。

## 架构执行顺序（视频跟踪）
1) 图像检测：同上，`Sam3ImageOnVideoMultiGPU` 在每帧生成候选掩码/框/presence。  
2) 掩码记忆编码：`SimpleMaskEncoder` 对历史掩码下采样 + `SimpleFuser` 融合；位置编码 `PositionEmbeddingSine`。  
3) 时序 Transformer：`TransformerEncoderCrossAttention` + RoPEAttention 读取记忆与当前特征，产出更新的 track 查询。  
4) 关联与消歧：`Sam3VideoInferenceWithInstanceInteractivity` 根据 `assoc_iou_thresh/new_det_thresh/hotstart_*` 等参数做匹配、重识别与寿命管理。  
5) 输出：跟踪 ID + 掩码序列；可通过 `Sam3VideoPredictorMultiGPU` 进行多 GPU 分发。

## 关键入口与扩展提示
- 图像模型：`sam3/model_builder.py::build_sam3_image_model`（可选 `enable_inst_interactivity`/`compile`/自定义 `checkpoint_path`）。  
- 视频模型：`build_sam3_video_model`（`apply_temporal_disambiguation` 控制启发式关联）。  
- 推理封装：`sam3/model/sam3_image_processor.py`；视频预测器 `sam3/model/sam3_video_predictor.py`。  
- 训练：Hydra 配置位于 `sam3/train/configs/**`；入口 `sam3/train/train.py`。  
- 若修改模块：保持 `TransformerWrapper`、`SAM3VLBackbone` 接口不变可最小化对上层的破坏；presence token/score 与 `DotProductScoring`/`UniversalSegmentationHead` 紧密耦合，调整需同步。  
