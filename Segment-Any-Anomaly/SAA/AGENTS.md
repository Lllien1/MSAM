# AGENTS.md

面向在 Segment-Any-Anomaly (SAA) 仓库内协作的代码智能体的速查指南。

## 仓库结构

- `model.py` - 核心 `Model` 类，串联 Grounding DINO（文本引导提案）、SAM（掩码细化）、ImageNet 显著性评估与置信度融合，主要推理逻辑与提示策略均在此实现。
- `modelinet.py` - 基于 TIMM 的特征提取与几何工具（如 `ResizeLongestSide`），为显著性与尺度对齐提供支持。
- `hybrid_prompts.py` - 数据集到 `manual_prompts`、`property_prompts` 的映射表，支持 VisA、MVTec、KSDD2、MTD 等。
- `prompts/` - 各数据集/类别专属的提示配置。
  - `general_prompts.py` 提供通用缺陷描述模板。
  - `{dataset}_parameters.py` 记录手工提示与属性约束。
- `CLAUDE.md` - 面向通用智能体的背景说明，可快速了解项目目标与约束。
- `__init__.py` - 包级入口，便于外部导入。

## 工作流速记

1. **加载提示**：通过 `hybrid_prompts` 调用不同数据集的提示配置。
2. **生成候选**：`Model.ensemble_text_guided_mask_proposal` 使用 Grounding DINO 输出框并依据属性提示过滤。
3. **细化掩码**：`Model.region_refine` 调用 SAM 将检测框转为像素掩码。
4. **显著性重评分**：`Model.saliency_prompting` 使用 `modelinet.py` 提供的特征工具计算相似度并调整分数。
5. **置信度融合**：`Model.confidence_prompting` 选取高分掩码合成为最终异常热图。

## 协作建议

- 目前默认 `batch_size = 1`，且属性提示作用于检测框级别；如需更改需同步评估整体影响。
- `model.py` 已包含详细中文注释，新增说明时保持同样的简洁、技术向风格。
- 新增数据集时，在 `prompts/` 下创建参数文件，并在 `hybrid_prompts.py` 注册即可。
- 若更改显著性分辨率或缩放策略，需同时更新 `modelinet.py` 中的相关工具函数以避免失配。
