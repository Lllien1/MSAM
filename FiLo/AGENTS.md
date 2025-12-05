# FiLo 仓库导航（AGENTS）

## 项目概览
- 本仓库实现了论文《FiLo: Zero-shot anomaly detection by fine-grained description and high-quality localization》，核心思想是通过细粒度文字描述（FG-Des）与高质量定位（HQ-Loc）实现工业品零样本异常检测。
- 代码基于 PyTorch，结合 Grounding DINO（粗定位）、定制化 OpenCLIP（多尺度视觉特征）与自适应 Prompt Learner（生成文本条件），并提供训练/测试脚本以及预训练权重。

## 目录结构速查
- `README.md`：论文简介、数据准备、训练/测试说明。
- `requirements.txt`：运行依赖（PyTorch、Grounding DINO、OpenCLIP 生态等）。
- `train.py` / `train.sh`：两阶段训练脚本（先训练解码器+Prompt，再单独微调 Adapter）。
- `test.py` / `test.sh`：推理脚本，串联 Grounding DINO、定位提示与 FiLo 推理，并输出指标日志。
- `models/`
  - `FiLo.py`：主模型定义，包含 PromptLearner（normal/abnormal）、TextEncoder、Adapter、MMCI 解码器等关键模块。
  - `vv_open_clip/`：定制 OpenCLIP 实现，`model.py` 的 `encode_image` 支持输出多层 patch tokens 供 MMCI 使用。
  - `GroundingDINO/`：第三方库（含 config、datasets、demo 等）用于生成候选框。
- `datasets/`：`mvtec_supervised.py`、`visa_supervised.py` 封装数据集，支持 mask/类别信息与多图拼接增强。
- `data/`：数据与元信息脚本（如 `mvtec.py`、`visa.py`），部分样例数据、`meta.json`、以及 dataset 原始目录。
- `ckpt/`：示例权重与日志（`filo_train_on_mvtec.pth`、`grounding_train_on_mvtec.pth` 等）。
- `models/FiLo.py` 同级还包含大权重文件（`filo_train_on_mvtec.pth`、`grounding_train_on_mvtec.pth`）供快速体验。
- `utils/loss.py`：训练阶段使用的 FocalLoss、BinaryDiceLoss 及辅助正则。
- 其他：`.idea/`（IDE 配置）、`figs/`（论文插图）、`datasets/`、`models/` 等均为源码支持目录。

## 资源文件
- `Gu 等 - 2024 - FiLo Zero-shot anomaly detection by fine-grained description and high-quality localization.pdf`：论文 PDF，描述 FG-Des 与 HQ-Loc 原理。
- `ckpt/` 与根目录权重：测试脚本默认读取的 FiLo/Grounding DINO 预训练模型。

## 开发入口
- 训练：编辑 `train.sh` 或直接运行 `python train.py --dataset ...`，确保 `data/<dataset>/meta.json` 和权重路径就绪。
- 推理：参考 `test.sh`，准备 Grounding DINO 配置与 FiLo 权重后运行 `python test.py --dataset ...`。
- 扩展研究：在 `models/FiLo.py` 中可替换 Prompt、Adapter、MMCI 结构；`test.py` 可调整定位策略、评价指标；`datasets/` 可添加新数据管道。
