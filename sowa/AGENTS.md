# AGENTS

该仓库实现 Soldier-Officer Window self-Attention（SOWA），用于在 PyTorch Lightning/CLIP/Hydra 组合上执行少样本视觉异常检测。第一次接入项目时，可把本文件当作结构地图。

## 顶层布局

| Path | 说明 |
| --- | --- |
| `src/` | 核心 Python 包，包含 Lightning 入口（`train.py`、`eval.py`）。子目录：`data/`（DataModule 与数据组件）、`models/`（LightningModule 及各类适配器/编码器/提示模块）、`utils/`（rootutils 初始化、日志、指标、视觉辅助函数）。 |
| `configs/` | Hydra 配置树。根部的 `train.yaml` / `eval.yaml` 负责拼装 data/model/trainer/callbacks/logger/prompt/sweeps 以及本地可选覆盖。各子目录（如 `data/`、`model/`、`prompt/`、`trainer/`）存放可复用的配置组。 |
| `scripts/` | Shell 脚本：`infer_auto.sh`、`infer_data.sh` 用于推理批处理，`schedule.sh` 用于定时任务，调用方式与 `src/train.py` / `src/eval.py` CLI 参数一致。 |
| `tests/` | Pytest 套件，覆盖配置组合、DataModule、训练/评估脚本、超参搜索。`tests/helpers/` 提供条件跳过、包可用性检测与 shell helper。 |
| `notebooks/` | 预留的探索区（`.gitkeep`）。 |
| 根目录文件 | `.env.example`、`requirements.txt`、`environment.yaml`、`Makefile`、`pyproject.toml`、`.pre-commit-config.yaml` 管理环境、依赖、格式化与测试；`.project-root` 供 `rootutils` 解析路径。 |

## 核心工作流

- **训练：** `python src/train.py trainer=gpu data=sowa_mvt model=sowa_hfwa`（或 `make train`）。可通过 Hydra CLI 覆盖任意配置，例如 `model.k_shot=true`。
- **评估 / 推理：** `python src/eval.py trainer=gpu data=sowa_visa model=sowa_hfwa ckpt_path=...`。
- **数据准备：** 按 README 所引的 HuggingFace `data_scripts` 处理数据集，并在 `configs/data/*.yaml` 或本地覆盖中设置路径。
- **自动化：** 使用 `scripts/` 中的脚本进行批量推理或调度，`Makefile` 提供 `test` / `test-full` / `format` / `clean*` 等常用目标。

## Hydra 配置导览

1. 根配置 `configs/train.yaml`、`configs/eval.yaml`：定义默认堆栈，并声明可选的 experiment、sweeps、local、debug 覆盖。
2. `configs/data/`：数据根路径、增强、k-shot 设定（如 `sowa_mvt.yaml`、`sowa_visa.yaml`）。
3. `configs/model/`：CLIP 架构、适配器、提示模板、优化器/调度器等。
4. `configs/prompt/`：可学习提示及类别模版。
5. `configs/trainer/`、`configs/callbacks/`、`configs/logger/`：Lightning Trainer 参数、监控与日志后端。
6. `configs/hparams_search/`：基于 Optuna 的超参搜索示例。
7. `configs/local/`：机器/用户专属覆盖（gitignore），用于自定义路径或资源限制。

## 核心代码说明

- `src/train.py` / `src/eval.py`：Hydra CLI，负责 root 初始化、实例化 DataModule + Model，并交给 Lightning `Trainer`。
- `src/data/anomaly_clip_datamodule.py`：封装数据加载、变换、批处理，依赖 `src/data/components/`（`anomal_dataset.py`、`kshot_dataset.py`、`transform.py`）。
- `src/models/anomaly_clip_module.py`：LightningModule，对接 CLIP Backbone、提示适配器、异常评分与损失；构件位于 `src/models/components/`（adapter、encoder、prompt head、scheduler、loss、`components/clip/` 里的 tokenizer/模型权重）。
- `src/utils/`：日志（`pylogger.py`、`rich_utils.py`）、实例化（`instantiators.py`）、指标（`metrics.py`）与视觉工具（`vision_utils.py`）等通用函数。

## 辅助资源

- `tests/`：执行 `pytest`（或 `make test`）。套件确保 Hydra 配置可组合、DataModule 能初始化、训练/评估脚本与 sweeps 定义保持可用。
- `requirements.txt` / `environment.yaml`：锁定 PyTorch ≥2.0、Lightning ≥2.0、Hydra 1.3、TorchMetrics、timm、scikit-image/learn、OpenCV 等依赖。
- `.pre-commit-config.yaml`：提交前格式/静态检查；使用 `pre-commit run -a` 或 `make format`。
- `LICENSE`：Apache-2.0（沿用上游模板）。
- `.idea/`：JetBrains 工程文件，非必须关注。

## 代理速查步骤

1. 使用 Conda（`conda env create -f environment.yaml -n sowa`）或 pip（`pip install -r requirements.txt`）创建环境。
2. 确认 `configs/data/*.yaml` 中的数据路径可用；如需修改，创建本地 Hydra 覆盖。
3. 按需运行 `python src/train.py ...` / `python src/eval.py ...` 并通过 CLI 传入 dataset/prompt/checkpoint 等覆写。
4. 修改推理流程后，执行 `pytest`（或 `make test`）以及相关 `scripts/*.sh` 做验证。
5. 若新增配置或模型变体，记得同步更新对应 Hydra 组与测试，保持组合有效。

