# Agent Guide

Quick map of the SAM 3 repository for coding agents.

## Overview
- SAM 3 is an open-vocabulary segmentation/tracking model for images and videos.
- Python package name: `sam3`; default entrypoint `build_sam3_image_model()` in `sam3/model_builder.py`.
- Checkpoints are pulled from Hugging Face (requires `hf auth login`); GPU + CUDA 12.6 expected.
- Extras in `pyproject.toml`: `.[notebooks]`, `.[train]`, `.[dev]` for notebook, training, and dev tooling.

## Repository Layout
- `sam3/model_builder.py`: constructs image/video models, vision-language backbone, and decoders.
- `sam3/model/`: core architecture (encoder/decoder, geometry encoders, processors, video predictors, tokenizer, text encoder, etc.).
- `sam3/sam/`: RoPE transformer/prompt encoding pieces adapted from the SAM stack.
- `sam3/agent/`: agent pipeline (`agent_core.py`, `inference.py`), LLM/vision clients, system prompts, visualization helpers, geometry/mask utilities.
- `sam3/eval/`: evaluation pipelines (COCO, HOTA, TETA, SACO), dataset wrappers, converters, and demo eval scripts.
- `sam3/train/`: Hydra-driven training entrypoint `train/train.py`; configs in `train/configs/` (roboflow, odinw13, gold/silver image/video evals); data loaders in `train/data/`; transforms/loss/optim/utils subpackages.
- `sam3/perflib/`: performance utilities (Triton kernels, mask ops) plus minimal tests/assets.
- `scripts/`: helper scripts for dataset prep and metrics (`scripts/eval/veval`, `scripts/eval/silver`, `scripts/eval/gold`, etc.).
- `examples/`: Jupyter notebooks demonstrating inference, agent usage, and SACO eval/visualization flows.
- `assets/`: demo media (images, gifs, videos), vocab gzip, diagrams; `assets/videos/0001/` contains many sample frames.
- Top-level docs: `README.md` (setup/inference), `README_TRAIN.md` (training/eval), plus `scripts/eval/*/README.md`.

## Common Workflows
- **Inference**: build model via `build_sam3_image_model()`, then use `sam3.model.sam3_image_processor.Sam3Processor`; agentified flow in `sam3/agent/inference.py`. Ensure HF credentials are available to fetch checkpoints.
- **Training/Eval**: `python sam3/train/train.py -c <config.yaml>` (Hydra). Configs specify dataset paths/checkpoints; supports local or SLURM via SubmitIt. Logs/checkpoints go under the experiment log dir described in `README_TRAIN.md`.
- **Dataset/Eval utilities**: SACO/VEval prep and evaluation scripts live under `scripts/eval/`; see per-folder README files for expected inputs.
- **Notebooks**: install with `pip install -e ".[notebooks]"` before running anything in `examples/`.

## Notes for Agents
- Keep large assets (PDF, `assets/videos/**`) untouched; they are sample data only.
- License: MIT; code formatting uses Black/ruff/usort; mypy config present.
