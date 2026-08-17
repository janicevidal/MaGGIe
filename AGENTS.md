# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `maggie/`. Put dataset readers and transforms in `maggie/dataloader/`, training and evaluation loops in `maggie/engine/`, model architectures and losses in `maggie/network/`, and shared helpers in `maggie/utils/`. `tools/main.py` is the distributed train/evaluate entry point. Experiment definitions belong in `configs/`; reusable evaluation wrappers live in `scripts/`. Documentation is under `docs/`, while `demo/`, `figs/`, and `onnx/` contain demo code, publication assets, and exported-model resources. Treat `output/` and `wandb/` as generated experiment data, not source.

## Build, Test, and Development Commands

Create the documented Python 3.8/CUDA environment, then install dependencies:

```bash
conda create -n maggie python=3.8 pip
conda activate maggie
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Run a single-GPU evaluation smoke test with:

```bash
torchrun --standalone --nproc_per_node=1 tools/main.py \
  --config configs/maggie_video.yaml --eval-only name smoke_test
```

Use `sh scripts/eval_image.sh configs/maggie_image.yaml 4 maggie` for the image benchmark and `sh scripts/eval_video.sh configs/maggie_video.yaml maggie` for video. Training uses the same entry point without `--eval-only`; pass overrides as key/value pairs, for example `--precision 16 name experiment model.weights ''`.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, `snake_case` for functions, variables, and modules, and `PascalCase` for classes. Group standard-library, third-party, and local imports. Keep tensor shape assumptions explicit in comments or docstrings, especially across batch, frame, instance, and spatial dimensions. No formatter or linter is configured, so keep changes focused and match neighboring code and YAML formatting.

## Testing Guidelines

There is currently no dedicated unit-test suite or coverage threshold. Validate changes with the smallest relevant `torchrun` evaluation, then run the applicable benchmark script for model, loss, decoder, or dataloader changes. Record the config, checkpoint, GPU count, and key metric changes in the pull request. New isolated utilities should include lightweight tests named `test_<behavior>.py` in a new `tests/` directory.

## Commit & Pull Request Guidelines

Recent history uses short, imperative summaries such as `Update loading weight` and `support Samurai`. Keep each commit scoped to one concern and avoid committing checkpoints, datasets, logs, or credentials. Pull requests should explain intent, list changed configs, provide reproduction commands, link related issues, and report evaluation results. Include before/after images or videos when output quality changes.

## Configuration & Data Safety

Do not hard-code local dataset paths, API keys, or W&B credentials. Put machine-specific paths in local config overrides, and follow `docs/DATASET.md` and `docs/MODEL_ZOO.md` for external data and weights.
