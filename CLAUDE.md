# CLAUDE.md - Lucida development guide

This repository contains Lucida, a Python background-removal application. It provides a CLI, a
FastAPI service, and a small browser UI. Keep changes focused on this repository and preserve
unrelated user changes.

## Runtime and setup

- Supported local Python: 3.11 or newer.
- Windows setup: run `setup.bat`. It recreates `.venv` in the repository and installs
  `requirements.txt`.
- Windows launch: run `run.bat`. It starts the service at `http://127.0.0.1:8756/` and opens the UI.
- Manual setup: `python -m venv .venv`, then `.venv\Scripts\python -m pip install -r requirements.txt`.
- Do not install project dependencies globally. Use `.venv` for local commands.

## Important files

- `bgr/registry.py`: model names, Hugging Face IDs, checkpoint paths, and model construction.
- `bgr/segmenter.py`: segmentation model adapters.
- `bgr/pipeline.py`: background-removal and post-processing pipeline.
- `bgr/cli.py`: `bgr remove` command-line interface.
- `serving/app.py`: FastAPI endpoints and model cache.
- `serving/static/index.html`: dependency-free browser UI.
- `tests/`: pytest test suite.
- `pyproject.toml`: package metadata and canonical project dependency list.
- `requirements.txt`: Python 3.11-compatible runtime dependencies used by `setup.bat`.
- `data/checkpoints/`: optional local model weights; large `.pth` files are ignored by Git.

## Development and verification

Run commands from the repository root:

```bat
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m uvicorn serving.app:app --host 127.0.0.1 --port 8756
.venv\Scripts\python.exe -m bgr.cli remove input.jpg -o output.png --model lucida
```

Use targeted tests while iterating and run the full test suite before handing off broad changes.
Tests marked `slow` may download or load real models and should only be run when that cost is
appropriate. A first real inference can download model weights from Hugging Face and may take time.

## Implementation rules

- Keep model-specific construction in `bgr/registry.py` or the segmenter abstraction; do not spread
  model-loading logic across the service and CLI.
- Preserve lazy model loading in the HTTP service. Model objects are expensive and must not be
  recreated for each request.
- Keep CLI and HTTP behavior aligned by routing both through `PipelineSegmenter`.
- Validate uploaded images and return useful client errors without exposing internal tracebacks.
- Close image/file resources where practical and avoid retaining full-resolution intermediate
  tensors longer than needed.
- Maintain CPU compatibility unless a change explicitly requires CUDA. Select accelerators at
  runtime rather than assuming a particular GPU.
- Do not commit downloaded weights, generated datasets, result images, `.env` files, or `.venv`.
- Keep dependencies in `requirements.txt` aligned with runtime dependencies in `pyproject.toml`.
- Keep source files and documentation UTF-8. Repair visible mojibake when editing affected text.
- Update README or relevant docs when user-facing commands, model names, ports, or behavior change.

## Git hygiene

Do not commit, push, create branches, or discard existing changes unless the user explicitly asks.
Before editing, inspect the working tree and avoid overwriting unrelated work.
