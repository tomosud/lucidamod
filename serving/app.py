"""Lokal bg-remove servisi: uv run uvicorn serving.app:app --port 8756"""
import io
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from bgr.registry import MODEL_SPECS, get_segmenter

app = FastAPI(title="my-bg-remover")
_STATIC_DIR = Path(__file__).parent / "static"
_DEFAULT_MODEL = "lucida"
_SEGMENTERS: dict[str, object] = {}
_SEGMENTERS_LOCK = threading.Lock()
_STATUS_LOCK = threading.Lock()
_STATUS = {"phase": "idle", "message": "Ready", "model": None}


def _set_status(phase: str, message: str, model: str | None = None):
    with _STATUS_LOCK:
        _STATUS.update(phase=phase, message=message, model=model)


@app.get("/status")
def status():
    with _STATUS_LOCK:
        return {**_STATUS, "loaded_models": sorted(_SEGMENTERS)}


def _load_segmenter(name: str):
    if name not in _SEGMENTERS:
        with _SEGMENTERS_LOCK:
            if name not in _SEGMENTERS:
                _SEGMENTERS[name] = get_segmenter(name)
    return _SEGMENTERS[name]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "models": sorted(MODEL_SPECS)}


@app.get("/models")
def models():
    return {"models": sorted(MODEL_SPECS), "default": _DEFAULT_MODEL}


@app.post("/remove")
def remove(
    file: UploadFile,
    model: str = "rmbg-2.0",
    refine: bool = False,
    decontaminate: bool = True,
):
    from bgr.pipeline import PipelineSegmenter

    model_key = model + ("+refine" if refine else "")
    try:
        if model_key in _SEGMENTERS:
            _set_status("processing", "Running background removal", model)
        else:
            _set_status("loading", "Loading model into memory", model)
        seg = _load_segmenter(model_key)
    except KeyError:
        _set_status("error", "Unknown model", model)
        raise HTTPException(400, f"bilinmeyen model: {model}")
    except Exception as exc:
        _set_status("error", f"Model loading failed: {exc}", model)
        raise
    try:
        _set_status("reading", "Reading input image", model)
        img = Image.open(io.BytesIO(file.file.read()))
        img.load()
    except Exception:
        _set_status("error", "Invalid image file", model)
        raise HTTPException(400, "geçersiz görsel dosyası")
    try:
        _set_status("processing", "Running background removal", model)
        pipe = seg if isinstance(seg, PipelineSegmenter) else PipelineSegmenter(seg)
        out = pipe.process(img, decontaminate=decontaminate)
        _set_status("encoding", "Creating transparent PNG", model)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        _set_status("complete", "Completed", model)
        return Response(buf.getvalue(), media_type="image/png")
    except Exception as exc:
        _set_status("error", f"Processing failed: {exc}", model)
        raise
