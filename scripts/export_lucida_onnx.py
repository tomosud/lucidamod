"""Export egeorcun/lucida to an unquantized, fixed-resolution ONNX model."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForImageSegmentation
from deform_conv_web import replace_deform_conv2d




class AlphaMaskModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(pixel_values)[-1])


def export_model(model_id: str, output: Path, input_size: int, opset: int) -> None:
    if opset < 19:
        raise ValueError("Lucida requires ONNX opset 19 or newer for DeformConv")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {model_id} ...", flush=True)
    base = AutoModelForImageSegmentation.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.float32
    ).eval()
    model = AlphaMaskModel(base).eval()
    replaced = replace_deform_conv2d(base)
    print(f"Replaced {replaced} DeformConv2d layers with web-compatible operations", flush=True)
    sample = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    print(f"Exporting FP32 ONNX ({input_size}x{input_size}, opset {opset}) ...", flush=True)
    started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            model, (sample,), str(output), input_names=["pixel_values"],
            output_names=["alpha"], opset_version=opset,
            do_constant_folding=True, dynamo=False,
        )
    elapsed = time.perf_counter() - started
    metadata = {
        "model_id": model_id, "format": "onnx", "precision": "float32",
        "quantized": False, "opset": opset,
        "input": {"name": "pixel_values", "shape": [1, 3, input_size, input_size]},
        "output": {"name": "alpha", "range": [0.0, 1.0]},
        "preprocessing": {
            "color": "RGB", "resize": [input_size, input_size], "layout": "NCHW",
            "scale": "uint8 / 255", "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "export_seconds": elapsed, "file_bytes": output.stat().st_size,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Exported: {output} ({output.stat().st_size / 1024**2:.1f} MiB)", flush=True)


def validate_model(output: Path, input_size: int) -> None:
    import onnx
    import onnxruntime as ort
    print("Checking ONNX structure ...", flush=True)
    onnx.checker.check_model(str(output))
    print("Running ONNX Runtime smoke test ...", flush=True)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    sample = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
    alpha = session.run(["alpha"], {"pixel_values": sample})[0]
    if alpha.shape != (1, 1, input_size, input_size):
        raise RuntimeError(f"Unexpected output shape: {alpha.shape}")
    if not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
        raise RuntimeError(f"Invalid alpha range: {alpha.min()}..{alpha.max()}")
    print(f"ONNX Runtime OK: shape={alpha.shape}, range={alpha.min():.6f}..{alpha.max():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="egeorcun/lucida")
    parser.add_argument("--output", type=Path, default=Path("models/lucida-fp32.onnx"))
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    export_model(args.model, args.output, args.input_size, args.opset)
    if not args.skip_validation:
        validate_model(args.output, args.input_size)


if __name__ == "__main__":
    main()
