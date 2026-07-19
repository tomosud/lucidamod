"""Compare Lucida PyTorch and ONNX outputs on the same deterministic input."""
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForImageSegmentation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="egeorcun/lucida")
    parser.add_argument("--onnx", type=Path, default=Path("models/lucida-fp32.onnx"))
    parser.add_argument("--input-size", type=int, default=1024)
    args = parser.parse_args()

    sample = torch.zeros(1, 3, args.input_size, args.input_size, dtype=torch.float32)
    model = AutoModelForImageSegmentation.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    ).eval()
    started = time.perf_counter()
    with torch.inference_mode():
        expected = model(sample)[-1].sigmoid().numpy()
    torch_seconds = time.perf_counter() - started

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    actual = session.run(["alpha"], {"pixel_values": sample.numpy()})[0]
    onnx_seconds = time.perf_counter() - started

    diff = np.abs(expected - actual)
    print(f"shape={actual.shape}")
    print(f"max_abs_error={diff.max():.9g}")
    print(f"mean_abs_error={diff.mean():.9g}")
    print(f"pytorch_seconds={torch_seconds:.3f}")
    print(f"onnxruntime_seconds={onnx_seconds:.3f}")
    if not np.allclose(expected, actual, rtol=2e-3, atol=2e-4):
        raise SystemExit("FAILED: ONNX output differs from PyTorch")
    print("PARITY OK")


if __name__ == "__main__":
    main()
