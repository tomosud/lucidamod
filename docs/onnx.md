# Lucida ONNX export

This is the first, unoptimized browser-model baseline. It preserves the original Lucida weights in
FP32 and uses the model's trained 1024 x 1024 input resolution.

```bat
.venv\Scripts\python.exe -m pip install -r requirements-onnx.txt
.venv\Scripts\python.exe scripts\export_lucida_onnx.py
```

The generated `models/lucida-fp32.onnx` has a fixed `[1, 3, 1024, 1024]` RGB input and a
`[1, 1, 1024, 1024]` alpha output. Its adjacent JSON file records preprocessing metadata.

Preprocessing is RGB resize, float conversion, NCHW layout, then ImageNet mean/std normalization.
The ONNX output already includes sigmoid and is an alpha mask in `[0, 1]`.

The baseline is intentionally large. After output parity is confirmed, evaluate FP16, INT8,
smaller input resolutions, and ONNX Runtime Web/WebGPU compatibility separately.
