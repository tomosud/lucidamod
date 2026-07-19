# Lucida ONNX / WebGPU experiment

Export the web-compatible FP32 baseline at the trained 1024 x 1024 resolution, then convert its
weights and internal compute tensors to FP16. Browser I/O is also FP16; the JavaScript test converts pixels to and from IEEE FP16 bit arrays.

```bat
.venv\Scripts\python.exe -m pip install -r requirements-onnx.txt
.venv\Scripts\python.exe scripts\export_lucida_onnx.py
.venv\Scripts\python.exe scripts\convert_onnx_fp16.py models\lucida-fp32.onnx models\lucida-web-1024-fp16.onnx
```

Start the separate browser experiment with `run_onnx_web.bat`, then use the page it opens at
`http://127.0.0.1:8760/web_onnx/`. This experiment requests only ONNX Runtime Web's `webgpu`
execution provider; it deliberately does not fall back to WASM/CPU. Use a current Chrome or Edge.
The diagnostic log reports whether WebGPU is exposed, the selected adapter information available
from the browser, its buffer limits, session creation, and inference timings.

The generated `models/lucida-web-1024-fp16.onnx` has a fixed `[1, 3, 1024, 1024]` RGB Float16
input and `[1, 1, 1024, 1024]` alpha output. Convolution weights and most compute tensors are FP16. GridSample is intentionally kept in FP32 because ONNX Runtime Web 1.22 generates invalid mixed f32/f16 WGSL for its FP16 GridSample kernel.
Its adjacent JSON file records preprocessing and conversion metadata.

Preprocessing is RGB resize, float conversion, NCHW layout, then ImageNet mean/std normalization.
The ONNX output already includes sigmoid and is an alpha mask in `[0, 1]`.

Wide Concat nodes from the DeformConv lowering are rewritten as trees with at most seven inputs. This keeps input plus output storage-buffer bindings within WebGPU's default per-stage limit of eight.
`r`nThe browser test applies a two-pass blur-fusion foreground colour estimate after inference (90 px, then 6 px), based on PhotoRoom's Approximate Fast Foreground Colour Estimation method. It reduces original-background colour bleeding around semi-transparent edges without changing the alpha matte.
`r`nThe FP16 file is still large because it preserves the original architecture and weights. INT8 is
mainly worth evaluating for a WASM/CPU target; for this GPU-only experiment, WebGPU + FP16 is the
first useful 1024 baseline. A genuinely small distributable build will require architecture-level
work such as distillation or a smaller backbone, not only numeric conversion.
