"""Fail if the browser ONNX still contains unsupported DeformConv nodes."""
from collections import Counter
from pathlib import Path
import onnx

path = Path("models/lucida-fp32.onnx")
model = onnx.load(str(path), load_external_data=False)
operators = Counter(node.op_type for node in model.graph.node)
print(f"nodes={len(model.graph.node)}")
print(f"DeformConv={operators['DeformConv']}")
print(f"GridSample={operators['GridSample']}")
if operators["DeformConv"]:
    raise SystemExit("FAILED: unsupported DeformConv remains")
print("WEB OPERATOR CHECK OK")
