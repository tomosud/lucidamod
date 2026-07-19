"""Convert a Lucida FP32 ONNX model to FP16 while keeping float32 browser I/O."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16
from optimize_webgpu_concat import split_wide_concats




def deduplicate_casts(model: onnx.ModelProto) -> int:
    """Collapse identical Cast nodes emitted for shared blocked-op inputs."""
    seen = {}
    kept = []
    removed = 0
    for node in model.graph.node:
        collisions = [seen[name] for name in node.output if name and name in seen]
        if collisions:
            previous = collisions[0]
            if node.op_type != "Cast" or previous.op_type != "Cast" or list(node.input) != list(previous.input):
                raise RuntimeError(f"Unexpected duplicate tensor output: {list(node.output)}")
            removed += 1
            continue
        kept.append(node)
        for name in node.output:
            if name:
                seen[name] = node
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return removed

def topological_sort(model: onnx.ModelProto) -> None:
    """Restore dependency order after the FP16 converter inserts Cast nodes."""
    nodes = list(model.graph.node)
    producer = {name: index for index, node in enumerate(nodes) for name in node.output if name}
    dependencies = [set(producer[name] for name in node.input if name in producer) for node in nodes]
    consumers = [[] for _ in nodes]
    for index, deps in enumerate(dependencies):
        for dep in deps:
            consumers[dep].append(index)
    ready = [index for index, deps in enumerate(dependencies) if not deps]
    ordered = []
    cursor = 0
    while cursor < len(ready):
        index = ready[cursor]
        cursor += 1
        ordered.append(nodes[index])
        for consumer in consumers[index]:
            dependencies[consumer].discard(index)
            if not dependencies[consumer]:
                ready.append(consumer)
    if len(ordered) != len(nodes):
        raise RuntimeError("FP16 graph contains a dependency cycle")
    del model.graph.node[:]
    model.graph.node.extend(ordered)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading FP32 model: {args.input}", flush=True)
    started = time.perf_counter()
    model = onnx.load(str(args.input))
    print("Converting weights and compute tensors to FP16 ...", flush=True)
    model = convert_float_to_float16(
        model, keep_io_types=False, disable_shape_infer=False,
        op_block_list=[*DEFAULT_OP_BLOCK_LIST, "GridSample"]
    )
    removed = deduplicate_casts(model)
    print(f"Deduplicated {removed} shared Cast nodes", flush=True)
    split_nodes, added_nodes = split_wide_concats(model, max_inputs=7)
    print(f"Split {split_nodes} wide Concat nodes; added {added_nodes} tree nodes", flush=True)
    topological_sort(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    elapsed = time.perf_counter() - started
    metadata = {}
    source_metadata = args.input.with_suffix(".json")
    if source_metadata.exists():
        metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    metadata.update({
        "precision": "mixed_float16", "io_precision": "float16",
        "execution_provider": "webgpu", "fp32_ops": ["GridSample"], "source_model": args.input.name,
        "conversion_seconds": elapsed, "file_bytes": args.output.stat().st_size,
    })
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    print(f"Elapsed: {elapsed:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()











