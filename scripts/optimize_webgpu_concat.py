"""Rewrite wide ONNX Concat nodes into WebGPU-safe trees."""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


def split_wide_concats(model: onnx.ModelProto, max_inputs: int = 7) -> tuple[int, int]:
    """Limit every Concat to max_inputs without changing input order or results."""
    rewritten = []
    split_nodes = 0
    added_nodes = 0
    for node in model.graph.node:
        if node.op_type != "Concat" or len(node.input) <= max_inputs:
            rewritten.append(node)
            continue
        axis = next(attribute.i for attribute in node.attribute if attribute.name == "axis")
        inputs = list(node.input)
        level = 0
        while len(inputs) > max_inputs:
            next_inputs = []
            for group_index, start in enumerate(range(0, len(inputs), max_inputs)):
                group = inputs[start:start + max_inputs]
                if len(group) == 1:
                    next_inputs.append(group[0])
                    continue
                output = f"{node.output[0]}__webgpu_concat_{level}_{group_index}"
                rewritten.append(helper.make_node(
                    "Concat", group, [output], axis=axis,
                    name=f"{node.name}__webgpu_concat_{level}_{group_index}",
                ))
                next_inputs.append(output)
                added_nodes += 1
            inputs = next_inputs
            level += 1
        final_node = helper.make_node(
            "Concat", inputs, list(node.output), axis=axis, name=node.name,
        )
        final_node.doc_string = node.doc_string
        rewritten.append(final_node)
        split_nodes += 1
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return split_nodes, added_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-inputs", type=int, default=7)
    args = parser.parse_args()
    model = onnx.load(str(args.input))
    split_nodes, added_nodes = split_wide_concats(model, args.max_inputs)
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    print(f"Split {split_nodes} wide Concat nodes; added {added_nodes} tree nodes")
    print(f"Saved: {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()

