"""Numerical check for the browser-compatible DeformConv2d lowering."""
import torch
from torchvision.ops import DeformConv2d
from deform_conv_web import WebCompatibleDeformConv2d


def main() -> None:
    torch.manual_seed(7)
    layer = DeformConv2d(4, 6, 3, padding=1, bias=True).eval()
    replacement = WebCompatibleDeformConv2d(layer).eval()
    image = torch.randn(1, 4, 12, 10)
    offset = torch.randn(1, 18, 12, 10) * 0.2
    mask = torch.sigmoid(torch.randn(1, 9, 12, 10))
    with torch.inference_mode():
        expected = layer(image, offset, mask)
        actual = replacement(image, offset, mask)
    diff = (expected - actual).abs()
    print(f"max_abs_error={diff.max().item():.9g}")
    print(f"mean_abs_error={diff.mean().item():.9g}")
    if not torch.allclose(expected, actual, rtol=1e-4, atol=1e-5):
        raise SystemExit("FAILED")
    print("DEFORM CONV PARITY OK")


if __name__ == "__main__":
    main()
