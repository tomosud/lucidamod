"""Pure-PyTorch DeformConv2d lowering for ONNX Runtime Web compatibility."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d


class WebCompatibleDeformConv2d(nn.Module):
    """Equivalent DeformConv2d expressed with GridSample and Conv."""

    def __init__(self, source: DeformConv2d | nn.Conv2d):
        super().__init__()
        self.weight = source.weight
        self.bias = source.bias
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.kernel_h = int(source.weight.shape[-2])
        self.kernel_w = int(source.weight.shape[-1])

    def forward(
        self, input: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        n, channels, in_h, in_w = input.shape
        out_h, out_w = offset.shape[-2:]
        kernel_h, kernel_w = self.kernel_h, self.kernel_w
        kernel_points = kernel_h * kernel_w
        offset_groups = offset.shape[1] // (2 * kernel_points)
        channels_per_offset_group = channels // offset_groups

        base_y = torch.arange(out_h, device=input.device, dtype=input.dtype) * self.stride[0]
        base_x = torch.arange(out_w, device=input.device, dtype=input.dtype) * self.stride[1]
        base_y, base_x = torch.meshgrid(base_y, base_x, indexing="ij")
        sampled_groups: list[torch.Tensor] = []

        for offset_group in range(offset_groups):
            channel_start = offset_group * channels_per_offset_group
            channel_end = channel_start + channels_per_offset_group
            input_group = input[:, channel_start:channel_end]
            sampled_points: list[torch.Tensor] = []
            offset_start = offset_group * 2 * kernel_points
            mask_start = offset_group * kernel_points
            for point in range(kernel_points):
                kernel_y, kernel_x = divmod(point, kernel_w)
                offset_y = offset[:, offset_start + 2 * point]
                offset_x = offset[:, offset_start + 2 * point + 1]
                sample_y = (
                    base_y - self.padding[0] + kernel_y * self.dilation[0] + offset_y
                )
                sample_x = (
                    base_x - self.padding[1] + kernel_x * self.dilation[1] + offset_x
                )
                grid_y = (sample_y + 0.5) * (2.0 / in_h) - 1.0
                grid_x = (sample_x + 0.5) * (2.0 / in_w) - 1.0
                grid = torch.stack((grid_x, grid_y), dim=-1)
                sampled = F.grid_sample(
                    input_group,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                if mask is not None:
                    sampled = sampled * mask[:, mask_start + point].unsqueeze(1)
                sampled_points.append(sampled)
            sampled_groups.append(torch.stack(sampled_points, dim=2))

        sampled_all = torch.cat(sampled_groups, dim=1)
        sampled_flat = sampled_all.flatten(1, 2)
        weight_1x1 = self.weight.reshape(
            self.weight.shape[0], self.weight.shape[1] * kernel_points, 1, 1
        )
        return F.conv2d(sampled_flat, weight_1x1, self.bias, groups=self.groups)


class WebCompatibleDeformableConv2d(nn.Module):
    """Replacement for BiRefNet's custom DeformableConv2d wrapper."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.offset_conv = source.offset_conv
        self.modulator_conv = source.modulator_conv
        self.regular_conv = WebCompatibleDeformConv2d(source.regular_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
        return self.regular_conv(x, offset, modulator)


def replace_deform_conv2d(module: nn.Module) -> int:
    """Replace every torchvision DeformConv2d child in-place and return the count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, DeformConv2d):
            setattr(module, name, WebCompatibleDeformConv2d(child))
            count += 1
        elif child.__class__.__name__ == "DeformableConv2d" and all(
            hasattr(child, attr) for attr in ("offset_conv", "modulator_conv", "regular_conv")
        ):
            setattr(module, name, WebCompatibleDeformableConv2d(child))
            count += 1
        else:
            count += replace_deform_conv2d(child)
    return count
