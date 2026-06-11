import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F

MIN_DENOMINATOR = 1e-6


def warp(
    image: Float[Tensor, "... C H W"],
    homography: Float[Tensor, "... 3 3"],
    output_shape: tuple[int, int],
) -> Float[Tensor, "... C out_H out_W"]:
    leading_shape = image.shape[:-3]
    channels = image.shape[-3]

    # Build one normalized coordinate for every output pixel; this is a dense raster warp.
    grid = _normalized_grid(output_shape, image.device)

    # Inverse warp: for each output pixel, find where to sample in the source image.
    inverse = torch.linalg.inv(homography)
    source = torch.einsum("hwc,...dc->...hwd", grid, inverse)
    denominator = source[..., 2:]
    denominator_sign = torch.where(denominator < 0, -1.0, 1.0)
    safe_denominator = denominator_sign * MIN_DENOMINATOR
    denominator = torch.where(denominator.abs() < MIN_DENOMINATOR, safe_denominator, denominator)
    source = source[..., :2] / denominator

    # grid_sample expects coordinates in [-1, 1] and one grid per input batch item.
    sample_grid = source * 2 - 1
    sample_grid = sample_grid.clamp(-2, 2)
    image = image.reshape(-1, channels, *image.shape[-2:])
    sample_grid = sample_grid.reshape(-1, *sample_grid.shape[-3:])
    warped = F.grid_sample(image, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    warped = warped.reshape(*leading_shape, channels, *output_shape)
    return warped


def _normalized_grid(output_shape: tuple[int, int], device: torch.device) -> Float[Tensor, "H W 3"]:
    height, width = output_shape
    x_axis = torch.linspace(0, 1, width, dtype=torch.float32, device=device)
    y_axis = torch.linspace(0, 1, height, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    return torch.stack([x, y, torch.ones_like(x)], dim=-1)
