import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F


def warp(
    image: Float[Tensor, "C H W"],
    homography: Float[Tensor, "3 3"],
    output_shape: tuple[int, int],
) -> Float[Tensor, "C out_H out_W"]:
    # Build one normalized coordinate for every output pixel; this is a dense raster warp.
    grid = _normalized_grid(output_shape, image.device)

    # Inverse warp: for each output pixel, find where to sample in the source image.
    inverse = torch.linalg.inv(homography)
    source = grid.reshape(-1, 3) @ inverse.T
    denominator = source[:, 2:].clamp_min(1e-6)
    source = source[:, :2] / denominator

    # grid_sample expects coordinates in [-1, 1] and one grid per input batch item.
    output_height, output_width = output_shape
    sample_grid = source.reshape(1, output_height, output_width, 2) * 2 - 1
    sample_grid = sample_grid.expand(image.shape[0], -1, -1, -1)
    return F.grid_sample(image[:, None], sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]


def _normalized_grid(output_shape: tuple[int, int], device: torch.device) -> Float[Tensor, "H W 3"]:
    height, width = output_shape
    x_axis = torch.linspace(0, 1, width, dtype=torch.float32, device=device)
    y_axis = torch.linspace(0, 1, height, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    return torch.stack([x, y, torch.ones_like(x)], dim=-1)
