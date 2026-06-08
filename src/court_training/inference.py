import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from court_training.flip import flip


def predict_masks(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    scales: tuple[float, ...],
    mask_names: tuple[str, ...],
) -> Float[Tensor, "B N H W"]:
    output_size = images.shape[-2:]
    logits_by_scale = []
    for scale in scales:
        scaled_images = resize_images(images, scale)
        logits = model(scaled_images)
        logits = (logits + predict_flipped(model, scaled_images, mask_names)) / 2
        logits_by_scale.append(F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False))
    return torch.stack(logits_by_scale).mean(dim=0)


def predict_flipped(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    mask_names: tuple[str, ...],
) -> Float[Tensor, "B N H W"]:
    images_numpy = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    flipped_images = flip(image=images_numpy)["image"]
    flipped_images = numpy_images_to_tensor(flipped_images, images)
    flipped_logits = model(flipped_images)
    logits_numpy = flipped_logits.detach().cpu().permute(0, 2, 3, 1).numpy()
    logits_numpy = flip(masks=logits_numpy, mask_names=mask_names)["masks"]
    return numpy_masks_to_tensor(logits_numpy, flipped_logits)


def resize_images(images: Tensor, scale: float) -> Tensor:
    return F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)


def numpy_images_to_tensor(images: np.ndarray, like: Tensor) -> Tensor:
    tensor = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
    return tensor.to(device=like.device, dtype=like.dtype)


def numpy_masks_to_tensor(masks: np.ndarray, like: Tensor) -> Tensor:
    tensor = torch.from_numpy(masks.copy()).permute(0, 3, 1, 2)
    return tensor.to(device=like.device, dtype=like.dtype)
