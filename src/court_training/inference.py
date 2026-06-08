from typing import TypedDict

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from court_training.flip import flip_torch


class Prediction(TypedDict):
    masks: Float[Tensor, "B N H W"]
    keypoints: Float[Tensor, "B K 2"]
    visibility: Float[Tensor, "B K"]


def predict(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    scales: tuple[float, ...],
    mask_names: tuple[str, ...],
    keypoint_names: tuple[str, ...],
) -> Prediction:
    keypoints = torch.empty(images.shape[0], 0, 2, device=images.device, dtype=images.dtype)
    visibility = torch.empty(images.shape[0], 0, device=images.device, dtype=images.dtype)
    if keypoint_names:
        keypoints, visibility = _predict_keypoints(model, images, scales, keypoint_names)
    return {
        "masks": _predict_masks(model, images, scales, mask_names),
        "keypoints": keypoints,
        "visibility": visibility,
    }


def _predict_masks(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    scales: tuple[float, ...],
    mask_names: tuple[str, ...],
) -> Float[Tensor, "B N H W"]:
    output_size = images.shape[-2:]
    logits_by_scale = []
    for scale in scales:
        scaled_images = _resize_images(images, scale)
        logits = model(scaled_images)
        logits = (logits + _predict_flipped(model, scaled_images, mask_names)) / 2
        logits_by_scale.append(F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False))
    return torch.stack(logits_by_scale).mean(dim=0)


def _predict_keypoints(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    scales: tuple[float, ...],
    keypoint_names: tuple[str, ...],
) -> tuple[Float[Tensor, "B K 2"], Float[Tensor, "B K"]]:
    keypoints_by_scale = []
    visibility_by_scale = []
    for scale in scales:
        scaled_images = _resize_images(images, scale)
        keypoints, visibility = model.predict_keypoints(scaled_images)
        flipped_keypoints, flipped_visibility = _predict_flipped_keypoints(model, scaled_images, keypoint_names)
        keypoints_by_scale.extend((keypoints, flipped_keypoints))
        visibility_by_scale.extend((visibility, flipped_visibility))
    keypoints = torch.stack(keypoints_by_scale).mean(dim=0)
    visibility = torch.stack(visibility_by_scale).mean(dim=0)
    return keypoints, visibility


def _predict_flipped(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    mask_names: tuple[str, ...],
) -> Float[Tensor, "B N H W"]:
    flipped_images = flip_torch(image=images)["image"]
    assert isinstance(flipped_images, Tensor)
    flipped_logits = model(flipped_images)
    logits = flip_torch(masks=flipped_logits, mask_names=mask_names)["masks"]
    assert isinstance(logits, Tensor)
    return logits


def _predict_flipped_keypoints(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    keypoint_names: tuple[str, ...],
) -> tuple[Float[Tensor, "B K 2"], Float[Tensor, "B K"]]:
    flipped_images = flip_torch(image=images)["image"]
    assert isinstance(flipped_images, Tensor)
    keypoints, visibility = model.predict_keypoints(flipped_images)
    flipped = flip_torch(
        keypoints=keypoints,
        visibility=visibility,
        x_max=1,
        keypoint_names=keypoint_names,
    )
    flipped_keypoints = flipped["keypoints"]
    flipped_visibility = flipped["visibility"]
    assert isinstance(flipped_keypoints, Tensor)
    assert isinstance(flipped_visibility, Tensor)
    return flipped_keypoints, flipped_visibility


def _resize_images(images: Tensor, scale: float) -> Tensor:
    return F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)
