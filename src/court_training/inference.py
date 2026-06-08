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
    output_size = images.shape[-2:]
    masks_by_scale = []
    keypoints_by_scale = []
    visibility_by_scale = []
    for scale in scales:
        scaled_images = _resize_images(images, scale)
        prediction = model(scaled_images)
        flipped = _predict_flipped(model, scaled_images, mask_names, keypoint_names)
        masks_by_scale.append(_resize_masks((prediction["masks"] + flipped["masks"]) / 2, output_size))
        if keypoint_names:
            keypoints_by_scale.extend((prediction["keypoints"], flipped["keypoints"]))
            visibility_by_scale.extend((prediction["visibility"], flipped["visibility"]))

    keypoints = torch.empty(images.shape[0], 0, 2, device=images.device, dtype=images.dtype)
    visibility = torch.empty(images.shape[0], 0, device=images.device, dtype=images.dtype)
    if keypoint_names:
        keypoints = torch.stack(keypoints_by_scale).mean(dim=0)
        visibility = torch.stack(visibility_by_scale).mean(dim=0)

    return {
        "masks": torch.stack(masks_by_scale).mean(dim=0),
        "keypoints": keypoints,
        "visibility": visibility,
    }


def _predict_flipped(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    mask_names: tuple[str, ...],
    keypoint_names: tuple[str, ...],
) -> Prediction:
    flipped_images = flip_torch(image=images)["image"]
    assert isinstance(flipped_images, Tensor)
    prediction = model(flipped_images)
    flipped = flip_torch(
        masks=prediction["masks"],
        keypoints=prediction["keypoints"] if keypoint_names else None,
        visibility=prediction["visibility"] if keypoint_names else None,
        mask_names=mask_names,
        keypoint_names=keypoint_names,
    )
    masks = flipped["masks"]
    assert isinstance(masks, Tensor)
    if not keypoint_names:
        return {
            "masks": masks,
            "keypoints": torch.empty(images.shape[0], 0, 2, device=images.device, dtype=images.dtype),
            "visibility": torch.empty(images.shape[0], 0, device=images.device, dtype=images.dtype),
        }

    flipped_keypoints = flipped["keypoints"]
    flipped_visibility = flipped["visibility"]
    assert isinstance(flipped_keypoints, Tensor)
    assert isinstance(flipped_visibility, Tensor)
    return {"masks": masks, "keypoints": flipped_keypoints, "visibility": flipped_visibility}


def _resize_masks(masks: Float[Tensor, "B N H W"], size: tuple[int, int]) -> Float[Tensor, "B N H W"]:
    return F.interpolate(masks, size=size, mode="bilinear", align_corners=False)


def _resize_images(images: Tensor, scale: float) -> Tensor:
    return F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)
