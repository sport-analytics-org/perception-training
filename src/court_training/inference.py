from typing import NotRequired, TypedDict

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from court_training.flip import flip_torch


class Prediction(TypedDict):
    masks: Float[Tensor, "B N H W"]
    keypoints: NotRequired[Float[Tensor, "B K 2"]]
    visibility: NotRequired[Float[Tensor, "B K"]]


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
        scaled_images = resize(images, scale)
        prediction = model(scaled_images)
        assert len(mask_names) == prediction["masks"].shape[1]
        assert len(keypoint_names) == prediction["keypoints"].shape[1]

        flipped_prediction = model(flip_torch(image=scaled_images)["image"])
        flipped = flip_torch(
            masks=flipped_prediction["masks"],
            keypoints=flipped_prediction["keypoints"],
            visibility=flipped_prediction["visibility"],
            mask_names=mask_names,
            keypoint_names=keypoint_names,
        )

        masks_by_scale.append(resize((prediction["masks"] + flipped["masks"]) / 2, output_size))
        keypoints_by_scale.append((prediction["keypoints"] + flipped["keypoints"]) / 2)
        visibility_by_scale.append((prediction["visibility"] + flipped["visibility"]) / 2)

    return {
        "masks": torch.stack(masks_by_scale).mean(dim=0),
        "keypoints": torch.stack(keypoints_by_scale).mean(dim=0),
        "visibility": torch.stack(visibility_by_scale).mean(dim=0),
    }


def resize(tensor: Float[Tensor, "B C H W"], size: tuple[int, int] | float) -> Float[Tensor, "B C H W"]:
    if isinstance(size, float):
        return F.interpolate(tensor, scale_factor=size, mode="bilinear", align_corners=False)
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
