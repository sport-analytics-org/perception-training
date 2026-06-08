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


def predict_keypoints(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    scales: tuple[float, ...],
    keypoint_names: tuple[str, ...],
) -> tuple[Float[Tensor, "B K 2"], Float[Tensor, "B K"]]:
    keypoints_by_scale = []
    visibility_by_scale = []
    for scale in scales:
        scaled_images = resize_images(images, scale)
        keypoints, visibility = model.predict_keypoints(scaled_images)
        flipped_keypoints, flipped_visibility = predict_flipped_keypoints(model, scaled_images, keypoint_names)
        keypoints_by_scale.extend((keypoints, flipped_keypoints))
        visibility_by_scale.extend((visibility, flipped_visibility))
    keypoints = torch.stack(keypoints_by_scale).mean(dim=0)
    visibility = torch.stack(visibility_by_scale).mean(dim=0)
    return keypoints, visibility


def predict_flipped(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    mask_names: tuple[str, ...],
) -> Float[Tensor, "B N H W"]:
    images_numpy = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    flipped_images = flip(image=images_numpy)["image"]
    assert isinstance(flipped_images, np.ndarray)
    flipped_images = numpy_images_to_tensor(flipped_images, images)
    flipped_logits = model(flipped_images)
    logits_numpy = flipped_logits.detach().cpu().permute(0, 2, 3, 1).numpy()
    logits_numpy = flip(masks=logits_numpy, mask_names=mask_names)["masks"]
    assert isinstance(logits_numpy, np.ndarray)
    return numpy_masks_to_tensor(logits_numpy, flipped_logits)


def predict_flipped_keypoints(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    keypoint_names: tuple[str, ...],
) -> tuple[Float[Tensor, "B K 2"], Float[Tensor, "B K"]]:
    images_numpy = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    flipped_images = flip(image=images_numpy)["image"]
    assert isinstance(flipped_images, np.ndarray)
    flipped_images = numpy_images_to_tensor(flipped_images, images)
    keypoints, visibility = model.predict_keypoints(flipped_images)
    keypoints_numpy = keypoints.detach().cpu().numpy()
    visibility_numpy = visibility.detach().cpu().numpy()
    flipped = flip(
        keypoints=keypoints_numpy,
        visibility=visibility_numpy,
        x_max=1,
        keypoint_names=keypoint_names,
    )
    flipped_keypoints = flipped["keypoints"]
    flipped_visibility = flipped["visibility"]
    assert isinstance(flipped_keypoints, np.ndarray)
    assert isinstance(flipped_visibility, np.ndarray)
    keypoints = numpy_keypoints_to_tensor(flipped_keypoints, keypoints)
    visibility = numpy_visibility_to_tensor(flipped_visibility, visibility)
    return keypoints, visibility


def resize_images(images: Tensor, scale: float) -> Tensor:
    return F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)


def numpy_images_to_tensor(images: np.ndarray, like: Tensor) -> Tensor:
    tensor = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
    return tensor.to(device=like.device, dtype=like.dtype)


def numpy_masks_to_tensor(masks: np.ndarray, like: Tensor) -> Tensor:
    tensor = torch.from_numpy(masks.copy()).permute(0, 3, 1, 2)
    return tensor.to(device=like.device, dtype=like.dtype)


def numpy_keypoints_to_tensor(keypoints: np.ndarray, like: Tensor) -> Tensor:
    return torch.from_numpy(keypoints.copy()).to(device=like.device, dtype=like.dtype)


def numpy_visibility_to_tensor(visibility: np.ndarray, like: Tensor) -> Tensor:
    return torch.from_numpy(visibility.copy()).to(device=like.device, dtype=like.dtype)
