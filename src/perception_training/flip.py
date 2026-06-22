import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from perception_training.dataset import NumpySample


class HorizontalFlip:
    def __init__(
        self,
        mask_names: tuple[str, ...] = (),
        keypoint_names: tuple[str, ...] = (),
        p: float = 0.5,
    ) -> None:
        self.mask_names = mask_names
        self.keypoint_names = keypoint_names
        self.p = p

    def __call__(self, sample: NumpySample) -> NumpySample:
        if np.random.random() >= self.p:
            return sample

        flipped = flip_numpy(
            image=sample["image"],
            masks=sample.get("mask"),
            keypoints=sample.get("keypoints"),
            visibility=sample.get("visibility"),
            boxes_xywh=sample.get("boxes_xywh"),
            mask_names=self.mask_names,
            keypoint_names=self.keypoint_names,
        )
        output: NumpySample = {"image": flipped["image"]}
        if "mask" in sample:
            output["mask"] = flipped["masks"]
        if "keypoints" in sample:
            output["keypoints"] = flipped["keypoints"]
            output["visibility"] = flipped["visibility"]
        if "boxes_xywh" in sample:
            output["boxes_xywh"] = flipped["boxes_xywh"]
            output["labels"] = sample["labels"]
            if "attributes" in sample:
                output["attributes"] = sample["attributes"]
        return output


def flip_numpy(
    image: Float[np.ndarray, "... H W 3"] | None = None,
    masks: Float[np.ndarray, "... H W N"] | None = None,
    keypoints: Float[np.ndarray, "... K 2"] | None = None,
    visibility: Float[np.ndarray, "... K"] | None = None,
    boxes_xywh: Float[np.ndarray, "... D 4"] | None = None,
    mask_names: tuple[str, ...] = (),
    keypoint_names: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    output = {}
    if image is not None:
        output["image"] = np.flip(image, axis=-2).copy()
    if masks is not None:
        masks = np.flip(masks, axis=-2)
        output["masks"] = np.take(masks, flip_indices(mask_names), axis=-1).copy()
    if keypoints is not None:
        keypoints = keypoints.copy()
        keypoints[..., 0] = 1 - keypoints[..., 0]
        output["keypoints"] = np.take(keypoints, flip_indices(keypoint_names), axis=-2)
    if visibility is not None:
        output["visibility"] = np.take(visibility, flip_indices(keypoint_names), axis=-1)
    if boxes_xywh is not None:
        boxes_xywh = boxes_xywh.copy()
        boxes_xywh[..., 0] = 1 - boxes_xywh[..., 0] - boxes_xywh[..., 2]
        output["boxes_xywh"] = boxes_xywh
    return output


def flip_torch(
    image: Float[Tensor, "... 3 H W"] | None = None,
    masks: Float[Tensor, "... N H W"] | None = None,
    keypoints: Float[Tensor, "... K 2"] | None = None,
    visibility: Float[Tensor, "... K"] | None = None,
    boxes_xywh: Float[Tensor, "... D 4"] | None = None,
    mask_names: tuple[str, ...] = (),
    keypoint_names: tuple[str, ...] = (),
) -> dict[str, Tensor]:
    output = {}
    if image is not None:
        output["image"] = torch.flip(image, dims=(-1,))
    if masks is not None:
        masks = torch.flip(masks, dims=(-1,))
        output["masks"] = masks[..., flip_indices(mask_names), :, :]
    if keypoints is not None:
        keypoints = keypoints.clone()
        keypoints[..., 0] = 1 - keypoints[..., 0]
        output["keypoints"] = keypoints[..., flip_indices(keypoint_names), :]
    if visibility is not None:
        output["visibility"] = visibility[..., flip_indices(keypoint_names)]
    if boxes_xywh is not None:
        x, y, w, h = boxes_xywh.unbind(-1)
        output["boxes_xywh"] = torch.stack([1 - x - w, y, w, h], dim=-1)
    return output


def flip_indices(labels: tuple[str, ...]) -> tuple[int, ...]:
    index_by_label = {label: index for index, label in enumerate(labels)}
    indices = []
    for label in labels:
        flipped_label = label
        if label.startswith("left_"):
            flipped_label = "right_" + label.removeprefix("left_")
        if label.startswith("right_"):
            flipped_label = "left_" + label.removeprefix("right_")
        indices.append(index_by_label[flipped_label])
    return tuple(indices)
