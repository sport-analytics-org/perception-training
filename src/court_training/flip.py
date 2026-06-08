import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from court_training.dataset import NumpySample


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

    def __call__(
        self,
        image: Float[np.ndarray, "H W 3"],
        mask: Float[np.ndarray, "H W N"],
        keypoints: Float[np.ndarray, "K 2"],
        visibility: Float[np.ndarray, "*K"],
    ) -> NumpySample:
        if np.random.random() >= self.p:
            return {
                "image": image,
                "mask": mask,
                "keypoints": keypoints,
                "visibility": visibility,
            }

        flipped = flip_numpy(
            image=image,
            masks=mask,
            keypoints=keypoints,
            visibility=visibility,
            mask_names=self.mask_names,
            keypoint_names=self.keypoint_names,
        )
        return {
            "image": flipped["image"],
            "mask": flipped["masks"],
            "keypoints": flipped["keypoints"],
            "visibility": flipped["visibility"],
        }


def flip_numpy(
    image: Float[np.ndarray, "... H W 3"] | None = None,
    masks: Float[np.ndarray, "... H W N"] | None = None,
    keypoints: Float[np.ndarray, "... K 2"] | None = None,
    visibility: Float[np.ndarray, "... K"] | None = None,
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
    return output


def flip_torch(
    image: Float[Tensor, "... 3 H W"] | None = None,
    masks: Float[Tensor, "... N H W"] | None = None,
    keypoints: Float[Tensor, "... K 2"] | None = None,
    visibility: Float[Tensor, "... K"] | None = None,
    mask_names: tuple[str, ...] = (),
    keypoint_names: tuple[str, ...] = (),
) -> dict[str, Tensor]:
    output = {}
    if image is not None:
        output["image"] = torch.flip(image, dims=(-1,))
    if masks is not None:
        masks = torch.flip(masks, dims=(-1,))
        output["masks"] = take_torch(masks, flip_indices(mask_names), dim=-3)
    if keypoints is not None:
        keypoints = keypoints.clone()
        keypoints[..., 0] = 1 - keypoints[..., 0]
        output["keypoints"] = take_torch(keypoints, flip_indices(keypoint_names), dim=-2)
    if visibility is not None:
        output["visibility"] = take_torch(visibility, flip_indices(keypoint_names), dim=-1)
    return output


def take_torch(tensor: Tensor, indices: tuple[int, ...], dim: int) -> Tensor:
    index = torch.tensor(indices, device=tensor.device, dtype=torch.long)
    return torch.index_select(tensor, dim=dim % tensor.ndim, index=index)


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
