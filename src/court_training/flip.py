import numpy as np
from jaxtyping import Float


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
        image: Float[np.ndarray, "... H W 3"] | None = None,
        masks: Float[np.ndarray, "... H W N"] | None = None,
        keypoints: Float[np.ndarray, "... K 2"] | None = None,
        visibility: Float[np.ndarray, "... K"] | None = None,
    ) -> dict[str, object]:
        if np.random.random() >= self.p:
            return {
                "image": image,
                "masks": masks,
                "keypoints": keypoints,
                "visibility": visibility,
                "applied": False,
            }

        flipped = flip(
            image=image,
            masks=masks,
            keypoints=keypoints,
            visibility=visibility,
            mask_names=self.mask_names,
            keypoint_names=self.keypoint_names,
        )
        flipped["applied"] = True
        return flipped


def flip(
    image: Float[np.ndarray, "... H W 3"] | None = None,
    masks: Float[np.ndarray, "... H W N"] | None = None,
    keypoints: Float[np.ndarray, "... K 2"] | None = None,
    visibility: Float[np.ndarray, "... K"] | None = None,
    mask_names: tuple[str, ...] = (),
    keypoint_names: tuple[str, ...] = (),
) -> dict[str, object]:
    output = {}
    if image is not None:
        output["image"] = flip_array(image)
    if masks is not None:
        output["masks"] = reorder(flip_array(masks), flip_indices(mask_names), axis=-1)
    if keypoints is not None:
        output["keypoints"] = reorder(flip_keypoints(keypoints, image), flip_indices(keypoint_names), axis=-2)
    if visibility is not None:
        output["visibility"] = reorder(visibility, flip_indices(keypoint_names), axis=-1)
    return output


def flip_array(array: np.ndarray) -> np.ndarray:
    return np.flip(array, axis=-2).copy()


def flip_keypoints(
    keypoints: Float[np.ndarray, "... K 2"],
    image: Float[np.ndarray, "... H W 3"] | None,
) -> Float[np.ndarray, "... K 2"]:
    flipped = keypoints.copy()
    if image is None:
        flipped[..., 0] = 1 - flipped[..., 0]
        return flipped

    width = image.shape[-2]
    flipped[..., 0] = width - 1 - flipped[..., 0]
    return flipped


def reorder(array: np.ndarray, indices: tuple[int, ...], axis: int) -> np.ndarray:
    return np.take(array, indices, axis=axis)


def flip_indices(labels: tuple[str, ...]) -> tuple[int, ...]:
    index_by_label = {label: index for index, label in enumerate(labels)}
    return tuple(index_by_label[flipped_label(label)] for label in labels)


def flipped_label(label: str) -> str:
    if label.startswith("left_"):
        return "right_" + label.removeprefix("left_")
    if label.startswith("right_"):
        return "left_" + label.removeprefix("right_")
    return label
