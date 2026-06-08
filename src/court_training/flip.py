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
            width=image.shape[-2] if image is not None else None,
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
    width: int | None = None,
    mask_names: tuple[str, ...] = (),
    keypoint_names: tuple[str, ...] = (),
) -> dict[str, object]:
    output = {}
    if image is not None:
        output["image"] = np.flip(image, axis=-2).copy()
    if masks is not None:
        masks = np.flip(masks, axis=-2)
        output["masks"] = np.take(masks, flip_indices(mask_names), axis=-1).copy()
    if keypoints is not None:
        assert width is not None
        keypoints = keypoints.copy()
        keypoints[..., 0] = width - 1 - keypoints[..., 0]
        output["keypoints"] = np.take(keypoints, flip_indices(keypoint_names), axis=-2)
    if visibility is not None:
        output["visibility"] = np.take(visibility, flip_indices(keypoint_names), axis=-1)
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
