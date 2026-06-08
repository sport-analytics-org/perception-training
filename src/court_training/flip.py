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
        output["masks"] = swap_left_right(flip_array(masks), mask_names, channel_axis(masks))
    if keypoints is not None:
        output["keypoints"] = swap_left_right(flip_keypoints(keypoints, image), keypoint_names, -2)
    if visibility is not None:
        output["visibility"] = swap_left_right(visibility, keypoint_names, -1)
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


def swap_left_right(array: np.ndarray, names: tuple[str, ...], axis: int) -> np.ndarray:
    swapped = array.copy()
    for left, right in left_right_pairs(names):
        left_index = channel_index(array.ndim, axis, left)
        right_index = channel_index(array.ndim, axis, right)
        swapped[left_index] = array[right_index]
        swapped[right_index] = array[left_index]
    return swapped


def channel_axis(array: np.ndarray) -> int:
    return -1


def channel_index(ndim: int, axis: int, index: int) -> tuple[slice | int, ...]:
    selectors: list[slice | int] = [slice(None)] * ndim
    selectors[axis] = index
    return tuple(selectors)


def left_right_pairs(names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    index_by_name = {name: index for index, name in enumerate(names)}
    pairs = []
    for name, index in index_by_name.items():
        if name.startswith("left_"):
            right_name = "right_" + name.removeprefix("left_")
            pairs.append((index, index_by_name[right_name]))
    return tuple(pairs)
