import albumentations as A
import numpy as np
from jaxtyping import Bool, Float

from court_training.dataset import Sample


class CourtAugment:
    def __init__(
        self,
        left_right_pairs: tuple[tuple[int, int], ...],
        keypoint_pairs: tuple[tuple[int, int], ...] = (),
        crop_cutout: bool = True,
        crop_p: float = 0.7,
        cutout_p: float = 0.5,
        flip_p: float = 0.5,
        affine_p: float = 1.0,
        affine_scale: tuple[float, float] = (0.88, 1.12),
        affine_translate: tuple[float, float] = (-0.08, 0.08),
        affine_rotate: tuple[float, float] = (-4.0, 4.0),
        affine_shear: tuple[float, float] = (-3.0, 3.0),
        color_p: float = 1.0,
        color_strength: float = 0.25,
        blur_p: float = 0.15,
        blur_sigma: tuple[float, float] = (0.2, 0.8),
        crop_scale: tuple[float, float] = (0.72, 1.0),
        crop_ratio: tuple[float, float] = (0.85, 1.15),
        cutout_holes: tuple[int, int] = (1, 4),
        cutout_size: tuple[float, float] = (0.04, 0.18),
    ) -> None:
        self.left_right_pairs = left_right_pairs
        self.keypoint_pairs = keypoint_pairs
        self.crop_cutout = crop_cutout
        self.crop_p = crop_p
        self.cutout_p = cutout_p
        self.flip_p = flip_p
        self.affine_p = affine_p
        self.affine_scale = affine_scale
        self.affine_translate = affine_translate
        self.affine_rotate = affine_rotate
        self.affine_shear = affine_shear
        self.color_p = color_p
        self.color_strength = color_strength
        self.blur_p = blur_p
        self.blur_sigma = blur_sigma
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.cutout_holes = cutout_holes
        self.cutout_size = cutout_size

    def __call__(self, sample: Sample) -> Sample:
        height, width = sample["image"].shape[:2]
        keypoints = normalized_to_pixels(sample["keypoints"], width, height)
        transformed = self.transform((width, height))(
            image=sample["image"],
            mask=sample["mask"],
            keypoints=keypoints,
        )
        keypoints = pixels_to_normalized(np.asarray(transformed["keypoints"], dtype=np.float32), width, height)
        visibility = sample["keypoint_visibility"] * points_inside_image(keypoints)
        sample = {
            "image": transformed["image"],
            "mask": transformed["mask"].astype(np.float32),
            "keypoints": keypoints,
            "keypoint_visibility": visibility.astype(np.float32),
        }
        if np.random.random() < self.flip_p:
            sample = horizontal_flip_sample(sample, self.left_right_pairs, self.keypoint_pairs)
        return sample

    def transform(self, image_size: tuple[int, int]) -> A.Compose:
        width, height = image_size
        transforms = [
            A.Affine(
                scale=self.affine_scale,
                translate_percent=self.affine_translate,
                rotate=self.affine_rotate,
                shear=self.affine_shear,
                p=self.affine_p,
            ),
            A.ColorJitter(
                brightness=self.color_strength,
                contrast=self.color_strength * 1.4,
                saturation=self.color_strength,
                hue=0.0,
                p=self.color_p,
            ),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=self.blur_sigma, p=self.blur_p),
        ]
        if self.crop_cutout:
            crop = A.RandomResizedCrop(
                size=(height, width),
                scale=self.crop_scale,
                ratio=self.crop_ratio,
                p=self.crop_p,
            )
            cutout = A.CoarseDropout(
                num_holes_range=self.cutout_holes,
                hole_height_range=self.cutout_size,
                hole_width_range=self.cutout_size,
                fill="random",
                p=self.cutout_p,
            )
            transforms.insert(0, crop)
            transforms.append(cutout)
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
        return A.Compose(transforms, keypoint_params=keypoint_params)


def swap_left_right_channels(
    mask: Float[np.ndarray, "H W N"],
    left_right_pairs: tuple[tuple[int, int], ...],
) -> Float[np.ndarray, "H W N"]:
    swapped = mask.copy()
    for left, right in left_right_pairs:
        swapped[:, :, left] = mask[:, :, right]
        swapped[:, :, right] = mask[:, :, left]
    return swapped


def horizontal_flip_sample(
    sample: Sample,
    left_right_pairs: tuple[tuple[int, int], ...],
    keypoint_pairs: tuple[tuple[int, int], ...],
) -> Sample:
    keypoints, visibility = flip_keypoints(
        sample["keypoints"],
        sample["keypoint_visibility"],
        keypoint_pairs,
    )
    flipped_mask = swap_left_right_channels(np.fliplr(sample["mask"]).copy(), left_right_pairs)
    return {
        "image": np.fliplr(sample["image"]).copy(),
        "mask": flipped_mask,
        "keypoints": keypoints,
        "keypoint_visibility": visibility,
    }


def flip_keypoints(
    keypoints: Float[np.ndarray, "keypoints 2"],
    visibility: Float[np.ndarray, "*keypoints"],
    keypoint_pairs: tuple[tuple[int, int], ...],
) -> tuple[Float[np.ndarray, "keypoints 2"], Float[np.ndarray, "*keypoints"]]:
    flipped_keypoints = keypoints.copy()
    flipped_visibility = visibility.copy()
    flipped_keypoints[:, 0] = 1 - flipped_keypoints[:, 0]
    for left, right in keypoint_pairs:
        flipped_keypoints[[left, right]] = flipped_keypoints[[right, left]]
        flipped_visibility[[left, right]] = flipped_visibility[[right, left]]
    return flipped_keypoints, flipped_visibility


def normalized_to_pixels(
    keypoints: Float[np.ndarray, "keypoints 2"],
    width: int,
    height: int,
) -> Float[np.ndarray, "keypoints 2"]:
    scale = np.array([width - 1, height - 1], dtype=np.float32)
    return keypoints * scale


def pixels_to_normalized(
    keypoints: Float[np.ndarray, "keypoints 2"],
    width: int,
    height: int,
) -> Float[np.ndarray, "keypoints 2"]:
    scale = np.array([width - 1, height - 1], dtype=np.float32)
    return keypoints / scale


def points_inside_image(keypoints: Float[np.ndarray, "keypoints 2"]) -> Bool[np.ndarray, "*keypoints"]:
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    return (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
