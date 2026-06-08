import albumentations as A
import numpy as np
from jaxtyping import Bool, Float

from court_training.dataset import Sample
from court_training.flip import HorizontalFlip

KEYPOINT_LABEL_FIELDS = ["class_labels", "class_sides", "keypoint_visibility"]


class CourtAugment:
    def __init__(
        self,
        mask_names: tuple[str, ...],
        keypoint_names: tuple[str, ...],
        image_size: tuple[int, int],
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
        self.keypoint_sides, self.keypoint_labels = split_side_names(keypoint_names)
        self.keypoint_ids = tuple(zip(self.keypoint_labels, self.keypoint_sides, strict=True))
        self.hflip = HorizontalFlip(mask_names=mask_names, keypoint_names=keypoint_names, p=flip_p)

        height, width = image_size
        transforms = [
            A.Affine(
                scale=affine_scale,
                translate_percent=affine_translate,
                rotate=affine_rotate,
                shear=affine_shear,
                p=affine_p,
            ),
            A.ColorJitter(
                brightness=color_strength,
                contrast=color_strength * 1.4,
                saturation=color_strength,
                hue=0.0,
                p=color_p,
            ),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=blur_sigma, p=blur_p),
        ]
        if crop_cutout:
            crop = A.RandomResizedCrop(size=(height, width), scale=crop_scale, ratio=crop_ratio, p=crop_p)
            cutout = A.CoarseDropout(
                num_holes_range=cutout_holes,
                hole_height_range=cutout_size,
                hole_width_range=cutout_size,
                fill="random",
                p=cutout_p,
            )
            transforms.insert(0, crop)
            transforms.append(cutout)

        self.geom = A.Compose(transforms, keypoint_params=keypoint_params_xy())

    def __call__(self, sample: Sample) -> Sample:
        height, width = sample["image"].shape[:2]
        keypoints = normalized_to_pixels(sample["keypoints"], width, height)
        transformed = self.geom(
            image=sample["image"],
            mask=sample["mask"],
            keypoints=keypoints,
            class_labels=np.array(self.keypoint_labels),
            class_sides=np.array(self.keypoint_sides),
            keypoint_visibility=sample["keypoint_visibility"],
        )
        flipped = self.hflip(
            image=transformed["image"],
            masks=transformed["mask"],
            keypoints=transformed["keypoints"],
            visibility=transformed["keypoint_visibility"],
        )
        transformed["image"] = flipped["image"]
        transformed["mask"] = flipped["masks"]
        transformed["keypoints"] = flipped["keypoints"]
        transformed["keypoint_visibility"] = flipped["visibility"]

        keypoints = pixels_to_normalized(np.asarray(transformed["keypoints"], dtype=np.float32), width, height)
        visibility = np.asarray(transformed["keypoint_visibility"], dtype=np.float32)
        visibility = visibility * points_inside_image(keypoints)
        sides = tuple(str(side) for side in transformed["class_sides"])
        labels = tuple(str(label) for label in transformed["class_labels"])
        ids = tuple(zip(labels, sides, strict=True))
        keypoints, visibility = order_keypoints(keypoints, visibility, ids, self.keypoint_ids)
        return {
            "image": transformed["image"],
            "mask": transformed["mask"].astype(np.float32),
            "keypoints": keypoints,
            "keypoint_visibility": visibility.astype(np.float32),
        }


def keypoint_params_xy() -> A.KeypointParams:
    return A.KeypointParams(format="xy", label_fields=KEYPOINT_LABEL_FIELDS, remove_invisible=False)


def split_side_names(names: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sides = []
    labels = []
    for name in names:
        side, label = name.split("_", 1)
        sides.append(side)
        labels.append(label)
    return tuple(sides), tuple(labels)


def order_keypoints(
    keypoints: Float[np.ndarray, "keypoints 2"],
    visibility: Float[np.ndarray, "*keypoints"],
    ids: tuple[tuple[str, str], ...],
    output_ids: tuple[tuple[str, str], ...],
) -> tuple[Float[np.ndarray, "keypoints 2"], Float[np.ndarray, "*keypoints"]]:
    index_by_id = {keypoint_id: index for index, keypoint_id in enumerate(ids)}
    order = [index_by_id[keypoint_id] for keypoint_id in output_ids]
    return keypoints[order], visibility[order]


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
