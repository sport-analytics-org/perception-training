import albumentations as A
import numpy as np
from jaxtyping import Bool, Float

from perception_training.dataset import NumpySample
from perception_training.flip import HorizontalFlip


class CourtAugment:
    def __init__(
        self,
        image_size: tuple[int, int],
        mask_names: tuple[str, ...] = (),
        keypoint_names: tuple[str, ...] = (),
        crop_cutout: bool = True,
        crop_p: float = 0.7,
        cutout_p: float = 0.5,
        flip_p: float = 0.5,
        affine_p: float = 0.8,
        affine_scale: tuple[float, float] = (0.88, 1.12),
        affine_translate: tuple[float, float] = (-0.08, 0.08),
        affine_rotate: tuple[float, float] = (-5.0, 5.0),
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

        self.geom = A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], min_visibility=0.1, clip=True),
        )

    def __call__(self, sample: NumpySample) -> NumpySample:
        height, width = sample["image"].shape[:2]
        keypoints = sample.get("keypoints", np.zeros((0, 2), dtype=np.float32))
        boxes_xywh = sample.get("boxes_xywh", np.zeros((0, 4), dtype=np.float32))
        boxes_cxcywh = boxes_xywh.copy()
        boxes_cxcywh[:, :2] += boxes_cxcywh[:, 2:] / 2
        targets = {
            "image": sample["image"],
            "keypoints": normalized_to_pixels(keypoints, width, height),
            "bboxes": boxes_cxcywh,
            "labels": sample.get("labels", np.zeros(0, dtype=np.int64)),
        }
        if "mask" in sample:
            targets["mask"] = sample["mask"]
        transformed = self.geom(**targets)

        output: NumpySample = {"image": transformed["image"]}
        if "mask" in sample:
            output["mask"] = transformed["mask"]
        if "keypoints" in sample:
            keypoints = pixels_to_normalized(transformed["keypoints"], width, height)
            output["keypoints"] = keypoints
            output["visibility"] = sample["visibility"] * points_inside_image(keypoints)
        if "boxes_xywh" in sample:
            boxes_cxcywh = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
            boxes_xywh = boxes_cxcywh.copy()
            boxes_xywh[:, :2] -= boxes_xywh[:, 2:] / 2
            output["boxes_xywh"] = boxes_xywh
            output["labels"] = np.array(transformed["labels"], dtype=np.int64)
        return self.hflip(output)


def normalized_to_pixels(
    keypoints: Float[np.ndarray, "K 2"],
    width: int,
    height: int,
) -> Float[np.ndarray, "K 2"]:
    scale = np.array([width - 1, height - 1], dtype=np.float32)
    return keypoints * scale


def pixels_to_normalized(
    keypoints: Float[np.ndarray, "K 2"],
    width: int,
    height: int,
) -> Float[np.ndarray, "K 2"]:
    scale = np.array([width - 1, height - 1], dtype=np.float32)
    return keypoints / scale


def points_inside_image(keypoints: Float[np.ndarray, "K 2"]) -> Bool[np.ndarray, "*K"]:
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    return (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
