import albumentations as A
import numpy as np

from court_training.detection.data import NumpySample


class DetectionAugment:
    def __init__(
        self,
        flip_p: float = 0.5,
        affine_p: float = 0.45,
        affine_scale: tuple[float, float] = (0.85, 1.2),
        affine_translate: tuple[float, float] = (-0.08, 0.08),
        affine_rotate: tuple[float, float] = (-6.0, 6.0),
        affine_shear: tuple[float, float] = (-3.0, 3.0),
        color_p: float = 0.5,
        color_strength: float = 0.18,
        blur_p: float = 0.12,
    ) -> None:
        self.pipeline = A.Compose(
            [
                A.HorizontalFlip(p=flip_p),
                A.Affine(
                    scale=affine_scale,
                    translate_percent=affine_translate,
                    rotate=affine_rotate,
                    shear=affine_shear,
                    p=affine_p,
                ),
                A.ColorJitter(
                    brightness=color_strength,
                    contrast=color_strength,
                    saturation=color_strength * 0.7,
                    hue=0.03,
                    p=color_p,
                ),
                A.GaussianBlur(blur_limit=(3, 3), p=blur_p),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], min_visibility=0.1, clip=True),
        )

    def __call__(self, sample: NumpySample) -> NumpySample:
        transformed = self.pipeline(image=sample["image"], bboxes=sample["boxes_cxcywh"], labels=sample["labels"])
        return {
            "image": transformed["image"],
            "boxes_cxcywh": np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4),
            "labels": np.array(transformed["labels"], dtype=np.int64),
        }
