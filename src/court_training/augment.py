import albumentations as A
import numpy as np

from court_training.dataset import MaskSample


class CourtAugment:
    def __init__(
        self,
        left_right_pairs: tuple[tuple[int, int], ...],
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

    def __call__(self, sample: MaskSample) -> MaskSample:
        height, width = sample["image"].shape[:2]
        transformed = self.transform((width, height))(image=sample["image"], mask=sample["mask"])
        return {"image": transformed["image"], "mask": transformed["mask"].astype(np.float32)}

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
            SideAwareHorizontalFlip(self.left_right_pairs, p=self.flip_p),
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
        return A.Compose(transforms)


class SideAwareHorizontalFlip(A.HorizontalFlip):
    def __init__(self, left_right_pairs: tuple[tuple[int, int], ...], p: float) -> None:
        super().__init__(p=p)
        self.left_right_pairs = left_right_pairs

    def apply_to_mask(self, mask: np.ndarray, *args: object, **params: object) -> np.ndarray:
        flipped = np.fliplr(mask).copy()
        return swap_left_right_channels(flipped, self.left_right_pairs)


def swap_left_right_channels(mask: np.ndarray, left_right_pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    swapped = mask.copy()
    for left, right in left_right_pairs:
        swapped[:, :, left] = mask[:, :, right]
        swapped[:, :, right] = mask[:, :, left]
    return swapped
