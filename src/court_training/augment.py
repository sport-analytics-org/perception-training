from typing import TypedDict

import albumentations as A
import numpy as np
from PIL import Image


class ReplayTransform(TypedDict):
    __class_fullname__: str
    applied: bool


class Replay(TypedDict):
    transforms: list[ReplayTransform]


class CourtAugment:
    def __init__(self, left_right_pairs: tuple[tuple[int, int], ...], crop_cutout: bool = True) -> None:
        self.left_right_pairs = left_right_pairs
        self.crop_cutout = crop_cutout

    def __call__(self, image: Image.Image, bitfield: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        transformed = self.transform(image.size)(image=np.asarray(image), mask=bitfield)
        mask = transformed["mask"].astype(np.uint8)
        if horizontal_flip_was_applied(transformed["replay"]):
            mask = swap_left_right_bits(mask, self.left_right_pairs)
        return Image.fromarray(transformed["image"]), mask

    def transform(self, image_size: tuple[int, int]) -> A.ReplayCompose:
        width, height = image_size
        transforms = [
            A.Affine(
                scale=(0.88, 1.12),
                translate_percent=(-0.08, 0.08),
                rotate=(-4, 4),
                shear=(-3, 3),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.25, contrast=0.35, saturation=0.25, hue=0.0, p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.2, 0.8), p=0.15),
        ]
        if self.crop_cutout:
            crop = A.RandomResizedCrop(size=(height, width), scale=(0.72, 1.0), ratio=(0.85, 1.15), p=0.7)
            cutout = A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.04, 0.18), fill="random", p=0.5)
            transforms.insert(0, crop)
            transforms.append(cutout)
        return A.ReplayCompose(transforms)


def horizontal_flip_was_applied(replay: Replay) -> bool:
    transforms = replay["transforms"]
    return any(transform["__class_fullname__"] == "HorizontalFlip" and transform["applied"] for transform in transforms)


def swap_left_right_bits(bitfield: np.ndarray, left_right_pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    swapped = bitfield.copy()
    for left_bit, right_bit in left_right_pairs:
        left = bitfield & (1 << left_bit)
        right = bitfield & (1 << right_bit)
        swapped &= np.uint8(255 ^ (1 << left_bit))
        swapped &= np.uint8(255 ^ (1 << right_bit))
        swapped |= left << (right_bit - left_bit)
        swapped |= right >> (right_bit - left_bit)
    return swapped
