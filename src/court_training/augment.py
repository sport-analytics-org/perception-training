import random

import albumentations as A
import numpy as np
from PIL import Image

from court_training.masks import LEFT_RIGHT_PAIRS


class BasketballAugment:
    def __init__(self) -> None:
        self.transforms = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.22, p=1),
                        A.ColorJitter(brightness=0.18, contrast=0.22, saturation=0.18, hue=0.03, p=1),
                        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=18, val_shift_limit=12, p=1),
                    ],
                    p=0.75,
                ),
                A.ImageCompression(compression_type="jpeg", quality_range=(55, 95), p=0.25),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1),
                        A.MotionBlur(blur_limit=(3, 5), p=1),
                        A.GaussNoise(std_range=(0.01, 0.05), p=1),
                    ],
                    p=0.35,
                ),
            ]
        )

    def __call__(self, image: Image.Image, bitfield: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        transformed = self.transforms(image=np.asarray(image), mask=bitfield)
        image = Image.fromarray(transformed["image"])
        bitfield = transformed["mask"].astype(np.uint8)

        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            bitfield = swap_left_right_bits(np.fliplr(bitfield).copy())

        return image, bitfield


def swap_left_right_bits(bitfield: np.ndarray) -> np.ndarray:
    swapped = bitfield.copy()
    for left_bit, right_bit in LEFT_RIGHT_PAIRS:
        left = bitfield & (1 << left_bit)
        right = bitfield & (1 << right_bit)
        swapped &= np.uint8(255 ^ (1 << left_bit))
        swapped &= np.uint8(255 ^ (1 << right_bit))
        swapped |= left << (right_bit - left_bit)
        swapped |= right >> (right_bit - left_bit)
    return swapped
