import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from court_training.constants import LEFT_RIGHT_PAIRS


class BasketballAugment:
    def __init__(self, crop_cutout: bool = True) -> None:
        self.crop_cutout = crop_cutout

    def __call__(self, image: Image.Image, bitfield: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        if self.crop_cutout and random.random() < 0.7:
            image, bitfield = random_resized_crop(image, bitfield)

        image, bitfield = random_affine(image, bitfield)
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            bitfield = swap_left_right_bits(np.fliplr(bitfield).copy())

        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.35))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
        if self.crop_cutout and random.random() < 0.5:
            image = random_cutout(image)
        return image, bitfield


def random_resized_crop(image: Image.Image, bitfield: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    width, height = image.size
    crop_area = random.uniform(0.72, 1.0) * width * height
    crop_ratio = (width / height) * random.uniform(0.85, 1.15)
    crop_width = min(width, round((crop_area * crop_ratio) ** 0.5))
    crop_height = min(height, round((crop_area / crop_ratio) ** 0.5))
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    box = (left, top, left + crop_width, top + crop_height)

    image = image.crop(box).resize((width, height), Image.Resampling.BILINEAR)
    mask = Image.fromarray(bitfield).crop(box).resize((width, height), Image.Resampling.NEAREST)
    return image, np.asarray(mask, dtype=np.uint8)


def random_affine(image: Image.Image, bitfield: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    mask = Image.fromarray(bitfield)
    angle = random.uniform(-4.0, 4.0)
    translate = (
        round(random.uniform(-0.08, 0.08) * image.width),
        round(random.uniform(-0.08, 0.08) * image.height),
    )
    scale = random.uniform(0.88, 1.12)
    shear = (random.uniform(-3.0, 3.0), random.uniform(-2.0, 2.0))
    image = TF.affine(
        image,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )
    mask = TF.affine(mask, angle, translate, scale, shear, InterpolationMode.NEAREST, fill=0)
    return image, np.asarray(mask, dtype=np.uint8)


def random_cutout(image: Image.Image) -> Image.Image:
    image = image.copy()
    pixels = np.asarray(image)
    fill = tuple(int(value) for value in pixels.reshape(-1, 3).mean(axis=0))
    for _ in range(random.randint(1, 4)):
        width = round(random.uniform(0.04, 0.18) * image.width)
        height = round(random.uniform(0.04, 0.18) * image.height)
        left = random.randint(0, max(0, image.width - width))
        top = random.randint(0, max(0, image.height - height))
        image.paste(fill, (left, top, left + width, top + height))
    return image


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
