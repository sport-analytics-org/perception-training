from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from court_training.constants import IMAGE_MEAN, IMAGE_STD
from court_training.masks import MASK_NAMES


class Transform(Protocol):
    def __call__(self, image: Image.Image, bitfield: np.ndarray) -> tuple[Image.Image, np.ndarray]: ...


class MaskDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        items: list[tuple[Path, Path]],
        image_size: tuple[int, int],
        transform: Transform | None = None,
    ) -> None:
        self.transform = transform
        self.image_size = image_size
        self.items = items
        if not self.items:
            raise ValueError("No image/mask pairs found")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.items[index]
        image = Image.open(image_path).convert("RGB")
        bitfield = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        image, bitfield = resize_pair(image, bitfield, self.image_size)
        if self.transform:
            image, bitfield = self.transform(image, bitfield)
        return image_to_tensor(image), bitfield_to_masks(bitfield)

    @staticmethod
    def items_for(root: Path, dataset_name: str) -> list[tuple[Path, Path]]:
        return image_mask_pairs(root, dataset_name)


def image_mask_pairs(root: Path, dataset_name: str) -> list[tuple[Path, Path]]:
    pairs = []
    image_root = root / dataset_name / "images"
    mask_root = root / dataset_name / "masks"
    for image_path in sorted(image_root.glob("*/*.jpg")):
        mask_path = mask_root / image_path.relative_to(image_root).with_suffix(".webp")
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def resize_pair(
    image: Image.Image,
    bitfield: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[Image.Image, np.ndarray]:
    if image.size == image_size and bitfield.shape == (image_size[1], image_size[0]):
        return image, bitfield
    image = image.resize(image_size, Image.Resampling.BILINEAR)
    mask = Image.fromarray(bitfield).resize(image_size, Image.Resampling.NEAREST)
    return image, np.asarray(mask, dtype=np.uint8)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (image_tensor - IMAGE_MEAN) / IMAGE_STD


def bitfield_to_masks(bitfield: np.ndarray) -> torch.Tensor:
    masks = [(bitfield & (1 << bit)) > 0 for bit in range(len(MASK_NAMES))]
    return torch.from_numpy(np.stack(masks).astype(np.float32))
