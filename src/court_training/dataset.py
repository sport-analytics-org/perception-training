from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from court_training.constants import IMAGE_MEAN, IMAGE_STD


class MaskSample(TypedDict):
    image: UInt8[np.ndarray, "H W 3"]
    mask: Float[np.ndarray, "H W N"]


class MaskDataset(Dataset):
    def __init__(
        self,
        root: Path,
        load_mask: Callable[[np.ndarray], Float[np.ndarray, "H W N"]],
        image_size: tuple[int, int],
        transform: Callable[[MaskSample], MaskSample] | None = None,
    ) -> None:
        self.load_mask = load_mask
        self.image_size = image_size
        self.transform = transform
        self.items = image_mask_pairs(root)
        if not self.items:
            raise ValueError(f"No image/mask pairs found under {root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> MaskSample:
        image_path, mask_path = self.items[index]
        image = Image.open(image_path).convert("RGB")
        bitfield = Image.open(mask_path).convert("L")
        height, width = self.image_size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        bitfield = bitfield.resize((width, height), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.uint8)
        bitfield_array = np.asarray(bitfield, dtype=np.uint8)
        sample: MaskSample = {"image": image_array, "mask": self.load_mask(bitfield_array)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def image_mask_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    image_root = root / "images"
    mask_root = root / "masks"
    for image_path in sorted(image_root.glob("*.jpg")):
        mask_path = mask_root / image_path.with_suffix(".webp").name
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def image_to_tensor(image: Image.Image) -> Float[Tensor, "3 H W"]:
    array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (image_tensor - IMAGE_MEAN) / IMAGE_STD
