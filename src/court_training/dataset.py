from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch.utils.data import Dataset

from court_training.constants import IMAGE_MEAN, IMAGE_STD


class Transform(Protocol):
    def __call__(self, image: Image.Image, bitfield: np.ndarray) -> tuple[Image.Image, np.ndarray]: ...


class MaskSample(TypedDict):
    image: Float[torch.Tensor, "3 H W"]
    mask: Float[torch.Tensor, "N H W"]


class MaskDataset(Dataset):
    def __init__(
        self,
        root: Path,
        load_mask: Callable[[np.ndarray], Float[torch.Tensor, "N H W"]],
        transform: Transform | None = None,
    ) -> None:
        self.load_mask = load_mask
        self.transform = transform
        self.items = image_mask_pairs(root)
        if not self.items:
            raise ValueError(f"No image/mask pairs found under {root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> MaskSample:
        image_path, mask_path = self.items[index]
        image = Image.open(image_path).convert("RGB")
        bitfield = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if self.transform:
            image, bitfield = self.transform(image, bitfield)
        return {"image": image_to_tensor(image), "mask": self.load_mask(bitfield)}


def image_mask_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    image_root = root / "images"
    mask_root = root / "masks"
    for image_path in sorted(image_root.glob("*.jpg")):
        mask_path = mask_root / image_path.with_suffix(".webp").name
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def image_to_tensor(image: Image.Image) -> Float[torch.Tensor, "3 H W"]:
    array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (image_tensor - IMAGE_MEAN) / IMAGE_STD
