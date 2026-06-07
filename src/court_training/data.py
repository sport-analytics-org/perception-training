from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch.utils.data import Dataset

from court_training.constants import IMAGE_MEAN, IMAGE_STD, MASK_NAMES


class MaskSample(TypedDict):
    image: Float[torch.Tensor, "3 height width"]
    mask: Float[torch.Tensor, "n_masks height width"]


class MaskDataset(Dataset):
    def __init__(self, root: Path, dataset_names: tuple[str, ...]) -> None:
        self.items = image_mask_pairs(root, dataset_names)
        if not self.items:
            raise ValueError(f"No image/mask pairs found under {root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> MaskSample:
        image_path, mask_path = self.items[index]
        image = Image.open(image_path).convert("RGB")
        bitfield = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        return {"image": image_to_tensor(image), "mask": bitfield_to_masks(bitfield)}


def image_mask_pairs(root: Path, dataset_names: tuple[str, ...]) -> list[tuple[Path, Path]]:
    pairs = []
    for dataset_name in dataset_names:
        image_root = root / dataset_name / "images"
        mask_root = root / dataset_name / "masks"
        for image_path in sorted(image_root.glob("*/*.jpg")):
            mask_path = mask_root / image_path.relative_to(image_root).with_suffix(".webp")
            if mask_path.is_file():
                pairs.append((image_path, mask_path))
    return pairs


def image_to_tensor(image: Image.Image) -> Float[torch.Tensor, "3 height width"]:
    array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (image_tensor - IMAGE_MEAN) / IMAGE_STD


def bitfield_to_masks(bitfield: np.ndarray) -> Float[torch.Tensor, "n_masks height width"]:
    masks = [(bitfield & (1 << bit)) > 0 for bit in range(len(MASK_NAMES))]
    return torch.from_numpy(np.stack(masks).astype(np.float32))
