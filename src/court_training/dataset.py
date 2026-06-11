import json
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


class NumpySample(TypedDict):
    image: UInt8[np.ndarray, "H W 3"]
    mask: Float[np.ndarray, "H W N"]
    keypoints: Float[np.ndarray, "K 2"]
    visibility: Float[np.ndarray, "*K"]


class TorchSample(TypedDict):
    image: Float[Tensor, "3 H W"]
    mask: Float[Tensor, "N H W"]
    keypoints: Float[Tensor, "K 2"]
    visibility: Float[Tensor, "*K"]


class MaskDataset(Dataset):
    def __init__(
        self,
        root: Path,
        load_mask: Callable[[np.ndarray], Float[np.ndarray, "H W N"]],
        image_size: tuple[int, int],
        transform: Callable[[NumpySample], NumpySample] | None = None,
    ) -> None:
        self.load_mask = load_mask
        self.image_size = image_size
        self.transform = transform
        self.keypoint_root = root / "keypoints"
        self.items = image_annotation_pairs(root, "masks", ".webp")
        if not self.items:
            raise ValueError(f"No image/mask pairs found under {root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> TorchSample:
        sample = self.load(index, self.image_size)
        if self.transform:
            sample = self.transform(sample)
        image = torch.from_numpy(sample["image"].astype(np.float32) / 255.0).permute(2, 0, 1)
        image = (image - IMAGE_MEAN) / IMAGE_STD
        return {
            "image": image,
            "mask": torch.from_numpy(sample["mask"]).permute(2, 0, 1),
            "keypoints": torch.from_numpy(sample["keypoints"]),
            "visibility": torch.from_numpy(sample["visibility"]),
        }

    def load(self, index: int, image_size: tuple[int, int]) -> NumpySample:
        image_path, mask_path = self.items[index]
        image = Image.open(image_path).convert("RGB")
        bitfield = Image.open(mask_path).convert("L")
        height, width = image_size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        bitfield = bitfield.resize((width, height), Image.Resampling.NEAREST)
        image_array = np.array(image, dtype=np.uint8)
        bitfield_array = np.array(bitfield, dtype=np.uint8)
        keypoint_path = self.keypoint_root / mask_path.with_suffix(".json").name
        keypoints, visibility = read_keypoints(keypoint_path)
        return {
            "image": image_array,
            "mask": self.load_mask(bitfield_array),
            "keypoints": keypoints,
            "visibility": visibility,
        }


def image_annotation_pairs(root: Path, annotation_dir: str, suffix: str) -> list[tuple[Path, Path]]:
    pairs = []
    image_root = root / "images"
    annotation_root = root / annotation_dir
    for image_path in sorted(image_root.glob("*.jpg")):
        annotation_path = annotation_root / image_path.with_suffix(suffix).name
        if annotation_path.is_file():
            pairs.append((image_path, annotation_path))
    return pairs


def read_keypoints(path: Path) -> tuple[Float[np.ndarray, "K 2"], Float[np.ndarray, "*K"]]:
    data = json.loads(path.read_text())
    points = data["points"]
    keypoints = np.array([point["position"] for point in points], dtype=np.float32)
    visibility = np.array([point["visible"] for point in points], dtype=np.float32)
    return keypoints, visibility
