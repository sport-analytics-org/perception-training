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


class Sample(TypedDict):
    image: UInt8[np.ndarray, "H W 3"]
    mask: Float[np.ndarray, "H W N"]
    keypoints: Float[np.ndarray, "K 2"]
    keypoint_visibility: Float[np.ndarray, "*K"]


class TensorSample(TypedDict):
    image: Float[Tensor, "3 H W"]
    mask: Float[Tensor, "N H W"]
    keypoints: Float[Tensor, "K 2"]
    keypoint_visibility: Float[Tensor, "*K"]


class MaskDataset(Dataset):
    def __init__(
        self,
        root: Path,
        load_mask: Callable[[np.ndarray], Float[np.ndarray, "H W N"]],
        image_size: tuple[int, int],
        transform: Callable[[Sample], Sample] | None = None,
    ) -> None:
        self.load_mask = load_mask
        self.image_size = image_size
        self.transform = transform
        self.keypoint_root = root / "keypoints"
        self.items = image_mask_pairs(root)
        if not self.items:
            raise ValueError(f"No image/mask pairs found under {root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> TensorSample:
        sample = self.load(index, self.image_size)
        if self.transform:
            sample = self.transform(sample)
        return to_tensor(sample)

    def load(self, index: int, image_size: tuple[int, int]) -> Sample:
        image_path, mask_path = self.items[index]
        image = Image.open(image_path).convert("RGB")
        bitfield = Image.open(mask_path).convert("L")
        height, width = image_size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        bitfield = bitfield.resize((width, height), Image.Resampling.NEAREST)
        image_array = np.array(image, dtype=np.uint8, copy=True)
        bitfield_array = np.array(bitfield, dtype=np.uint8, copy=True)
        keypoint_path = self.keypoint_root / mask_path.with_suffix(".json").name
        keypoints, visibility = read_keypoints(keypoint_path)
        return {
            "image": image_array,
            "mask": self.load_mask(bitfield_array),
            "keypoints": keypoints,
            "keypoint_visibility": visibility,
        }


def image_mask_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    image_root = root / "images"
    mask_root = root / "masks"
    for image_path in sorted(image_root.glob("*.jpg")):
        mask_path = mask_root / image_path.with_suffix(".webp").name
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def read_keypoints(path: Path) -> tuple[Float[np.ndarray, "K 2"], Float[np.ndarray, "*K"]]:
    data = json.loads(path.read_text())
    points = data["points"]
    keypoints = np.array([point["position"] for point in points], dtype=np.float32, copy=True)
    visibility = np.array([point["visible"] for point in points], dtype=np.float32, copy=True)
    return keypoints, visibility


def image_to_tensor(image: Image.Image) -> Float[Tensor, "3 H W"]:
    array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (image_tensor - IMAGE_MEAN) / IMAGE_STD


def to_tensor(sample: Sample) -> TensorSample:
    image = torch.from_numpy(sample["image"].astype(np.float32) / 255.0).permute(2, 0, 1)
    image = (image - IMAGE_MEAN) / IMAGE_STD
    return {
        "image": image,
        "mask": torch.from_numpy(sample["mask"]).permute(2, 0, 1),
        "keypoints": torch.from_numpy(sample["keypoints"]),
        "keypoint_visibility": torch.from_numpy(sample["keypoint_visibility"]),
    }
