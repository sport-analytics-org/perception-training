import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from court_training.constants import IMAGE_MEAN, IMAGE_STD


class NumpySample(TypedDict):
    image_path: Path
    image: UInt8[np.ndarray, "H W 3"]
    mask: NotRequired[Float[np.ndarray, "H W N"]]
    keypoints: NotRequired[Float[np.ndarray, "K 2"]]
    visibility: NotRequired[Float[np.ndarray, "*K"]]
    boxes_xywh: NotRequired[Float[np.ndarray, "D 4"]]
    category_names: NotRequired[tuple[str, ...]]


class TorchSample(TypedDict):
    image: Float[Tensor, "3 H W"]
    mask: NotRequired[Float[Tensor, "N H W"]]
    keypoints: NotRequired[Float[Tensor, "K 2"]]
    visibility: NotRequired[Float[Tensor, "*K"]]
    boxes_xywh: NotRequired[Float[Tensor, "D 4"]]
    category_names: NotRequired[tuple[str, ...]]


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    mask_path: Path | None = None
    keypoint_path: Path | None = None
    detection_path: Path | None = None


class CourtDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_size: tuple[int, int] | None = None,
        load_mask: Callable[[np.ndarray], Float[np.ndarray, "H W N"]] | None = None,
        load_masks: bool = False,
        load_keypoints: bool = False,
        load_detections: bool = False,
        transform: Callable[[NumpySample], NumpySample] | None = None,
    ) -> None:
        if load_masks and load_mask is None:
            raise ValueError("load_mask is required when load_masks=True")
        self.load_mask = load_mask
        self.image_size = image_size
        self.transform = transform
        self.load_masks = load_masks
        self.load_keypoints = load_keypoints
        self.load_detections = load_detections
        self.items = dataset_items(root, load_masks, load_keypoints, load_detections)
        if not self.items:
            raise ValueError(f"No samples found under {root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> TorchSample:
        sample = self.load(index)
        if self.transform:
            sample = self.transform(sample)
        image = torch.from_numpy(sample["image"].astype(np.float32) / 255.0).permute(2, 0, 1)
        image = (image - IMAGE_MEAN) / IMAGE_STD
        tensors: TorchSample = {"image": image}
        if "mask" in sample:
            tensors["mask"] = torch.from_numpy(sample["mask"]).permute(2, 0, 1)
        if "keypoints" in sample:
            tensors["keypoints"] = torch.from_numpy(sample["keypoints"])
            tensors["visibility"] = torch.from_numpy(sample["visibility"])
        if "boxes_xywh" in sample:
            tensors["boxes_xywh"] = torch.from_numpy(sample["boxes_xywh"])
            tensors["category_names"] = sample["category_names"]
        return tensors

    def load(self, index: int) -> NumpySample:
        item = self.items[index]
        image_path = item.image_path
        image = Image.open(image_path).convert("RGB")
        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        image_array = np.array(image, dtype=np.uint8)
        sample: NumpySample = {"image_path": image_path, "image": image_array}
        if self.load_masks:
            assert item.mask_path is not None
            bitfield = Image.open(item.mask_path).convert("L")
            if self.image_size is not None:
                bitfield = bitfield.resize((width, height), Image.Resampling.NEAREST)
            bitfield_array = np.array(bitfield, dtype=np.uint8)
            assert self.load_mask is not None
            sample["mask"] = self.load_mask(bitfield_array)
        if self.load_keypoints:
            assert item.keypoint_path is not None
            sample["keypoints"], sample["visibility"] = read_keypoints(item.keypoint_path)
        if self.load_detections:
            assert item.detection_path is not None
            sample["boxes_xywh"], sample["category_names"] = read_detections(item.detection_path)
        return sample


def dataset_items(
    root: Path,
    load_masks: bool,
    load_keypoints: bool,
    load_detections: bool,
) -> list[DatasetItem]:
    items = []
    image_root = root / "images"
    mask_root = root / "masks"
    keypoint_root = root / "keypoints"
    detection_root = root / "detections"
    for image_path in sorted(image_root.glob("*.jpg")):
        mask_path = mask_root / image_path.with_suffix(".webp").name
        keypoint_path = keypoint_root / image_path.with_suffix(".json").name
        detection_path = detection_root / image_path.with_suffix(".npz").name
        if load_masks and not mask_path.is_file():
            continue
        if load_keypoints and not keypoint_path.is_file():
            continue
        if load_detections and not detection_path.is_file():
            continue
        items.append(
            DatasetItem(
                image_path=image_path,
                mask_path=mask_path if load_masks else None,
                keypoint_path=keypoint_path if load_keypoints else None,
                detection_path=detection_path if load_detections else None,
            )
        )
    return items


def read_keypoints(path: Path) -> tuple[Float[np.ndarray, "K 2"], Float[np.ndarray, "*K"]]:
    data = json.loads(path.read_text())
    points = data["points"]
    keypoints = np.array([point["position"] for point in points], dtype=np.float32)
    visibility = np.array([point["visible"] for point in points], dtype=np.float32)
    return keypoints, visibility


def read_detections(path: Path) -> tuple[Float[np.ndarray, "D 4"], tuple[str, ...]]:
    data = np.load(path)
    boxes_xywh = data["boxes_xywh"].astype(np.float32)
    category_names = tuple(data["category_names"].astype(str).tolist())
    if len(boxes_xywh) != len(category_names):
        raise ValueError(f"{path} has {len(boxes_xywh)} boxes and {len(category_names)} category names")
    return boxes_xywh, category_names
