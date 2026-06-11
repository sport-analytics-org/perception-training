import json
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int64, UInt8
from PIL import Image
from sportanalytics import NbaCourt
from torch import Tensor
from torch.utils.data import Dataset

from court_training.constants import IMAGE_MEAN, IMAGE_STD

BASKETBALL_MASK_NAMES = tuple(NbaCourt.areas())
BASKETBALL_KEYPOINT_NAMES = tuple(NbaCourt.keypoints())
BASKETBALL_DETECTION_CLASSES = ("ball", "player", "number", "referee", "rim")


class NumpySample(TypedDict):
    image: UInt8[np.ndarray, "H W 3"]
    mask: NotRequired[Float[np.ndarray, "H W N"]]
    keypoints: NotRequired[Float[np.ndarray, "K 2"]]
    visibility: NotRequired[Float[np.ndarray, "*K"]]
    boxes_xywh: NotRequired[Float[np.ndarray, "D 4"]]
    labels: NotRequired[Int64[np.ndarray, "D"]]


class TorchSample(TypedDict):
    image: Float[Tensor, "3 H W"]
    mask: NotRequired[Float[Tensor, "N H W"]]
    keypoints: NotRequired[Float[Tensor, "K 2"]]
    visibility: NotRequired[Float[Tensor, "*K"]]
    boxes_xywh: NotRequired[Float[Tensor, "D 4"]]
    labels: NotRequired[Int64[Tensor, "D"]]


class CourtDataset(Dataset):
    """Images from a flat export with any combination of masks, keypoints, and boxes.

    Every modality is enabled by a boolean; box labels index into BASKETBALL_DETECTION_CLASSES.
    Every image must have each enabled annotation; with boxes enabled, images without any box
    are dropped.
    """

    def __init__(
        self,
        root: Path,
        image_size: tuple[int, int],
        load_masks: bool = False,
        load_keypoints: bool = False,
        load_bbox: bool = False,
        transform: Callable[[NumpySample], NumpySample] | None = None,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.load_masks = load_masks
        self.load_keypoints = load_keypoints
        self.transform = transform

        image_paths = sorted((root / "images").glob("*.jpg"))
        self.boxes = None
        if load_bbox:
            self.boxes = {}
            for path in image_paths:
                detection_path = annotation_path(root, path, "detections", ".npz")
                boxes_xywh, labels = read_detections(detection_path)
                if len(labels):
                    self.boxes[path] = (boxes_xywh, labels)
            image_paths = [path for path in image_paths if path in self.boxes]
        self.image_paths = image_paths
        if not image_paths:
            raise ValueError(f"No samples found under {root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> TorchSample:
        sample = self.load(index)
        if self.transform:
            sample = self.transform(sample)
        return self.to_tensor(sample)

    def to_tensor(self, sample: NumpySample) -> TorchSample:
        image = torch.from_numpy(sample["image"].astype(np.float32) / 255.0).permute(2, 0, 1)
        tensors: TorchSample = {"image": (image - IMAGE_MEAN) / IMAGE_STD}
        if "mask" in sample:
            tensors["mask"] = torch.from_numpy(sample["mask"]).permute(2, 0, 1)
        if "keypoints" in sample:
            tensors["keypoints"] = torch.from_numpy(sample["keypoints"])
            tensors["visibility"] = torch.from_numpy(sample["visibility"])
        if "boxes_xywh" in sample:
            tensors["boxes_xywh"] = torch.from_numpy(sample["boxes_xywh"])
            tensors["labels"] = torch.from_numpy(sample["labels"])
        return tensors

    def load(self, index: int) -> NumpySample:
        image_path = self.image_paths[index]
        height, width = self.image_size
        image = Image.open(image_path).convert("RGB")
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        sample: NumpySample = {"image": np.array(image, dtype=np.uint8)}
        if self.load_masks:
            sample["mask"] = read_mask(annotation_path(self.root, image_path, "masks", ".webp"), self.image_size)
        if self.load_keypoints:
            keypoints, visibility = read_keypoints(annotation_path(self.root, image_path, "keypoints", ".json"))
            sample["keypoints"] = keypoints
            sample["visibility"] = visibility
        if self.boxes is not None:
            boxes_xywh, labels = self.boxes[image_path]
            sample["boxes_xywh"] = boxes_xywh
            sample["labels"] = labels
        return sample


def annotation_path(root: Path, image_path: Path, annotation_dir: str, suffix: str) -> Path:
    return root / annotation_dir / image_path.with_suffix(suffix).name


def read_mask(path: Path, image_size: tuple[int, int]) -> Float[np.ndarray, "H W N"]:
    height, width = image_size
    bitfield = Image.open(path).convert("L").resize((width, height), Image.Resampling.NEAREST)
    bitfield_array = np.array(bitfield, dtype=np.uint8)
    masks = [(bitfield_array & (1 << bit)) > 0 for bit in range(len(BASKETBALL_MASK_NAMES))]
    return np.stack(masks, axis=-1).astype(np.float32)


def read_keypoints(path: Path) -> tuple[Float[np.ndarray, "K 2"], Float[np.ndarray, "*K"]]:
    data = json.loads(path.read_text())
    points = data["points"]
    keypoints = np.array([point["position"] for point in points], dtype=np.float32)
    visibility = np.array([point["visible"] for point in points], dtype=np.float32)
    return keypoints, visibility


def read_detections(path: Path) -> tuple[Float[np.ndarray, "D 4"], Int64[np.ndarray, "D"]]:
    data = np.load(path)
    boxes_xywh = data["boxes_xywh"].astype(np.float32)
    category_names = data["category_names"].astype(str).tolist()
    if len(boxes_xywh) != len(category_names):
        raise ValueError(f"{path} has {len(boxes_xywh)} boxes and {len(category_names)} category names")
    for name in category_names:
        if name not in BASKETBALL_DETECTION_CLASSES:
            raise ValueError(f"{path} has unknown detection class: {name}")
    labels = [BASKETBALL_DETECTION_CLASSES.index(name) for name in category_names]
    return boxes_xywh.reshape(-1, 4), np.array(labels, dtype=np.int64)
