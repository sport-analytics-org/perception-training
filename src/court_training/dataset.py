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
CATEGORY_ALIASES = {
    "basketball": "ball",
    "hoop": "rim",
}


class NumpySample(TypedDict):
    image: UInt8[np.ndarray, "H W 3"]
    mask: NotRequired[Float[np.ndarray, "H W N"]]
    keypoints: NotRequired[Float[np.ndarray, "K 2"]]
    visibility: NotRequired[Float[np.ndarray, "*K"]]
    boxes_cxcywh: NotRequired[Float[np.ndarray, "D 4"]]
    labels: NotRequired[Int64[np.ndarray, "D"]]


class TorchSample(TypedDict):
    image: Float[Tensor, "3 H W"]
    mask: NotRequired[Float[Tensor, "N H W"]]
    keypoints: NotRequired[Float[Tensor, "K 2"]]
    visibility: NotRequired[Float[Tensor, "*K"]]
    boxes_cxcywh: NotRequired[Float[Tensor, "D 4"]]
    labels: NotRequired[Int64[Tensor, "D"]]


class CourtDataset(Dataset):
    """Images from a flat export with any combination of masks, keypoints, and boxes.

    Masks and keypoints are enabled by booleans; boxes by the `box_classes` to encode.
    Images missing an enabled annotation are skipped; with boxes enabled, images without
    any box of the requested classes are dropped.
    """

    def __init__(
        self,
        root: Path,
        image_size: tuple[int, int],
        load_masks: bool = False,
        load_keypoints: bool = False,
        box_classes: tuple[str, ...] | None = None,
        box_scales: dict[str, float] | None = None,
        transform: Callable[[NumpySample], NumpySample] | None = None,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.load_masks = load_masks
        self.load_keypoints = load_keypoints
        self.transform = transform

        image_paths = sorted((root / "images").glob("*.jpg"))
        if load_masks:
            image_paths = [path for path in image_paths if annotation_path(root, path, "masks", ".webp").is_file()]
        if load_keypoints:
            image_paths = [path for path in image_paths if annotation_path(root, path, "keypoints", ".json").is_file()]
        self.boxes = None
        if box_classes is not None:
            self.boxes = {}
            for path in image_paths:
                detection_path = annotation_path(root, path, "detections", ".npz")
                if not detection_path.is_file():
                    continue
                boxes_xywh, category_names = read_detections(detection_path)
                boxes_cxcywh, labels = encode_boxes(boxes_xywh, category_names, box_classes, box_scales or {})
                if len(labels):
                    self.boxes[path] = (boxes_cxcywh, labels)
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
        image = torch.from_numpy(sample["image"].astype(np.float32) / 255.0).permute(2, 0, 1)
        tensors: TorchSample = {"image": (image - IMAGE_MEAN) / IMAGE_STD}
        if "mask" in sample:
            tensors["mask"] = torch.from_numpy(sample["mask"]).permute(2, 0, 1)
        if "keypoints" in sample:
            tensors["keypoints"] = torch.from_numpy(sample["keypoints"])
            tensors["visibility"] = torch.from_numpy(sample["visibility"])
        if "boxes_cxcywh" in sample:
            tensors["boxes_cxcywh"] = torch.from_numpy(sample["boxes_cxcywh"])
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
            boxes_cxcywh, labels = self.boxes[image_path]
            sample["boxes_cxcywh"] = boxes_cxcywh
            sample["labels"] = labels
        return sample


def annotation_path(root: Path, image_path: Path, annotation_dir: str, suffix: str) -> Path:
    return root / annotation_dir / image_path.with_suffix(suffix).name


def image_annotation_pairs(root: Path, annotation_dir: str, suffix: str) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted((root / "images").glob("*.jpg")):
        candidate = annotation_path(root, image_path, annotation_dir, suffix)
        if candidate.is_file():
            pairs.append((image_path, candidate))
    return pairs


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


def read_detections(path: Path) -> tuple[Float[np.ndarray, "D 4"], tuple[str, ...]]:
    data = np.load(path)
    boxes_xywh = data["boxes_xywh"].astype(np.float32)
    category_names = tuple(data["category_names"].astype(str).tolist())
    if len(boxes_xywh) != len(category_names):
        raise ValueError(f"{path} has {len(boxes_xywh)} boxes and {len(category_names)} category names")
    return boxes_xywh, category_names


def encode_boxes(
    boxes_xywh: Float[np.ndarray, "D 4"],
    category_names: tuple[str, ...],
    class_names: tuple[str, ...],
    box_scales: dict[str, float],
) -> tuple[Float[np.ndarray, "D 4"], Int64[np.ndarray, "D"]]:
    boxes = []
    labels = []
    for box, name in zip(boxes_xywh, category_names, strict=True):
        category_name = canonical_category(name)
        if category_name not in class_names:
            continue
        x, y, width, height = scale_box(box, box_scales.get(category_name, 1.0)).tolist()
        boxes.append([x + width / 2, y + height / 2, width, height])
        labels.append(class_names.index(category_name))
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)


def canonical_category(name: str) -> str:
    return CATEGORY_ALIASES.get(name, name)


def scale_box(box: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return box
    x, y, box_width, box_height = box.tolist()
    scaled_width = min(box_width * scale, 1.0)
    scaled_height = min(box_height * scale, 1.0)
    scaled_x = min(max(x - (scaled_width - box_width) / 2, 0.0), 1.0 - scaled_width)
    scaled_y = min(max(y - (scaled_height - box_height) / 2, 0.0), 1.0 - scaled_height)
    return np.array([scaled_x, scaled_y, scaled_width, scaled_height], dtype=np.float32)
