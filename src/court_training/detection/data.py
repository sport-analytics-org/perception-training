from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int64, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from court_training import dataset
from court_training.constants import IMAGE_MEAN, IMAGE_STD

BASKETBALL_DETECTION_CLASSES = ("ball", "player", "number", "referee", "rim")
CATEGORY_ALIASES = {
    "basketball": "ball",
    "hoop": "rim",
}


class NumpySample(TypedDict):
    image: UInt8[np.ndarray, "H W 3"]
    boxes_cxcywh: Float[np.ndarray, "D 4"]
    labels: Int64[np.ndarray, " D"]


class Target(TypedDict):
    boxes: Float[Tensor, "D 4"]
    labels: Int64[Tensor, " D"]


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    boxes_xywh: np.ndarray
    category_names: tuple[str, ...]


class DetectionDataset(Dataset):
    """Square-resized images with normalized cxcywh boxes and class-index labels."""

    def __init__(
        self,
        samples: list[DetectionSample],
        class_names: tuple[str, ...],
        resolution: int,
        box_scales: dict[str, float],
        transform: Callable[[NumpySample], NumpySample] | None = None,
    ) -> None:
        self.resolution = resolution
        self.transform = transform
        self.items = [(sample.image_path, *encode_boxes(sample, class_names, box_scales)) for sample in samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Float[Tensor, "3 H W"], Target]:
        sample = self.load(index)
        if self.transform:
            sample = self.transform(sample)
        image = torch.from_numpy(sample["image"].astype(np.float32) / 255.0).permute(2, 0, 1)
        image = (image - IMAGE_MEAN) / IMAGE_STD
        target: Target = {
            "boxes": torch.from_numpy(sample["boxes_cxcywh"]),
            "labels": torch.from_numpy(sample["labels"]),
        }
        return image, target

    def load(self, index: int) -> NumpySample:
        image_path, boxes_cxcywh, labels = self.items[index]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
        return {
            "image": np.array(image, dtype=np.uint8),
            "boxes_cxcywh": boxes_cxcywh,
            "labels": labels,
        }


def collate(batch: list[tuple[Tensor, Target]]) -> tuple[Float[Tensor, "B 3 H W"], list[Target]]:
    images = torch.stack([image for image, _ in batch])
    targets = [target for _, target in batch]
    return images, targets


def load_split(root: Path) -> list[DetectionSample]:
    pairs = dataset.image_annotation_pairs(root, "detections", ".npz")
    if not pairs:
        raise ValueError(f"No image/detection pairs found under {root}")
    return [load_sample(image_path, detection_path) for image_path, detection_path in pairs]


def load_sample(image_path: Path, detection_path: Path) -> DetectionSample:
    data = np.load(detection_path)
    boxes_xywh = data["boxes_xywh"].astype(np.float32)
    category_names = tuple(canonical_category(name) for name in data["category_names"].astype(str).tolist())
    if len(boxes_xywh) != len(category_names):
        raise ValueError(f"{detection_path} has {len(boxes_xywh)} boxes and {len(category_names)} category names")
    return DetectionSample(image_path=image_path, boxes_xywh=boxes_xywh, category_names=category_names)


def filter_by_class(samples: list[DetectionSample], class_names: tuple[str, ...]) -> list[DetectionSample]:
    class_set = set(class_names)
    return [sample for sample in samples if class_set.intersection(sample.category_names)]


def subsample(samples: list[DetectionSample], max_samples: int, seed: int) -> list[DetectionSample]:
    if max_samples <= 0 or len(samples) <= max_samples:
        return samples
    rng = np.random.default_rng(seed)
    indexes = sorted(rng.choice(len(samples), size=max_samples, replace=False).tolist())
    return [samples[index] for index in indexes]


def parse_classes(classes: str) -> tuple[str, ...]:
    names = tuple(canonical_category(name.strip()) for name in classes.split(","))
    unknown = sorted(set(names) - set(BASKETBALL_DETECTION_CLASSES))
    if unknown:
        raise ValueError(f"Unknown detection classes: {', '.join(unknown)}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate detection classes: {', '.join(names)}")
    return names


def canonical_category(name: str) -> str:
    return CATEGORY_ALIASES.get(name, name)


def encode_boxes(
    sample: DetectionSample,
    class_names: tuple[str, ...],
    box_scales: dict[str, float],
) -> tuple[Float[np.ndarray, "D 4"], Int64[np.ndarray, " D"]]:
    boxes = []
    labels = []
    for box, category_name in zip(sample.boxes_xywh, sample.category_names, strict=True):
        if category_name not in class_names:
            continue
        x, y, width, height = scale_box(box, box_scales.get(category_name, 1.0)).tolist()
        boxes.append([x + width / 2, y + height / 2, width, height])
        labels.append(class_names.index(category_name))
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)


def boxes_to_xyxy(boxes_cxcywh: Float[Tensor, "D 4"], width: float, height: float) -> Float[Tensor, "D 4"]:
    centers = boxes_cxcywh[:, :2]
    half_sizes = boxes_cxcywh[:, 2:] / 2
    corners = torch.cat([centers - half_sizes, centers + half_sizes], dim=1)
    return corners * torch.tensor([width, height, width, height], dtype=boxes_cxcywh.dtype)


def scale_box(box: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return box
    x, y, box_width, box_height = box.tolist()
    scaled_width = min(box_width * scale, 1.0)
    scaled_height = min(box_height * scale, 1.0)
    scaled_x = min(max(x - (scaled_width - box_width) / 2, 0.0), 1.0 - scaled_width)
    scaled_y = min(max(y - (scaled_height - box_height) / 2, 0.0), 1.0 - scaled_height)
    return np.array([scaled_x, scaled_y, scaled_width, scaled_height], dtype=np.float32)
