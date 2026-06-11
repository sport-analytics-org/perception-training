from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int64
from torch import Tensor

from court_training import dataset

BASKETBALL_DETECTION_CLASSES = ("ball", "player", "number", "referee", "rim")
CATEGORY_ALIASES = {
    "basketball": "ball",
    "hoop": "rim",
}


class Target(TypedDict):
    boxes: Float[Tensor, "D 4"]
    labels: Int64[Tensor, " D"]


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    boxes_xywh: np.ndarray
    category_names: tuple[str, ...]


def load_split(root: Path) -> list[DetectionSample]:
    pairs = dataset.image_annotation_pairs(root, "detections", ".npz")
    if not pairs:
        raise ValueError(f"No image/detection pairs found under {root}")
    samples = []
    for image_path, detection_path in pairs:
        boxes_xywh, category_names = dataset.read_detections(detection_path)
        canonical_names = tuple(canonical_category(name) for name in category_names)
        samples.append(DetectionSample(image_path=image_path, boxes_xywh=boxes_xywh, category_names=canonical_names))
    return samples


def filter_by_class(samples: list[DetectionSample], class_names: tuple[str, ...]) -> list[DetectionSample]:
    class_set = set(class_names)
    return [sample for sample in samples if class_set.intersection(sample.category_names)]


def subsample_indexes(count: int, max_samples: int, seed: int) -> list[int]:
    if max_samples <= 0 or count <= max_samples:
        return list(range(count))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(count, size=max_samples, replace=False).tolist())


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


def collate(batch: list[dataset.TorchSample]) -> tuple[Float[Tensor, "B 3 H W"], list[Target]]:
    images = torch.stack([sample["image"] for sample in batch])
    targets: list[Target] = [{"boxes": sample["boxes_cxcywh"], "labels": sample["labels"]} for sample in batch]
    return images, targets


def encode_boxes(
    boxes_xywh: Float[np.ndarray, "D 4"],
    category_names: tuple[str, ...],
    class_names: tuple[str, ...],
    box_scales: dict[str, float],
) -> tuple[Float[np.ndarray, "D 4"], Int64[np.ndarray, " D"]]:
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
