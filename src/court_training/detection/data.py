from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int64
from torch import Tensor

from court_training import dataset


class Target(TypedDict):
    boxes: Float[Tensor, "D 4"]
    labels: Int64[Tensor, "D"]


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    boxes_cxcywh: np.ndarray
    labels: np.ndarray


def load_split(root: Path) -> list[DetectionSample]:
    image_paths = sorted((root / "images").glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No images found under {root}")
    samples = []
    for image_path in image_paths:
        detection_path = dataset.annotation_path(root, image_path, "detections", ".npz")
        boxes_cxcywh, labels = dataset.read_detections(detection_path)
        samples.append(DetectionSample(image_path=image_path, boxes_cxcywh=boxes_cxcywh, labels=labels))
    return samples


def subsample_indexes(count: int, max_samples: int, seed: int) -> list[int]:
    if max_samples <= 0 or count <= max_samples:
        return list(range(count))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(count, size=max_samples, replace=False).tolist())


def collate(batch: list[dataset.TorchSample]) -> tuple[Float[Tensor, "B 3 H W"], list[Target]]:
    images = torch.stack([sample["image"] for sample in batch])
    targets: list[Target] = [{"boxes": sample["boxes_cxcywh"], "labels": sample["labels"]} for sample in batch]
    return images, targets


def boxes_to_xyxy(boxes_cxcywh: Float[Tensor, "D 4"], width: float, height: float) -> Float[Tensor, "D 4"]:
    centers = boxes_cxcywh[:, :2]
    half_sizes = boxes_cxcywh[:, 2:] / 2
    corners = torch.cat([centers - half_sizes, centers + half_sizes], dim=1)
    return corners * torch.tensor([width, height, width, height], dtype=boxes_cxcywh.dtype)
