import json
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int64, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from court_training.constants import IMAGE_MEAN, IMAGE_STD


class NumpySample(TypedDict):
    image: UInt8[np.ndarray, "H W 3"]
    mask: NotRequired[Float[np.ndarray, "H W N"]]
    keypoints: NotRequired[Float[np.ndarray, "K 2"]]
    visibility: NotRequired[Float[np.ndarray, "*K"]]
    boxes_cxcywh: NotRequired[Float[np.ndarray, "D 4"]]
    labels: NotRequired[Int64[np.ndarray, " D"]]


class TorchSample(TypedDict):
    image: Float[Tensor, "3 H W"]
    mask: NotRequired[Float[Tensor, "N H W"]]
    keypoints: NotRequired[Float[Tensor, "K 2"]]
    visibility: NotRequired[Float[Tensor, "*K"]]
    boxes_cxcywh: NotRequired[Float[Tensor, "D 4"]]
    labels: NotRequired[Int64[Tensor, " D"]]


class CourtDataset(Dataset):
    """Images from a flat export with any combination of masks, keypoints, and boxes.

    Each modality is enabled by its argument: masks by the `load_mask` decoder, boxes by the
    `load_boxes` encoder, keypoints by `load_keypoints`. Images missing an enabled annotation
    are skipped; with boxes enabled, images whose encoder returns no boxes are dropped.
    """

    def __init__(
        self,
        root: Path,
        image_size: tuple[int, int],
        load_mask: Callable[[UInt8[np.ndarray, "H W"]], Float[np.ndarray, "H W N"]] | None = None,
        load_keypoints: bool = False,
        load_boxes: Callable[
            [Float[np.ndarray, "D 4"], tuple[str, ...]],
            tuple[Float[np.ndarray, "D 4"], Int64[np.ndarray, " D"]],
        ]
        | None = None,
        transform: Callable[[NumpySample], NumpySample] | None = None,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.load_mask = load_mask
        self.load_keypoints = load_keypoints
        self.transform = transform

        image_paths = sorted((root / "images").glob("*.jpg"))
        if load_mask is not None:
            image_paths = [path for path in image_paths if annotation_path(root, path, "masks", ".webp").is_file()]
        if load_keypoints:
            image_paths = [path for path in image_paths if annotation_path(root, path, "keypoints", ".json").is_file()]
        self.boxes = None
        if load_boxes is not None:
            self.boxes = {}
            for path in image_paths:
                detection_path = annotation_path(root, path, "detections", ".npz")
                if not detection_path.is_file():
                    continue
                boxes_cxcywh, labels = load_boxes(*read_detections(detection_path))
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
        if self.load_mask is not None:
            bitfield = Image.open(annotation_path(self.root, image_path, "masks", ".webp")).convert("L")
            bitfield = bitfield.resize((width, height), Image.Resampling.NEAREST)
            sample["mask"] = self.load_mask(np.array(bitfield, dtype=np.uint8))
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
