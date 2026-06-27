import json
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict

import courts_and_fields as cnf
import numpy as np
import torch
from jaxtyping import Bool, Float, Int64, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from perception_training.constants import IMAGE_MEAN, IMAGE_STD

BASKETBALL_MASK_NAMES = tuple(cnf.NbaCourt.areas())
BASKETBALL_KEYPOINT_NAMES = tuple(cnf.NbaCourt.keypoints())
BASKETBALL_DETECTION_CLASSES = ("ball", "player", "number", "referee", "rim")
BASKETBALL_DETECTION_ATTRIBUTES = ("in_basket", "in_possession", "jump_shot", "layup_dunk", "shot_block")
BASKETBALL_ATTRIBUTE_BASE_CLASSES = {
    "in_basket": "ball",
    "in_possession": "player",
    "jump_shot": "player",
    "layup_dunk": "player",
    "shot_block": "player",
}


class NumpySample(TypedDict):
    image: UInt8[np.ndarray, "H W 3"]
    mask: NotRequired[Float[np.ndarray, "H W N"]]
    keypoints: NotRequired[Float[np.ndarray, "K 2"]]
    visibility: NotRequired[Float[np.ndarray, "*K"]]
    boxes_xywh: NotRequired[Float[np.ndarray, "D 4"]]
    labels: NotRequired[Int64[np.ndarray, "D"]]
    attributes: NotRequired[Bool[np.ndarray, "D A"]]


class TorchSample(TypedDict):
    image: Float[Tensor, "3 H W"]
    mask: NotRequired[Float[Tensor, "N H W"]]
    keypoints: NotRequired[Float[Tensor, "K 2"]]
    visibility: NotRequired[Float[Tensor, "*K"]]
    boxes_xywh: NotRequired[Float[Tensor, "D 4"]]
    labels: NotRequired[Int64[Tensor, "D"]]
    attributes: NotRequired[Bool[Tensor, "D A"]]


class Target(TypedDict):
    boxes_xywh: Float[Tensor, "D 4"]
    labels: Int64[Tensor, "D"]
    attributes: Bool[Tensor, "D A"]


class CourtDataset(Dataset):
    """Images from a flat export with any combination of masks, keypoints, and boxes.

    Every modality is enabled by a boolean; box labels index into class_names.
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
        class_names: tuple[str, ...] = BASKETBALL_DETECTION_CLASSES,
        attribute_names: tuple[str, ...] = BASKETBALL_DETECTION_ATTRIBUTES,
        transform: Callable[[NumpySample], NumpySample] | None = None,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.load_masks = load_masks
        self.load_keypoints = load_keypoints
        self.class_names = class_names
        self.attribute_names = attribute_names
        self.transform = transform

        image_paths = sorted((root / "images").glob("*.jpg"))
        self.boxes = None
        if load_bbox:
            self.boxes = {}
            for path in image_paths:
                detection_path = annotation_path(root, path, "detections", ".npz")
                boxes_xywh, labels, attributes = read_detections(detection_path, class_names, attribute_names)
                if len(labels):
                    self.boxes[path] = (boxes_xywh, labels, attributes)
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
            tensors["attributes"] = torch.from_numpy(sample["attributes"])
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
            boxes_xywh, labels, attributes = self.boxes[image_path]
            sample["boxes_xywh"] = boxes_xywh
            sample["labels"] = labels
            sample["attributes"] = attributes
        return sample


def collate(batch: list[TorchSample]) -> tuple[Float[Tensor, "B 3 H W"], list[Target]]:
    """Detection batches stay ragged: stacked images plus per-image box targets."""
    images = torch.stack([sample["image"] for sample in batch])
    targets: list[Target] = [
        {
            "boxes_xywh": sample["boxes_xywh"],
            "labels": sample["labels"],
            "attributes": sample["attributes"],
        }
        for sample in batch
    ]
    return images, targets


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


def read_detections(
    path: Path,
    class_names: tuple[str, ...] = BASKETBALL_DETECTION_CLASSES,
    attribute_names: tuple[str, ...] = BASKETBALL_DETECTION_ATTRIBUTES,
) -> tuple[Float[np.ndarray, "D 4"], Int64[np.ndarray, "D"], Bool[np.ndarray, "D A"]]:
    data = np.load(path)
    boxes_xywh = data["boxes_xywh"].astype(np.float32)
    category_names = data["category_names"].astype(str).tolist()
    if len(boxes_xywh) != len(category_names):
        raise ValueError(f"{path} has {len(boxes_xywh)} boxes and {len(category_names)} category names")
    for name in category_names:
        if name not in class_names:
            raise ValueError(f"{path} has unknown detection class: {name}")
    labels = [class_names.index(name) for name in category_names]
    attributes = read_detection_attributes(data, path, len(category_names), attribute_names)
    return boxes_xywh.reshape(-1, 4), np.array(labels, dtype=np.int64), attributes


def read_detection_attributes(
    data: np.lib.npyio.NpzFile,
    path: Path,
    detection_count: int,
    attribute_names: tuple[str, ...],
) -> Bool[np.ndarray, "D A"]:
    if "attributes" not in data:
        return np.zeros((detection_count, len(attribute_names)), dtype=np.bool_)

    raw_attributes = data["attributes"].astype(np.bool_)
    if raw_attributes.ndim != 2 or raw_attributes.shape[0] != detection_count:
        raise ValueError(f"{path} has invalid attribute shape {raw_attributes.shape} for {detection_count} detections")
    raw_attribute_names = data["attribute_names"].astype(str).tolist()
    if raw_attributes.shape[1] != len(raw_attribute_names):
        raise ValueError(
            f"{path} has {raw_attributes.shape[1]} attribute columns and {len(raw_attribute_names)} attribute names"
        )

    attributes = np.zeros((detection_count, len(attribute_names)), dtype=np.bool_)
    raw_name_indexes = {name: index for index, name in enumerate(raw_attribute_names)}
    for output_index, name in enumerate(attribute_names):
        raw_index = raw_name_indexes.get(name)
        if raw_index is not None:
            attributes[:, output_index] = raw_attributes[:, raw_index]
    return attributes
