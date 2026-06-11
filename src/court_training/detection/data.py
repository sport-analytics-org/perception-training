import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from court_training.dataset import CourtDataset

BASKETBALL_DETECTION_CLASSES = ("ball", "player", "number", "referee", "rim")
CATEGORY_ALIASES = {
    "basketball": "ball",
    "hoop": "rim",
}


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    boxes_xywh: np.ndarray
    category_names: tuple[str, ...]


def load_split(root: Path) -> list[DetectionSample]:
    dataset = CourtDataset(root, load_detections=True)
    samples = []
    for index in range(len(dataset)):
        sample = dataset.load(index)
        category_names = tuple(canonical_category(name) for name in sample["category_names"])
        samples.append(
            DetectionSample(
                image_path=sample["image_path"],
                boxes_xywh=sample["boxes_xywh"],
                category_names=category_names,
            )
        )
    if not samples:
        raise ValueError(f"No image/detection pairs found under {root}")
    return samples


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


def category_ids(class_names: tuple[str, ...]) -> dict[str, int]:
    return {name: index + 1 for index, name in enumerate(class_names)}


def write_coco_dataset(
    train_root: Path,
    val_root: Path,
    output_root: Path,
    class_names: tuple[str, ...],
    val_max_samples: int,
    seed: int,
    train_box_scales: dict[str, float],
) -> None:
    train_samples = filter_by_class(load_split(train_root), class_names)
    val_samples = subsample(filter_by_class(load_split(val_root), class_names), val_max_samples, seed)
    write_coco_split(train_samples, output_root / "train", class_names, train_box_scales)
    write_coco_split(val_samples, output_root / "valid", class_names, {})
    write_coco_split(val_samples, output_root / "test", class_names, {})


def filter_by_class(samples: list[DetectionSample], class_names: tuple[str, ...]) -> list[DetectionSample]:
    class_set = set(class_names)
    return [sample for sample in samples if class_set.intersection(sample.category_names)]


def subsample(samples: list[DetectionSample], max_samples: int, seed: int) -> list[DetectionSample]:
    if max_samples <= 0 or len(samples) <= max_samples:
        return samples
    rng = np.random.default_rng(seed)
    indexes = sorted(rng.choice(len(samples), size=max_samples, replace=False).tolist())
    return [samples[index] for index in indexes]


def write_coco_split(
    samples: list[DetectionSample],
    output_root: Path,
    class_names: tuple[str, ...],
    box_scales: dict[str, float],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        link_image(sample.image_path, output_root / sample.image_path.name)
    coco = coco_annotations(samples, class_names, box_scales)
    (output_root / "_annotations.coco.json").write_text(json.dumps(coco))


def coco_annotations(
    samples: list[DetectionSample],
    class_names: tuple[str, ...],
    box_scales: dict[str, float],
) -> dict:
    category_id_by_name = category_ids(class_names)
    images = []
    annotations = []
    annotation_id = 1
    for image_id, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": sample.image_path.name, "width": width, "height": height})
        for box, category_name in zip(sample.boxes_xywh, sample.category_names, strict=True):
            if category_name not in category_id_by_name:
                continue
            scaled_box = scale_box(box, box_scales.get(category_name, 1.0))
            x, y, box_width, box_height = box_to_pixels(scaled_box, width, height)
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id_by_name[category_name],
                "bbox": [x, y, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            }
            annotations.append(annotation)
            annotation_id += 1
    categories = [{"id": category_id, "name": name} for name, category_id in category_id_by_name.items()]
    return {"images": images, "annotations": annotations, "categories": categories}


def scale_box(box: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return box
    x, y, box_width, box_height = box.tolist()
    scaled_width = min(box_width * scale, 1.0)
    scaled_height = min(box_height * scale, 1.0)
    scaled_x = min(max(x - (scaled_width - box_width) / 2, 0.0), 1.0 - scaled_width)
    scaled_y = min(max(y - (scaled_height - box_height) / 2, 0.0), 1.0 - scaled_height)
    return np.array([scaled_x, scaled_y, scaled_width, scaled_height], dtype=np.float32)


def box_to_pixels(box: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    x, y, box_width, box_height = box.tolist()
    return x * width, y * height, box_width * width, box_height * height


def link_image(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    destination.symlink_to(source.resolve().relative_to(destination.parent.resolve(), walk_up=True))
