import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

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
    image_root = root / "images"
    detection_root = root / "detections"
    samples = []
    for image_path in sorted(image_root.glob("*.jpg")):
        detection_path = detection_root / image_path.with_suffix(".npz").name
        if detection_path.is_file():
            samples.append(load_sample(image_path, detection_path))
    if not samples:
        raise ValueError(f"No image/detection pairs found under {root}")
    return samples


def load_sample(image_path: Path, detection_path: Path) -> DetectionSample:
    data = np.load(detection_path)
    boxes_xywh = data["boxes_xywh"].astype(np.float32)
    category_names = tuple(canonical_category(name) for name in data["category_names"].astype(str).tolist())
    if len(boxes_xywh) != len(category_names):
        raise ValueError(f"{detection_path} has {len(boxes_xywh)} boxes and {len(category_names)} category names")
    return DetectionSample(image_path=image_path, boxes_xywh=boxes_xywh, category_names=category_names)


def canonical_category(name: str) -> str:
    return CATEGORY_ALIASES.get(name, name)


def canonical_classes(class_names: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(canonical_category(name) for name in class_names)
    if not names:
        raise ValueError("At least one detection class is required")
    unknown = sorted(set(names) - set(BASKETBALL_DETECTION_CLASSES))
    if unknown:
        raise ValueError(f"Unknown detection classes: {', '.join(unknown)}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate detection classes: {', '.join(names)}")
    return names


def parse_classes(classes: str | None) -> tuple[str, ...]:
    if classes is None:
        return BASKETBALL_DETECTION_CLASSES
    return canonical_classes(tuple(name.strip() for name in classes.split(",") if name.strip()))


def write_coco_dataset(
    train_root: Path,
    val_root: Path,
    output_root: Path,
    class_names: tuple[str, ...] = BASKETBALL_DETECTION_CLASSES,
    val_max_samples: int = 0,
    sample_seed: int = 42,
    train_box_scales: dict[str, float] | None = None,
) -> Path:
    class_names = canonical_classes(class_names)
    train_samples = select_samples(load_split(train_root), class_names, max_samples=0, seed=sample_seed)
    val_samples = select_samples(load_split(val_root), class_names, val_max_samples, sample_seed)
    write_coco_split(train_samples, output_root / "train", class_names, box_scales=train_box_scales)
    write_coco_split(val_samples, output_root / "valid", class_names)
    write_coco_split(val_samples, output_root / "test", class_names)
    return output_root


def select_samples(
    samples: list[DetectionSample],
    class_names: tuple[str, ...],
    max_samples: int,
    seed: int,
) -> list[DetectionSample]:
    class_set = set(class_names)
    filtered = [sample for sample in samples if class_set.intersection(sample.category_names)]
    if max_samples <= 0 or len(filtered) <= max_samples:
        return filtered
    rng = np.random.default_rng(seed)
    indexes = sorted(rng.choice(len(filtered), size=max_samples, replace=False).tolist())
    return [filtered[index] for index in indexes]


def write_coco_split(
    samples: list[DetectionSample],
    output_root: Path,
    class_names: tuple[str, ...],
    box_scales: dict[str, float] | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    annotation_id = 1
    category_ids = {name: index + 1 for index, name in enumerate(class_names)}
    for image_id, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            width, height = image.size
        output_image = output_root / sample.image_path.name
        link_image(sample.image_path, output_image)
        images.append({"id": image_id, "file_name": output_image.name, "width": width, "height": height})
        for box, category_name in zip(sample.boxes_xywh, sample.category_names, strict=True):
            category_id = category_ids.get(category_name)
            if category_id is None:
                continue
            x, y, box_width, box_height = box_to_pixels(
                scale_box(box, box_scales.get(category_name, 1.0) if box_scales else 1.0),
                width,
                height,
            )
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            }
            annotations.append(annotation)
            annotation_id += 1
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(class_names)]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    (output_root / "_annotations.coco.json").write_text(json.dumps(coco))


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
    pixel_width = box_width * width
    pixel_height = box_height * height
    return x * width, y * height, pixel_width, pixel_height


def link_image(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    relative_source = os.path.relpath(source.resolve(), destination.parent.resolve())
    destination.symlink_to(relative_source)
