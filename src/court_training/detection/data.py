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


def class_id(name: str, class_names: tuple[str, ...]) -> int:
    return class_names.index(canonical_category(name))


def write_yolo_dataset(
    train_root: Path,
    val_root: Path,
    output_root: Path,
    class_names: tuple[str, ...] = BASKETBALL_DETECTION_CLASSES,
) -> Path:
    train_samples = load_split(train_root)
    val_samples = load_split(val_root)
    write_yolo_split(train_samples, output_root, "train", class_names)
    write_yolo_split(val_samples, output_root, "val", class_names)
    data_yaml = output_root / "data.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    data_yaml.write_text(f"path: {output_root}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return data_yaml


def write_yolo_split(
    samples: list[DetectionSample],
    output_root: Path,
    split: str,
    class_names: tuple[str, ...],
) -> None:
    image_root = output_root / "images" / split
    label_root = output_root / "labels" / split
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        image_path = image_root / sample.image_path.name
        link_image(sample.image_path, image_path)
        lines = []
        for box, category_name in zip(sample.boxes_xywh, sample.category_names, strict=True):
            values = [class_id(category_name, class_names), *box.tolist()]
            lines.append(" ".join(str(value) for value in values))
        (label_root / sample.image_path.with_suffix(".txt").name).write_text("\n".join(lines) + ("\n" if lines else ""))


def write_coco_dataset(
    train_root: Path,
    val_root: Path,
    output_root: Path,
    class_names: tuple[str, ...] = BASKETBALL_DETECTION_CLASSES,
) -> Path:
    write_coco_split(load_split(train_root), output_root / "train", class_names)
    write_coco_split(load_split(val_root), output_root / "valid", class_names)
    write_coco_split(load_split(val_root), output_root / "test", class_names)
    return output_root


def write_coco_split(samples: list[DetectionSample], output_root: Path, class_names: tuple[str, ...]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    annotation_id = 1
    for image_id, sample in enumerate(samples, start=1):
        image = Image.open(sample.image_path)
        width, height = image.size
        output_image = output_root / sample.image_path.name
        link_image(sample.image_path, output_image)
        images.append({"id": image_id, "file_name": output_image.name, "width": width, "height": height})
        for box, category_name in zip(sample.boxes_xywh, sample.category_names, strict=True):
            x, y, box_width, box_height = normalized_xywh_to_pixels(box, width, height)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id(category_name, class_names) + 1,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(class_names)]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    (output_root / "_annotations.coco.json").write_text(json.dumps(coco))


def normalized_xywh_to_pixels(box: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    x_center, y_center, box_width, box_height = box.tolist()
    pixel_width = box_width * width
    pixel_height = box_height * height
    x = x_center * width - pixel_width / 2
    y = y_center * height - pixel_height / 2
    return x, y, pixel_width, pixel_height


def link_image(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    relative_source = os.path.relpath(source.resolve(), destination.parent.resolve())
    destination.symlink_to(relative_source)
