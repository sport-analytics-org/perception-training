from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETR
from torchvision.ops import batched_nms

from court_training.detection import data


@dataclass(frozen=True)
class Prediction:
    category_id: int
    bbox: list[float]
    score: float


def predict(
    model: RFDETR,
    image_path: Path,
    category_ids: dict[str, int],
    resolution: int,
    hflip: bool,
    threshold: float,
    nms_iou: float,
    max_detections: int,
) -> list[Prediction]:
    model_class_names = tuple(data.canonical_category(name) for name in model.class_names)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, _ = image.size
        variants = [(image, False)]
        if hflip:
            variants.append((image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), True))

        boxes = []
        scores = []
        labels = []
        for variant, flipped in variants:
            detections = model.predict(
                variant,
                threshold=threshold,
                shape=(resolution, resolution),
                include_source_image=False,
            )
            xyxy = np.asarray(detections.xyxy, dtype=np.float32)
            if not len(xyxy):
                continue
            if flipped:
                xyxy[:, [0, 2]] = width - xyxy[:, [2, 0]]

            class_ids = np.asarray(detections.class_id, dtype=np.int64)
            boxes.append(xyxy)
            scores.append(np.asarray(detections.confidence, dtype=np.float32))
            labels.append(np.array([category_ids[model_class_names[class_id]] for class_id in class_ids]))

    if not boxes:
        return []

    merged_boxes = np.concatenate(boxes)
    merged_scores = np.concatenate(scores)
    merged_labels = np.concatenate(labels)
    keep = batched_nms(
        torch.from_numpy(merged_boxes),
        torch.from_numpy(merged_scores),
        torch.from_numpy(merged_labels),
        nms_iou,
    )

    predictions = []
    for index in keep[:max_detections].tolist():
        x1, y1, x2, y2 = merged_boxes[index].tolist()
        predictions.append(
            Prediction(
                category_id=int(merged_labels[index]),
                bbox=[x1, y1, x2 - x1, y2 - y1],
                score=float(merged_scores[index]),
            )
        )
    return predictions
