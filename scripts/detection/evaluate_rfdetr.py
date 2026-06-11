import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETR
from torchvision.ops import nms
from tqdm import tqdm

from court_training.detection import data

app = typer.Typer(help="Evaluate an RF-DETR checkpoint on exported basketball detections.")


@dataclass(frozen=True)
class Prediction:
    image_id: int
    category_id: int
    bbox: list[float]
    score: float


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: Path
    classes: tuple[str, ...]


@app.command()
def main(
    checkpoint: Path,
    val_root: Path,
    output_json: Path,
    classes: str | None = typer.Option(None, help="Comma-separated class names to evaluate."),
    extra_checkpoint: Annotated[
        list[Path] | None,
        typer.Option("--extra-checkpoint", help="Additional checkpoint to ensemble. Repeat as needed."),
    ] = None,
    extra_classes: Annotated[
        list[str] | None,
        typer.Option("--extra-classes", help="Class filter for the matching --extra-checkpoint."),
    ] = None,
    threshold: float = typer.Option(0.001, help="Low confidence threshold before NMS and COCO evaluation."),
    nms_iou: float = typer.Option(0.6, help="Per-class NMS IoU threshold after TTA merge."),
    resolutions: Annotated[
        list[int] | None,
        typer.Option("--resolution", help="Inference resolution. Repeat for multi-resolution TTA."),
    ] = None,
    hflip: bool = typer.Option(False, "--hflip/--no-hflip", help="Use horizontal-flip TTA."),
    max_detections: int = typer.Option(300, help="Maximum detections per image after NMS."),
) -> None:
    class_names = data.parse_classes(classes)
    category_ids = {name: index + 1 for index, name in enumerate(class_names)}
    model_specs = build_model_specs(checkpoint, class_names, extra_checkpoint or [], extra_classes or [])
    samples = data.load_split(val_root.expanduser().resolve())
    coco_gt_path = write_ground_truth(samples, output_json.parent / "ground_truth.coco.json", class_names)
    models = [(RFDETR.from_checkpoint(spec.checkpoint.expanduser().resolve()), spec.classes) for spec in model_specs]
    predictions = []
    for image_id, sample in enumerate(tqdm(samples, desc="evaluate"), start=1):
        image_predictions = predict_with_models(
            models,
            sample.image_path,
            category_ids,
            threshold=threshold,
            resolutions=resolutions or [704],
            hflip=hflip,
            nms_iou=nms_iou,
            max_detections=max_detections,
        )
        predictions.extend(
            Prediction(
                image_id=image_id,
                category_id=prediction.category_id,
                bbox=prediction.bbox,
                score=prediction.score,
            )
            for prediction in image_predictions
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps([prediction.__dict__ for prediction in predictions]) + "\n")
    metrics = evaluate(coco_gt_path, output_json, category_ids)
    metrics_path = output_json.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    typer.echo(json.dumps(metrics, indent=2))


def build_model_specs(
    checkpoint: Path,
    class_names: tuple[str, ...],
    extra_checkpoints: list[Path],
    extra_classes: list[str],
) -> list[ModelSpec]:
    if len(extra_checkpoints) != len(extra_classes):
        raise typer.BadParameter("--extra-checkpoint and --extra-classes must be provided in matching pairs")
    specs = [ModelSpec(checkpoint, class_names)]
    specs.extend(
        ModelSpec(extra_checkpoint, data.parse_classes(classes))
        for extra_checkpoint, classes in zip(extra_checkpoints, extra_classes, strict=True)
    )
    return specs


def write_ground_truth(samples: list[data.DetectionSample], path: Path, class_names: tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    category_ids = {name: index + 1 for index, name in enumerate(class_names)}
    images = []
    annotations = []
    annotation_id = 1
    for image_id, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": sample.image_path.name, "width": width, "height": height})
        for box, category_name in zip(sample.boxes_xywh, sample.category_names, strict=True):
            category_id = category_ids.get(category_name)
            if category_id is None:
                continue
            x, y, box_width, box_height = data.box_to_pixels(box, width, height)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(class_names)],
    }
    path.write_text(json.dumps(coco) + "\n")
    return path


def predict_with_models(
    models: list[tuple[RFDETR, tuple[str, ...]]],
    image_path: Path,
    category_ids: dict[str, int],
    threshold: float,
    resolutions: list[int],
    hflip: bool,
    nms_iou: float,
    max_detections: int,
) -> list[Prediction]:
    boxes = []
    scores = []
    labels = []
    for model, class_names in models:
        model_boxes, model_scores, model_labels = predict_image(
            model,
            image_path,
            category_ids,
            class_names,
            threshold=threshold,
            resolutions=resolutions,
            hflip=hflip,
        )
        boxes.extend(model_boxes)
        scores.extend(model_scores)
        labels.extend(model_labels)

    if not boxes:
        return []

    boxes_np = np.concatenate(boxes)
    scores_np = np.concatenate(scores)
    labels_np = np.concatenate(labels)
    keep = keep_after_nms(boxes_np, scores_np, labels_np, nms_iou, max_detections)
    predictions = []
    category_names = {category_id: name for name, category_id in category_ids.items()}
    for index in keep:
        category_id = int(labels_np[index])
        if category_id not in category_names:
            continue
        x1, y1, x2, y2 = boxes_np[index].tolist()
        predictions.append(
            Prediction(
                image_id=0,
                category_id=category_id,
                bbox=[x1, y1, x2 - x1, y2 - y1],
                score=float(scores_np[index]),
            )
        )
    return predictions


def predict_image(
    model: RFDETR,
    image_path: Path,
    category_ids: dict[str, int],
    allowed_classes: tuple[str, ...],
    threshold: float,
    resolutions: list[int],
    hflip: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    model_class_names = tuple(data.canonical_category(name) for name in model.class_names)
    allowed_class_names = set(allowed_classes)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        variants = [(image, False)]
        if hflip:
            variants.append((image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), True))

        boxes = []
        scores = []
        labels = []
        for resolution in resolutions:
            for variant_image, flipped in variants:
                detections = model.predict(
                    variant_image,
                    threshold=threshold,
                    shape=(resolution, resolution),
                    include_source_image=False,
                )
                xyxy = np.asarray(detections.xyxy, dtype=np.float32)
                if xyxy.size == 0:
                    continue
                if flipped:
                    xyxy[:, [0, 2]] = width - xyxy[:, [2, 0]]
                class_ids = np.asarray(detections.class_id, dtype=np.int64)
                category_labels = np.array(
                    [category_id_for_class_id(class_id, model_class_names, category_ids) for class_id in class_ids],
                    dtype=np.int64,
                )
                keep = np.array(
                    [
                        label > 0 and model_class_names[class_id] in allowed_class_names
                        for class_id, label in zip(class_ids, category_labels, strict=True)
                    ]
                )
                if keep.any():
                    boxes.append(xyxy[keep])
                    scores.append(np.asarray(detections.confidence, dtype=np.float32)[keep])
                    labels.append(category_labels[keep])
    return boxes, scores, labels


def category_id_for_class_id(
    class_id: int,
    model_class_names: tuple[str, ...],
    category_ids: dict[str, int],
) -> int:
    if class_id < 0 or class_id >= len(model_class_names):
        return 0
    return category_ids.get(model_class_names[class_id], 0)


def keep_after_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    nms_iou: float,
    max_detections: int,
) -> list[int]:
    kept = []
    for label in sorted(set(labels.tolist())):
        label_indexes = np.flatnonzero(labels == label)
        keep = nms(
            torch.as_tensor(boxes[label_indexes], dtype=torch.float32),
            torch.as_tensor(scores[label_indexes], dtype=torch.float32),
            nms_iou,
        )
        kept.extend(label_indexes[keep.numpy()].tolist())
    kept.sort(key=lambda index: float(scores[index]), reverse=True)
    return kept[:max_detections]


def evaluate(
    coco_gt_path: Path,
    prediction_path: Path,
    category_ids: dict[str, int],
) -> dict[str, float | dict[str, float]]:
    coco_gt = COCO(str(coco_gt_path))
    coco_dt = coco_gt.loadRes(str(prediction_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.maxDets = [1, 10, 300]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    per_class_ap50 = {}
    precision = coco_eval.eval["precision"]
    area_index = 0
    max_det_index = 2
    iou_index = np.where(np.isclose(coco_eval.params.iouThrs, 0.5))[0][0]
    for class_name, category_id in category_ids.items():
        category_index = list(coco_eval.params.catIds).index(category_id)
        class_precision = precision[iou_index, :, category_index, area_index, max_det_index]
        valid = class_precision[class_precision > -1]
        if len(valid):
            per_class_ap50[class_name] = float(valid.mean())
    valid_precision = precision[:, :, :, area_index, max_det_index]
    valid_precision = valid_precision[valid_precision > -1]
    ap50_precision = precision[iou_index, :, :, area_index, max_det_index]
    ap50_precision = ap50_precision[ap50_precision > -1]
    ap75_index = np.where(np.isclose(coco_eval.params.iouThrs, 0.75))[0][0]
    ap75_precision = precision[ap75_index, :, :, area_index, max_det_index]
    ap75_precision = ap75_precision[ap75_precision > -1]
    return {
        "map50_95": float(valid_precision.mean()) if len(valid_precision) else 0.0,
        "map50": float(ap50_precision.mean()) if len(ap50_precision) else 0.0,
        "map75": float(ap75_precision.mean()) if len(ap75_precision) else 0.0,
        "per_class_ap50": per_class_ap50,
    }


if __name__ == "__main__":
    app()
