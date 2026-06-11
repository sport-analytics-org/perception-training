import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETR
from tqdm import tqdm

from court_training.detection import data, inference

app = typer.Typer(help="Evaluate an RF-DETR checkpoint on basketball detections.")
DEFAULT_CLASSES = "ball,player,referee"


@dataclass(frozen=True)
class Prediction:
    image_id: int
    category_id: int
    bbox: list[float]
    score: float


@app.command()
def main(
    checkpoint: Path,
    val_root: Path,
    output_json: Path,
    classes: str = typer.Option(DEFAULT_CLASSES, help="Comma-separated class names to evaluate."),
    resolution: int = typer.Option(704, help="Square inference resolution."),
    hflip: bool = typer.Option(False, "--hflip/--no-hflip", help="Use horizontal-flip TTA."),
    threshold: float = typer.Option(0.001, help="Confidence threshold before NMS and COCO evaluation."),
    nms_iou: float = typer.Option(0.6, help="Per-class NMS IoU threshold after TTA merge."),
    max_detections: int = typer.Option(300, help="Maximum detections per image after NMS."),
) -> None:
    if resolution <= 0:
        raise typer.BadParameter("Resolution must be positive.")

    class_names = data.parse_classes(classes)
    category_ids = {name: index + 1 for index, name in enumerate(class_names)}
    samples = data.load_split(val_root.expanduser().resolve())
    model = RFDETR.from_checkpoint(checkpoint.expanduser().resolve())

    output_json.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_path = output_json.with_suffix(".ground_truth.coco.json")
    write_ground_truth(samples, ground_truth_path, category_ids)

    predictions = []
    for image_id, sample in enumerate(tqdm(samples, desc="evaluate"), start=1):
        image_predictions = inference.predict(
            model,
            sample.image_path,
            category_ids,
            resolution=resolution,
            hflip=hflip,
            threshold=threshold,
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

    output_json.write_text(json.dumps([prediction.__dict__ for prediction in predictions]) + "\n")
    metrics = evaluate(ground_truth_path, output_json, category_ids)
    output_json.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    typer.echo(json.dumps(metrics, indent=2))


def write_ground_truth(
    samples: list[data.DetectionSample],
    path: Path,
    category_ids: dict[str, int],
) -> None:
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

    categories = [{"id": category_id, "name": name} for name, category_id in category_ids.items()]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    path.write_text(json.dumps(coco) + "\n")


def evaluate(
    ground_truth_path: Path,
    prediction_path: Path,
    category_ids: dict[str, int],
) -> dict[str, float | dict[str, float]]:
    coco_gt = COCO(str(ground_truth_path))
    coco_dt = coco_gt.loadRes(str(prediction_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.maxDets = [1, 10, 300]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precision = coco_eval.eval["precision"]
    iou50_index = np.where(np.isclose(coco_eval.params.iouThrs, 0.5))[0][0]
    iou75_index = np.where(np.isclose(coco_eval.params.iouThrs, 0.75))[0][0]
    all_area_index = 0
    max_det_index = 2

    ap50 = precision[iou50_index, :, :, all_area_index, max_det_index]
    ap75 = precision[iou75_index, :, :, all_area_index, max_det_index]
    map50_95 = precision[:, :, :, all_area_index, max_det_index]
    cat_ids = list(coco_eval.params.catIds)

    return {
        "map50_95": mean_valid(map50_95),
        "map50": mean_valid(ap50),
        "map75": mean_valid(ap75),
        "per_class_ap50": {
            class_name: mean_valid(ap50[:, cat_ids.index(category_id)])
            for class_name, category_id in category_ids.items()
        },
    }


def mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    return float(valid.mean())


if __name__ == "__main__":
    app()
