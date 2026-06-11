import json
from pathlib import Path

import numpy as np
import typer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETR
from tqdm import tqdm

from court_training.detection import data, inference

app = typer.Typer(help="Evaluate an RF-DETR checkpoint on basketball detections.")


@app.command()
def main(
    checkpoint: Path,
    val_root: Path,
    output_json: Path,
    classes: str = typer.Option("ball,player,referee", help="Comma-separated class names to evaluate."),
    resolution: int = typer.Option(704, help="Square inference resolution."),
    hflip: bool = typer.Option(False, "--hflip/--no-hflip", help="Use horizontal-flip TTA."),
    threshold: float = typer.Option(0.001, help="Confidence threshold before NMS and COCO evaluation."),
    nms_iou: float = typer.Option(0.6, help="Per-class NMS IoU threshold after TTA merge."),
    max_detections: int = typer.Option(300, help="Maximum detections per image after NMS."),
) -> None:
    class_names = data.parse_classes(classes)
    category_ids = data.category_ids(class_names)
    samples = data.load_split(val_root.expanduser().resolve())
    model = RFDETR.from_checkpoint(checkpoint.expanduser().resolve())

    output_json.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_path = output_json.with_suffix(".ground_truth.coco.json")
    ground_truth = data.coco_annotations(samples, class_names, box_scales={})
    ground_truth_path.write_text(json.dumps(ground_truth) + "\n")

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
            {
                "image_id": image_id,
                "category_id": prediction.category_id,
                "bbox": prediction.bbox,
                "score": prediction.score,
            }
            for prediction in image_predictions
        )

    output_json.write_text(json.dumps(predictions) + "\n")
    metrics = evaluate(ground_truth_path, output_json, category_ids)
    output_json.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    typer.echo(json.dumps(metrics, indent=2))


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
