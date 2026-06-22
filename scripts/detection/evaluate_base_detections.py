import json
from pathlib import Path

import torch
import typer
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_convert
from tqdm import tqdm

import perception_training.detection as detection
from perception_training.dataset import BASKETBALL_DETECTION_CLASSES, CourtDataset
from perception_training.detection.model import CourtDetector

app = typer.Typer(help="Evaluate any basketball detector after mapping special labels back to base classes.")

CHECKPOINT_ARGUMENT = typer.Argument(help="Trained CourtDetector state dict.")
DATA_ROOT_ARGUMENT = typer.Argument(help="Flat exported base/attribute dataset root.")
OUTPUT_JSON_ARGUMENT = typer.Argument(help="Where the metrics report is written.")

SPECIAL_TO_BASE = {
    "ball-in-basket": "ball",
    "player-in-possession": "player",
    "player-jump-shot": "player",
    "player-layup-dunk": "player",
    "player-shot-block": "player",
}


@app.command()
def main(
    checkpoint: Path = CHECKPOINT_ARGUMENT,
    data_root: Path = DATA_ROOT_ARGUMENT,
    output_json: Path = OUTPUT_JSON_ARGUMENT,
    threshold: float = typer.Option(0.001, help="Confidence threshold before NMS and mAP evaluation."),
    nms_iou: float = typer.Option(0.6, help="Per-class NMS IoU threshold after TTA merge."),
    max_detections: int = typer.Option(300, help="Maximum detections per image after NMS."),
) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = CourtDetector.load(checkpoint.expanduser().resolve(), device)
    data = CourtDataset(
        data_root.expanduser().resolve(),
        model.image_size,
        load_bbox=True,
        class_names=BASKETBALL_DETECTION_CLASSES,
    )

    metric = MeanAveragePrecision(box_format="xywh", class_metrics=True)
    for image_path in tqdm(data.image_paths, desc="Evaluating base boxes"):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        prediction = model.predict(
            [image],
            threshold=threshold,
            nms_iou=nms_iou,
            max_detections=max_detections,
        )[0]
        predicted_labels = [
            BASKETBALL_DETECTION_CLASSES.index(base_class_name(model.class_names[int(label)]))
            for label in prediction["labels"]
        ]
        boxes = torch.from_numpy(prediction["boxes"]).to(dtype=torch.float32)
        prediction_for_metric = {
            "boxes": box_convert(boxes, "xyxy", "xywh"),
            "scores": torch.from_numpy(prediction["scores"]).to(dtype=torch.float32),
            "labels": torch.tensor(predicted_labels, dtype=torch.long),
        }
        boxes_xywh, labels, _attributes = data.boxes[image_path]
        ground_truth = {"boxes": torch.from_numpy(boxes_xywh), "labels": torch.from_numpy(labels)}
        metric.update([prediction_for_metric], [ground_truth])

    results = detection.metrics.summarize(metric, BASKETBALL_DETECTION_CLASSES)
    output_json = output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2) + "\n")
    typer.echo(json.dumps(results, indent=2))


def base_class_name(class_name: str) -> str:
    return SPECIAL_TO_BASE.get(class_name, class_name)


if __name__ == "__main__":
    app()
