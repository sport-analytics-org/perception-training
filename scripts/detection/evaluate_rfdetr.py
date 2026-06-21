import json
from pathlib import Path

import torch
import typer
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_convert
from tqdm import tqdm

from perception_training.dataset import BASKETBALL_DETECTION_CLASSES, CourtDataset
from perception_training.detection import metrics
from perception_training.detection.model import CourtDetector

app = typer.Typer(help="Evaluate an RF-DETR checkpoint on basketball detections.")

CHECKPOINT_ARGUMENT = typer.Argument(help="Trained CourtDetector state dict.")
VAL_ROOT_ARGUMENT = typer.Argument(help="Flat exported validation dataset root.")
OUTPUT_JSON_ARGUMENT = typer.Argument(help="Where the metrics report is written.")


@app.command()
def main(
    checkpoint: Path = CHECKPOINT_ARGUMENT,
    val_root: Path = VAL_ROOT_ARGUMENT,
    output_json: Path = OUTPUT_JSON_ARGUMENT,
    hflip: bool = typer.Option(False, "--hflip/--no-hflip", help="Use horizontal-flip TTA."),
    threshold: float = typer.Option(0.001, help="Confidence threshold before NMS and mAP evaluation."),
    nms_iou: float = typer.Option(0.6, help="Per-class NMS IoU threshold after TTA merge."),
    max_detections: int = typer.Option(300, help="Maximum detections per image after NMS."),
) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = CourtDetector.load(checkpoint.expanduser().resolve(), device)
    val_data = CourtDataset(val_root.expanduser().resolve(), model.image_size, load_bbox=True)

    metric = MeanAveragePrecision(box_format="xywh", class_metrics=True)
    for image_path in tqdm(val_data.image_paths, desc="Evaluating"):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        prediction = model.predict(
            [image],
            hflip=hflip,
            threshold=threshold,
            nms_iou=nms_iou,
            max_detections=max_detections,
        )[0]
        boxes = torch.from_numpy(prediction["boxes"]).to(dtype=torch.float32)
        prediction = {
            "boxes": box_convert(boxes, "xyxy", "xywh"),
            "scores": torch.from_numpy(prediction["scores"]).to(dtype=torch.float32),
            "labels": torch.from_numpy(prediction["labels"]).to(dtype=torch.long),
        }
        boxes_xywh, labels = val_data.boxes[image_path]
        ground_truth = {"boxes": torch.from_numpy(boxes_xywh), "labels": torch.from_numpy(labels)}
        metric.update([prediction], [ground_truth])

    results = metrics.summarize(metric, BASKETBALL_DETECTION_CLASSES)
    output_json = output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2) + "\n")
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
