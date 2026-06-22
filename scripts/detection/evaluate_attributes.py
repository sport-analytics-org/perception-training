import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import typer
from PIL import Image
from tqdm import tqdm

from perception_training.dataset import BASKETBALL_ATTRIBUTE_BASE_CLASSES, BASKETBALL_DETECTION_ATTRIBUTES, CourtDataset
from perception_training.detection.model import CourtDetector

app = typer.Typer(help="Evaluate basketball detection attributes at IoU 0.50.")

CHECKPOINT_ARGUMENT = typer.Argument(help="Trained CourtDetector state dict.")
DATA_ROOT_ARGUMENT = typer.Argument(help="Flat exported dataset root with attribute annotations.")
OUTPUT_JSON_ARGUMENT = typer.Argument(help="Where to write the attribute metrics report.")

SPECIAL_CLASS_ATTRIBUTES = {
    "ball-in-basket": "in_basket",
    "player-in-possession": "in_possession",
    "player-jump-shot": "jump_shot",
    "player-layup-dunk": "layup_dunk",
    "player-shot-block": "shot_block",
}


@dataclass(frozen=True)
class AttributePrediction:
    image_index: int
    attribute: str
    score: float
    box_xywh: np.ndarray


@app.command()
def main(
    checkpoint: Path = CHECKPOINT_ARGUMENT,
    data_root: Path = DATA_ROOT_ARGUMENT,
    output_json: Path = OUTPUT_JSON_ARGUMENT,
    batch_size: int = typer.Option(8, help="Images per prediction batch."),
    threshold: float = typer.Option(0.001, help="Base detection threshold before attribute scoring."),
    nms_iou: float = typer.Option(0.9, help="Per-class NMS IoU threshold."),
    max_detections: int = typer.Option(300, help="Maximum detections per image after NMS."),
    fixed_score_threshold: float = typer.Option(0.4, help="Score threshold for one fixed precision/recall readout."),
    iou_threshold: float = typer.Option(0.5, help="IoU threshold for attribute true positives."),
    attribute_score: str = typer.Option("product", help="For attribute-head models: product or probability."),
) -> None:
    if attribute_score not in {"product", "probability"}:
        raise typer.BadParameter("attribute_score must be 'product' or 'probability'")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = CourtDetector.load(checkpoint.expanduser().resolve(), device)
    data = CourtDataset(
        data_root.expanduser().resolve(),
        model.image_size,
        load_bbox=True,
        class_names=model.class_names,
        attribute_names=BASKETBALL_DETECTION_ATTRIBUTES,
    )

    ground_truth = collect_ground_truth(data)
    predictions = collect_predictions(
        model,
        data,
        batch_size=batch_size,
        threshold=threshold,
        nms_iou=nms_iou,
        max_detections=max_detections,
        attribute_score=attribute_score,
    )
    report = summarize_attributes(
        predictions,
        ground_truth,
        iou_threshold=iou_threshold,
        fixed_score_threshold=fixed_score_threshold,
    )
    report = {
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "data_root": str(data_root.expanduser().resolve()),
        "mode": "attribute_head" if model.attribute_names else "special_classes",
        "attribute_score": attribute_score if model.attribute_names else "class_score",
        "base_detection_threshold": threshold,
        "nms_iou": nms_iou,
        "max_detections": max_detections,
        "iou_threshold": iou_threshold,
        "fixed_score_threshold": fixed_score_threshold,
        **report,
    }

    output_json = output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    typer.echo(json.dumps(report, indent=2))


def collect_ground_truth(data: CourtDataset) -> dict[str, dict[int, np.ndarray]]:
    ground_truth = {attribute: {} for attribute in BASKETBALL_DETECTION_ATTRIBUTES}
    for image_index, image_path in enumerate(data.image_paths):
        boxes_xywh, _labels, attributes = data.boxes[image_path]
        for attribute_index, attribute in enumerate(BASKETBALL_DETECTION_ATTRIBUTES):
            ground_truth[attribute][image_index] = boxes_xywh[attributes[:, attribute_index]].astype(np.float32)
    return ground_truth


def collect_predictions(
    model: CourtDetector,
    data: CourtDataset,
    batch_size: int,
    threshold: float,
    nms_iou: float,
    max_detections: int,
    attribute_score: str,
) -> list[AttributePrediction]:
    predictions = []
    indexed_paths = list(enumerate(data.image_paths))
    for batch in tqdm(list(chunks(indexed_paths, batch_size)), desc="Predicting attributes"):
        image_indexes = [image_index for image_index, _path in batch]
        images = []
        for _image_index, image_path in batch:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        batch_predictions = model.predict(
            images,
            threshold=threshold,
            nms_iou=nms_iou,
            max_detections=max_detections,
        )
        for image_index, prediction in zip(image_indexes, batch_predictions, strict=True):
            if model.attribute_names:
                predictions.extend(attribute_head_predictions(model, image_index, prediction, attribute_score))
            else:
                predictions.extend(special_class_predictions(model, image_index, prediction))
    return predictions


def chunks[T](items: list[T], size: int) -> Iterable[list[T]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def attribute_head_predictions(
    model: CourtDetector,
    image_index: int,
    prediction: dict,
    attribute_score: str,
) -> list[AttributePrediction]:
    outputs = []
    boxes_xywh = xyxy_to_xywh(prediction["boxes"])
    for detection_index, label_index in enumerate(prediction["labels"]):
        class_name = model.class_names[int(label_index)]
        for attribute_index, attribute in enumerate(model.attribute_names):
            if BASKETBALL_ATTRIBUTE_BASE_CLASSES[attribute] != class_name:
                continue
            probability = float(prediction["attributes"][detection_index, attribute_index])
            score = probability
            if attribute_score == "product":
                score *= float(prediction["scores"][detection_index])
            outputs.append(AttributePrediction(image_index, attribute, score, boxes_xywh[detection_index]))
    return outputs


def special_class_predictions(model: CourtDetector, image_index: int, prediction: dict) -> list[AttributePrediction]:
    outputs = []
    boxes_xywh = xyxy_to_xywh(prediction["boxes"])
    for detection_index, label_index in enumerate(prediction["labels"]):
        class_name = model.class_names[int(label_index)]
        attribute = SPECIAL_CLASS_ATTRIBUTES.get(class_name)
        if attribute is None:
            continue
        outputs.append(
            AttributePrediction(
                image_index,
                attribute,
                float(prediction["scores"][detection_index]),
                boxes_xywh[detection_index],
            )
        )
    return outputs


def summarize_attributes(
    predictions: list[AttributePrediction],
    ground_truth: dict[str, dict[int, np.ndarray]],
    iou_threshold: float,
    fixed_score_threshold: float,
) -> dict:
    per_attribute = {}
    aps = []
    for attribute in BASKETBALL_DETECTION_ATTRIBUTES:
        attribute_predictions = [prediction for prediction in predictions if prediction.attribute == attribute]
        total_gt = sum(len(boxes) for boxes in ground_truth[attribute].values())
        result = evaluate_attribute(attribute_predictions, ground_truth[attribute], total_gt, iou_threshold)
        fixed_counts = match_at_threshold(
            attribute_predictions,
            ground_truth[attribute],
            iou_threshold=iou_threshold,
            score_threshold=fixed_score_threshold,
        )
        result["fixed_threshold"] = with_rates(fixed_counts)
        per_attribute[attribute] = result
        if total_gt:
            aps.append(result["ap"])

    return {
        "mean_attribute_ap": float(np.mean(aps)) if aps else None,
        "per_attribute": per_attribute,
    }


def evaluate_attribute(
    predictions: list[AttributePrediction],
    ground_truth: dict[int, np.ndarray],
    total_gt: int,
    iou_threshold: float,
) -> dict:
    predictions = sorted(predictions, key=lambda prediction: prediction.score, reverse=True)
    matched = {image_index: np.zeros(len(boxes), dtype=np.bool_) for image_index, boxes in ground_truth.items()}
    tp = np.zeros(len(predictions), dtype=np.float32)
    fp = np.zeros(len(predictions), dtype=np.float32)
    for prediction_index, prediction in enumerate(predictions):
        gt_boxes = ground_truth[prediction.image_index]
        if len(gt_boxes) == 0:
            fp[prediction_index] = 1
            continue
        ious = pairwise_iou_xywh(gt_boxes, prediction.box_xywh[None, :])[:, 0]
        gt_index = int(np.argmax(ious))
        if ious[gt_index] >= iou_threshold and not matched[prediction.image_index][gt_index]:
            tp[prediction_index] = 1
            matched[prediction.image_index][gt_index] = True
        else:
            fp[prediction_index] = 1

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    precision = np.divide(
        cumulative_tp,
        cumulative_tp + cumulative_fp,
        out=np.zeros_like(cumulative_tp),
        where=(cumulative_tp + cumulative_fp) > 0,
    )
    recall = cumulative_tp / total_gt if total_gt else np.zeros_like(cumulative_tp)
    best = best_f1(predictions, cumulative_tp, cumulative_fp, total_gt)
    return {
        "gt": total_gt,
        "predictions": len(predictions),
        "ap": average_precision(precision, recall),
        "best_f1": best,
    }


def best_f1(
    predictions: list[AttributePrediction],
    cumulative_tp: np.ndarray,
    cumulative_fp: np.ndarray,
    total_gt: int,
) -> dict:
    if not predictions or total_gt == 0:
        return {"threshold": None, "tp": 0, "fp": 0, "fn": total_gt, "precision": None, "recall": None, "f1": None}
    fn = total_gt - cumulative_tp
    precision = np.divide(
        cumulative_tp,
        cumulative_tp + cumulative_fp,
        out=np.zeros_like(cumulative_tp),
        where=(cumulative_tp + cumulative_fp) > 0,
    )
    recall = cumulative_tp / total_gt
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    index = int(np.argmax(f1))
    return {
        "threshold": float(predictions[index].score),
        "tp": int(cumulative_tp[index]),
        "fp": int(cumulative_fp[index]),
        "fn": int(fn[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
    }


def match_at_threshold(
    predictions: list[AttributePrediction],
    ground_truth: dict[int, np.ndarray],
    iou_threshold: float,
    score_threshold: float,
) -> dict[str, int]:
    filtered = [prediction for prediction in predictions if prediction.score >= score_threshold]
    filtered = sorted(filtered, key=lambda prediction: prediction.score, reverse=True)
    matched = {image_index: np.zeros(len(boxes), dtype=np.bool_) for image_index, boxes in ground_truth.items()}
    tp = 0
    fp = 0
    for prediction in filtered:
        gt_boxes = ground_truth[prediction.image_index]
        if len(gt_boxes) == 0:
            fp += 1
            continue
        ious = pairwise_iou_xywh(gt_boxes, prediction.box_xywh[None, :])[:, 0]
        gt_index = int(np.argmax(ious))
        if ious[gt_index] >= iou_threshold and not matched[prediction.image_index][gt_index]:
            tp += 1
            matched[prediction.image_index][gt_index] = True
        else:
            fp += 1
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    return {"tp": tp, "fp": fp, "fn": total_gt - tp}


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float | None:
    if len(precision) == 0:
        return None
    padded_precision = np.concatenate([[0.0], precision, [0.0]])
    padded_recall = np.concatenate([[0.0], recall, [1.0]])
    for index in range(len(padded_precision) - 1, 0, -1):
        padded_precision[index - 1] = max(padded_precision[index - 1], padded_precision[index])
    indexes = np.flatnonzero(padded_recall[1:] != padded_recall[:-1])
    return float(np.sum((padded_recall[indexes + 1] - padded_recall[indexes]) * padded_precision[indexes + 1]))


def with_rates(counts: dict[str, int]) -> dict[str, float | int | None]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def pairwise_iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_xyxy = xywh_to_xyxy(a)
    b_xyxy = xywh_to_xyxy(b)
    top_left = np.maximum(a_xyxy[:, None, :2], b_xyxy[None, :, :2])
    bottom_right = np.minimum(a_xyxy[:, None, 2:], b_xyxy[None, :, 2:])
    wh = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    a_area = np.clip(a[:, 2], 0.0, None) * np.clip(a[:, 3], 0.0, None)
    b_area = np.clip(b[:, 2], 0.0, None) * np.clip(b[:, 3], 0.0, None)
    union = a_area[:, None] + b_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    output = boxes.copy()
    output[:, 2] -= output[:, 0]
    output[:, 3] -= output[:, 1]
    return output


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = boxes.copy()
    output[:, 2] += output[:, 0]
    output[:, 3] += output[:, 1]
    return output


if __name__ == "__main__":
    app()
