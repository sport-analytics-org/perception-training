from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchvision.ops import batched_nms, box_convert

from perception_training import flip

if TYPE_CHECKING:
    from perception_training.detection.model import CourtDetector


class Prediction(TypedDict):
    boxes: Float[np.ndarray, "N 4"]
    scores: Float[np.ndarray, "N"]
    labels: Int[np.ndarray, "N"]


def predict(
    model: "CourtDetector",
    images: Float[Tensor, "B 3 H W"],
    image_indexes: list[int],
    flipped: list[bool],
    image_count: int,
    threshold: float = 0.0,
    nms_iou: float | None = None,
    max_detections: int | None = None,
) -> list[Prediction]:
    """Per-image detections with normalized xyxy boxes as numpy arrays."""
    outputs = model.model(images)
    unit_sizes = torch.ones((len(images), 2), device=images.device)
    results = model.postprocess(outputs, unit_sizes)
    predictions_by_image = [[] for _ in range(image_count)]
    for result, image_index, is_flipped in zip(results, image_indexes, flipped, strict=True):
        keep = result["labels"] < len(model.class_names)
        prediction = {key: value[keep] for key, value in result.items()}
        prediction["boxes"] = box_convert(prediction["boxes"], "xyxy", "xywh")
        if is_flipped:
            prediction["boxes"] = flip.flip_torch(boxes_xywh=prediction["boxes"])["boxes_xywh"]
        predictions_by_image[image_index].append(prediction)

    outputs = []
    for predictions in predictions_by_image:
        prediction = merge(predictions)
        if threshold > 0:
            keep = prediction["scores"] >= threshold
            prediction = {key: value[keep] for key, value in prediction.items()}
        if nms_iou is not None:
            boxes_xyxy = box_convert(prediction["boxes"], "xywh", "xyxy")
            keep = batched_nms(boxes_xyxy, prediction["scores"], prediction["labels"], nms_iou)
            if max_detections is not None:
                keep = keep[:max_detections]
            prediction = {key: value[keep] for key, value in prediction.items()}
        outputs.append(
            {
                "boxes": box_convert(prediction["boxes"], "xywh", "xyxy").cpu().numpy().astype(np.float32),
                "scores": prediction["scores"].cpu().numpy().astype(np.float32),
                "labels": prediction["labels"].cpu().numpy(),
            }
        )
    return outputs


def merge(predictions: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    if not predictions:
        empty_float = torch.empty((0,), dtype=torch.float32)
        empty_long = torch.empty((0,), dtype=torch.long)
        return {"boxes": empty_float.reshape(0, 4), "scores": empty_float, "labels": empty_long}
    return {
        "boxes": torch.cat([prediction["boxes"] for prediction in predictions]),
        "scores": torch.cat([prediction["scores"] for prediction in predictions]),
        "labels": torch.cat([prediction["labels"] for prediction in predictions]),
    }
