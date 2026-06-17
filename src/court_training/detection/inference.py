import torch
from PIL import Image
from torch import Tensor
from torchvision.ops import batched_nms, box_convert

from court_training.detection.model import CourtDetector


@torch.inference_mode()
def predict(
    model: CourtDetector,
    image: Image.Image,
    hflip: bool,
    threshold: float,
    nms_iou: float,
    max_detections: int,
) -> dict[str, Tensor]:
    """Thresholded, NMS-merged detections with normalized xywh boxes, optionally pooling hflip TTA."""
    detections = model.predict([image], hflip=hflip)[0]
    boxes_xyxy = torch.from_numpy(detections.xyxy).to(dtype=torch.float32)
    boxes = box_convert(boxes_xyxy, "xyxy", "xywh")
    scores = torch.from_numpy(detections.confidence).to(dtype=torch.float32)
    labels = torch.from_numpy(detections.class_id).to(dtype=torch.long)
    confident = scores >= threshold
    boxes, scores, labels = boxes[confident], scores[confident], labels[confident]
    boxes_xyxy = box_convert(boxes, "xywh", "xyxy")
    keep = batched_nms(boxes_xyxy, scores, labels, nms_iou)[:max_detections]
    return {"boxes": boxes[keep].cpu(), "scores": scores[keep].cpu(), "labels": labels[keep].cpu()}
