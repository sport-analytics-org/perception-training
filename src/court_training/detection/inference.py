import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torchvision.ops import batched_nms, box_convert

from court_training import flip
from court_training.constants import IMAGE_MEAN, IMAGE_STD
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
    variants = [image]
    if hflip:
        variants.append(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))

    images = torch.stack([image_to_tensor(variant, model.resolution) for variant in variants]).to(model.device)
    results = model.predict(images)
    if hflip:
        results[1]["boxes"] = flip.flip_torch(boxes_xywh=results[1]["boxes"])["boxes_xywh"]

    boxes = torch.cat([result["boxes"] for result in results])
    scores = torch.cat([result["scores"] for result in results])
    labels = torch.cat([result["labels"] for result in results])
    confident = scores >= threshold
    boxes, scores, labels = boxes[confident], scores[confident], labels[confident]
    boxes_xyxy = box_convert(boxes, "xywh", "xyxy")
    keep = batched_nms(boxes_xyxy, scores, labels, nms_iou)[:max_detections]
    return {"boxes": boxes[keep].cpu(), "scores": scores[keep].cpu(), "labels": labels[keep].cpu()}


def image_to_tensor(image: Image.Image, resolution: int) -> Float[Tensor, "3 H W"]:
    resized = image.resize((resolution, resolution), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (tensor - IMAGE_MEAN) / IMAGE_STD
