import base64
import io
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from sportanalytics import NbaCourt

from court_training.constants import IMAGE_MEAN, IMAGE_STD, TTA_SCALES
from court_training.dataset import BASKETBALL_DETECTION_CLASSES
from court_training.detection import inference as detection_inference
from court_training.detection.model import CourtDetector
from court_training.segmentation.model import CourtSegmenter

IMAGE_SIZE = (360, 480)
MASK_NAMES = tuple(NbaCourt.areas())
KEYPOINT_NAMES = tuple(NbaCourt.keypoints())
DETECTION_RESOLUTION = 704


@dataclass
class Models:
    segmenter: CourtSegmenter | None
    detector: CourtDetector | None
    device: torch.device


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.models = load_models()
    yield
    app.state.models = Models(segmenter=None, detector=None, device=torch.device("cpu"))


app = FastAPI(title="Court Training API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    models: Models = app.state.models
    return {
        "ok": True,
        "device": str(models.device),
        "segmentation": models.segmenter is not None,
        "detection": models.detector is not None,
    }


@app.post("/predict")
async def predict(
    image: Annotated[UploadFile, File()],
    segmentation: Annotated[bool, Form()] = True,
    detection: Annotated[bool, Form()] = True,
    segmentation_threshold: Annotated[float, Form()] = 0.5,
    detection_threshold: Annotated[float, Form()] = 0.25,
    detection_hflip: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    models: Models = app.state.models
    frame = read_image(await image.read())
    response: dict[str, object] = {}

    if segmentation:
        if models.segmenter is None:
            raise HTTPException(status_code=503, detail="Segmentation model is not loaded")
        response["segmentation"] = predict_segmentation(models.segmenter, frame, models.device, segmentation_threshold)

    if detection:
        if models.detector is None:
            raise HTTPException(status_code=503, detail="Detection model is not loaded")
        response["detections"] = predict_detections(models.detector, frame, detection_threshold, detection_hflip)

    return response


def load_models() -> Models:
    device = prediction_device()
    segmentation_checkpoint = os.getenv("COURT_SEGMENTATION_CHECKPOINT")
    detection_checkpoint = os.getenv("COURT_DETECTION_CHECKPOINT")
    segmenter = load_segmenter(Path(segmentation_checkpoint), device) if segmentation_checkpoint else None
    detector = None
    if detection_checkpoint:
        detector = load_detector(Path(detection_checkpoint), device)
    return Models(segmenter=segmenter, detector=detector, device=device)


def load_segmenter(checkpoint: Path, device: torch.device) -> CourtSegmenter:
    model = CourtSegmenter(
        num_masks=len(MASK_NAMES),
        num_keypoints=len(KEYPOINT_NAMES),
        mask_names=MASK_NAMES,
        keypoint_names=KEYPOINT_NAMES,
        backbone="vit_large_patch16_dinov3",
        pretrained=False,
    )
    model.load_state_dict(torch.load(checkpoint.expanduser().resolve(), map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    return model


def load_detector(checkpoint: Path, device: torch.device) -> CourtDetector:
    model = CourtDetector(BASKETBALL_DETECTION_CLASSES, DETECTION_RESOLUTION, pretrained=False)
    model.load_state_dict(torch.load(checkpoint.expanduser().resolve(), map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    return model


def predict_segmentation(
    model: CourtSegmenter,
    image: Image.Image,
    device: torch.device,
    threshold: float,
) -> dict[str, object]:
    resized = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BILINEAR)
    tensor = image_to_tensor(resized, device)
    with torch.inference_mode():
        prediction = model.predict(tensor, TTA_SCALES)

    probabilities = prediction["masks"][0].sigmoid().cpu().numpy()
    keypoints = prediction["keypoints"][0].cpu().numpy()
    visibility = prediction["visibility"][0].sigmoid().cpu().numpy()

    return {
        "width": IMAGE_SIZE[1],
        "height": IMAGE_SIZE[0],
        "masks": [
            {
                "name": name,
                "score": float(mask.mean()),
                "png": encode_mask_png(mask >= threshold),
            }
            for name, mask in zip(MASK_NAMES, probabilities, strict=True)
        ],
        "keypoints": [
            {
                "name": name,
                "x": float(point[0]),
                "y": float(point[1]),
                "visible": bool(score >= threshold),
                "score": float(score),
            }
            for name, point, score in zip(KEYPOINT_NAMES, keypoints, visibility, strict=True)
        ],
    }


def predict_detections(
    model: CourtDetector,
    image: Image.Image,
    threshold: float,
    hflip: bool,
) -> list[dict[str, object]]:
    predictions = detection_inference.predict(
        model,
        image,
        hflip=hflip,
        threshold=threshold,
        nms_iou=0.6,
        max_detections=300,
    )
    boxes = predictions["boxes"].tolist()
    scores = predictions["scores"].tolist()
    labels = predictions["labels"].tolist()
    return [
        {
            "label": model.class_names[label],
            "score": score,
            "box": {
                "x": box[0],
                "y": box[1],
                "width": box[2],
                "height": box[3],
            },
        }
        for box, score, label in zip(boxes, scores, labels, strict=True)
    ]


def read_image(contents: bytes) -> Image.Image:
    return Image.open(io.BytesIO(contents)).convert("RGB")


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).to(device)
    mean = IMAGE_MEAN.to(device)
    std = IMAGE_STD.to(device)
    return ((tensor - mean) / std)[None]


def encode_mask_png(mask: np.ndarray) -> str:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def prediction_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
