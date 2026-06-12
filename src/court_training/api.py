import base64
import io
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from jaxtyping import Bool, Float
from PIL import Image
from sportanalytics import NbaCourt
from torch import Tensor

from court_training import homography
from court_training.constants import TTA_SCALES
from court_training.dataset import BASKETBALL_DETECTION_CLASSES
from court_training.detection import inference
from court_training.detection.model import CourtDetector
from court_training.segmentation.inference import image_to_tensor
from court_training.segmentation.model import CourtSegmenter

IMAGE_SIZE = (360, 480)
MASK_NAMES = tuple(NbaCourt.areas())
KEYPOINT_NAMES = tuple(NbaCourt.keypoints())
DETECTION_RESOLUTION = 704


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    app.state.segmenter = load_segmenter(Path(os.environ["COURT_SEGMENTATION_CHECKPOINT"]), device)
    app.state.detector = load_detector(Path(os.environ["COURT_DETECTION_CHECKPOINT"]), device)
    yield


app = FastAPI(title="Court Training API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "device": str(app.state.segmenter.device)}


@app.post("/predict")
async def predict(
    image: Annotated[UploadFile, File()],
    segmentation: Annotated[bool, Form()] = True,
    detection: Annotated[bool, Form()] = True,
    segmentation_threshold: Annotated[float, Form()] = 0.5,
    detection_threshold: Annotated[float, Form()] = 0.25,
    detection_hflip: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    frame = Image.open(io.BytesIO(await image.read())).convert("RGB")
    response: dict[str, object] = {}
    if segmentation:
        response["segmentation"] = predict_segmentation(app.state.segmenter, frame, segmentation_threshold)
    if detection:
        response["detections"] = predict_detections(app.state.detector, frame, detection_threshold, detection_hflip)
    return response


def predict_segmentation(model: CourtSegmenter, image: Image.Image, threshold: float) -> dict[str, object]:
    resized = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BILINEAR)
    prediction = model.predict(image_to_tensor(resized, model.device), TTA_SCALES)
    probabilities = prediction["masks"][0].sigmoid().cpu()
    keypoints = prediction["keypoints"][0].cpu().numpy()
    visibility = prediction["visibility"][0].sigmoid().cpu().numpy()

    return {
        "homography": fit_nba_homography(probabilities, keypoints, visibility),
        "masks": [
            {
                "name": name,
                "score": float(mask.mean()),
                "png": encode_mask_png(mask >= threshold),
            }
            for name, mask in zip(MASK_NAMES, probabilities.numpy(), strict=True)
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


def fit_nba_homography(
    probabilities: Float[Tensor, "N H W"],
    keypoints: Float[np.ndarray, "K 2"],
    visibility: Float[np.ndarray, "K"],
) -> dict[str, object] | None:
    visible = visibility >= 0.5
    if visible.sum() < 4:
        return None
    matrix, _, score = homography.fit_court(NbaCourt, MASK_NAMES, KEYPOINT_NAMES, probabilities, keypoints, visible)
    return {"court": "nba", "matrix": matrix.cpu().tolist(), "soft_iou": score}


def predict_detections(
    model: CourtDetector,
    image: Image.Image,
    threshold: float,
    hflip: bool,
) -> list[dict[str, object]]:
    predictions = inference.predict(model, image, hflip=hflip, threshold=threshold, nms_iou=0.6, max_detections=300)
    boxes = predictions["boxes"].tolist()
    scores = predictions["scores"].tolist()
    labels = predictions["labels"].tolist()
    return [
        {
            "label": model.class_names[label],
            "score": score,
            "box": {"x": box[0], "y": box[1], "width": box[2], "height": box[3]},
        }
        for box, score, label in zip(boxes, scores, labels, strict=True)
    ]


def encode_mask_png(mask: Bool[np.ndarray, "H W"]) -> str:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_segmenter(checkpoint: Path, device: torch.device) -> CourtSegmenter:
    model = CourtSegmenter(
        num_masks=len(MASK_NAMES),
        num_keypoints=len(KEYPOINT_NAMES),
        mask_names=MASK_NAMES,
        keypoint_names=KEYPOINT_NAMES,
        backbone="vit_large_patch16_dinov3",
        pretrained=False,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    return model


def load_detector(checkpoint: Path, device: torch.device) -> CourtDetector:
    model = CourtDetector(BASKETBALL_DETECTION_CLASSES, DETECTION_RESOLUTION, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    return model
