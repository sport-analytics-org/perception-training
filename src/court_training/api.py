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
from court_training.constants import IMAGE_MEAN, IMAGE_STD, TTA_SCALES
from court_training.dataset import BASKETBALL_DETECTION_CLASSES
from court_training.detection import inference
from court_training.detection.model import CourtDetector
from court_training.segmentation.model import CourtSegmenter
from court_training.warp import warp

IMAGE_SIZE = (360, 480)
MASK_NAMES = tuple(NbaCourt.areas())
KEYPOINT_NAMES = tuple(NbaCourt.keypoints())
DETECTION_RESOLUTION = 704


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    device = prediction_device()
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

    source_masks = homography.template_masks(NbaCourt, MASK_NAMES, probabilities.shape[-1], probabilities.device)
    source_keypoints = homography.normalized_keypoints(NbaCourt, KEYPOINT_NAMES)
    initial = torch.tensor(
        homography.find_keypoints_homography(source_keypoints[visible], keypoints[visible]),
        dtype=probabilities.dtype,
    )
    multipliers = torch.tensor(
        [1.5 if "3pt_area" in name or "painted_area" in name else 1.0 for name in MASK_NAMES],
        device=probabilities.device,
    )
    matrix = homography.fit_homography(source_masks, probabilities, initial, multipliers)
    fitted = warp(source_masks, matrix, probabilities.shape[-2:])
    return {"court": "nba", "matrix": matrix.cpu().tolist(), "soft_iou": homography.soft_iou(fitted, probabilities)}


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


def image_to_tensor(image: Image.Image, device: torch.device) -> Float[Tensor, "1 3 H W"]:
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).to(device)
    return ((tensor - IMAGE_MEAN.to(device)) / IMAGE_STD.to(device))[None]


def encode_mask_png(mask: Bool[np.ndarray, "H W"]) -> str:
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
