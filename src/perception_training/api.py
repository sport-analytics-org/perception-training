import io
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import sportkit as sk
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from jaxtyping import Bool, Float
from PIL import Image
from pydantic import BaseModel
from torch import Tensor

import perception_training as pt
from perception_training.detection.model import CourtDetector
from perception_training.segmentation.model import CourtSegmenter

CourtType = Literal["nba", "fiba"]
COURTS: dict[CourtType, sk.courts.BasketCourt] = {
    "nba": sk.courts.NbaCourt,
    "fiba": sk.courts.FibaCourt,
}


class Point(BaseModel):
    x: float
    y: float


class Polygon(BaseModel):
    label: str
    points: list[Point]


class Keypoint(BaseModel):
    position: tuple[float, float]
    visible: bool


class Homography(BaseModel):
    court: str
    matrix: list[list[float]]
    soft_iou: float


class Segmentation(BaseModel):
    polygons: list[Polygon]
    keypoints: list[Keypoint]
    homography: Homography | None


class DetectionCategory(BaseModel):
    id: int
    name: str


class DetectionBox(BaseModel):
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    source: str = "rfdetr"


class Detections(BaseModel):
    categories: list[DetectionCategory]
    boxes: list[DetectionBox]


class Prediction(BaseModel):
    width: int
    height: int
    segmentation: Segmentation | None = None
    detections: Detections | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    segmentation_checkpoint = Path(os.environ["COURT_SEGMENTATION_CHECKPOINT"])
    detection_checkpoint = Path(os.environ["COURT_DETECTION_CHECKPOINT"])
    app.state.segmenter = CourtSegmenter.load(segmentation_checkpoint, device)
    app.state.detector = CourtDetector.load(detection_checkpoint, device)
    yield


app = FastAPI(title="Court Training API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "device": str(app.state.segmenter.device)}


@app.post("/predict")
async def predict(
    image: Annotated[UploadFile, File()],
    segmentation: Annotated[bool, Form()] = True,
    detection: Annotated[bool, Form()] = True,
    court_type: Annotated[CourtType, Form()] = "nba",
    segmentation_threshold: Annotated[float, Form()] = 0.5,
    homography_iterations: Annotated[int, Form(gt=0)] = pt.homography.DEFAULT_MAX_ITERATIONS,
    detection_threshold: Annotated[float, Form()] = 0.25,
    detection_hflip: Annotated[bool, Form()] = False,
) -> Prediction:
    frame = Image.open(io.BytesIO(await image.read())).convert("RGB")
    prediction = Prediction(width=frame.width, height=frame.height)
    if segmentation:
        prediction.segmentation = predict_segmentation(
            app.state.segmenter,
            frame,
            court_type,
            segmentation_threshold,
            homography_iterations,
        )
    if detection:
        prediction.detections = predict_detections(app.state.detector, frame, detection_threshold, detection_hflip)
    return prediction


def predict_segmentation(
    model: CourtSegmenter,
    image: Image.Image,
    court_type: CourtType,
    threshold: float,
    homography_iterations: int,
) -> Segmentation:
    with torch.inference_mode():
        prediction = model.predict([image])
    probabilities = torch.from_numpy(prediction["masks"][0])
    keypoints = prediction["keypoints"][0]
    visibility = prediction["visibility"][0]

    backboard_indices = [index for index, label in enumerate(model.mask_names) if label.endswith("_backboard")]
    fitted_indices = [index for index, label in enumerate(model.mask_names) if not label.endswith("_backboard")]
    fitted_labels = tuple(model.mask_names[index] for index in fitted_indices)
    backboard_labels = tuple(model.mask_names[index] for index in backboard_indices)

    fitted_homography, fitted_masks = fit_homography(
        probabilities[fitted_indices],
        keypoints,
        visibility,
        fitted_labels,
        model.keypoint_names,
        court_type,
        homography_iterations,
    )

    fitted_polygons = []
    if fitted_masks is not None:
        fitted_mask_array = fitted_masks.numpy() >= threshold
        fitted_polygons = mask_polygons(fitted_mask_array, fitted_labels)
    backboard_masks = probabilities[backboard_indices].numpy() >= threshold
    backboard_polygons = mask_polygons(backboard_masks, backboard_labels)
    return Segmentation(
        polygons=[*fitted_polygons, *backboard_polygons],
        keypoints=[
            Keypoint(position=(float(x), float(y)), visible=bool(score >= threshold))
            for (x, y), score in zip(keypoints, visibility, strict=True)
        ],
        homography=fitted_homography,
    )


def mask_polygons(masks: Bool[np.ndarray, "N H W"], labels: tuple[str, ...]) -> list[Polygon]:
    polygons = []
    for label, mask in zip(labels, masks, strict=True):
        if mask.sum() < 3:
            continue
        traced = sk.polygons.trace_mask(mask)
        points = [Point(x=x, y=y) for x, y in traced.points]
        polygons.append(Polygon(label=label, points=points))
    return polygons


def fit_homography(
    probabilities: Float[Tensor, "N H W"],
    keypoints: Float[np.ndarray, "K 2"],
    visibility: Float[np.ndarray, "K"],
    mask_names: tuple[str, ...],
    keypoint_names: tuple[str, ...],
    court_type: CourtType,
    max_iterations: int,
) -> tuple[Homography | None, Float[Tensor, "N H W"] | None]:
    visible = visibility >= 0.5
    if visible.sum() < 4:
        return None, None
    matrix, fitted_masks, score = pt.homography.fit_court(
        COURTS[court_type],
        mask_names,
        keypoint_names,
        probabilities,
        keypoints,
        visible,
        max_iterations,
    )
    fitted_homography = Homography(court=court_type, matrix=matrix.cpu().tolist(), soft_iou=score)
    return fitted_homography, fitted_masks.cpu()


def predict_detections(
    model: CourtDetector,
    image: Image.Image,
    threshold: float,
    hflip: bool,
) -> Detections:
    predictions = model.predict([image], hflip=hflip, threshold=threshold, nms_iou=0.6, max_detections=300)[0]
    width, height = image.size
    boxes = [
        DetectionBox(
            category_id=label,
            bbox_xyxy=(x1 * width, y1 * height, x2 * width, y2 * height),
            score=score,
        )
        for (x1, y1, x2, y2), score, label in zip(
            predictions["boxes"].tolist(),
            predictions["scores"].tolist(),
            predictions["labels"].tolist(),
            strict=True,
        )
    ]
    categories = [DetectionCategory(id=index, name=name) for index, name in enumerate(model.class_names)]
    return Detections(categories=categories, boxes=boxes)
