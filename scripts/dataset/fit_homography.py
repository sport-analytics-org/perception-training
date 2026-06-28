import json
from pathlib import Path

import numpy as np
import sportkit as sk
import torch
import typer
from jaxtyping import Float
from torch import Tensor

from perception_training.homography import centered_homography, fit_homography
from perception_training.warp import warp

app = typer.Typer(help="Fit a basketball court homography from polygon mask JSON.")
MASK_ARGUMENT = typer.Argument(help="Polygon mask JSON.")
COURT_OPTION = typer.Option("nba", help="Court template to fit: nba or fiba.")
RASTER_SIZE = (1280, 720)

COURTS = {
    "nba": sk.courts.NbaCourt,
    "fiba": sk.courts.FibaCourt,
}


@app.command()
def main(
    mask: Path = MASK_ARGUMENT,
    court: str = COURT_OPTION,
) -> None:
    mask_path = mask.expanduser().resolve()

    court_template = COURTS[court]
    labels, target_masks = load_masks(mask_path, court_template, RASTER_SIZE)
    source_masks = load_template_masks(court_template, labels, target_masks.shape[-1])

    mask_multipliers = [1.5 if "3pt_area" in label or "painted_area" in label else 1.0 for label in labels]
    multipliers = torch.tensor(mask_multipliers)
    initial = torch.tensor(centered_homography(), dtype=target_masks.dtype)
    homography = fit_homography(source_masks, target_masks, initial, multipliers)
    homography = homography.numpy()

    initial_metrics = iou_metrics(source_masks, target_masks, initial.numpy())
    final_metrics = iou_metrics(source_masks, target_masks, homography)

    result = {
        "mask": str(mask_path),
        "court": court,
        "labels": labels,
        "initial_iou": initial_metrics["macro_iou"],
        "initial_area_weighted_iou": initial_metrics["area_weighted_iou"],
        "final_iou": final_metrics["macro_iou"],
        "final_area_weighted_iou": final_metrics["area_weighted_iou"],
        "homography": homography.tolist(),
    }
    print(json.dumps(result, indent=2))


def load_masks(
    mask_path: Path,
    court: sk.courts.BasketCourt,
    size: tuple[int, int],
) -> tuple[tuple[str, ...], Float[Tensor, "N H W"]]:
    mask_names = tuple(court.planar_areas())
    data = json.loads(mask_path.read_text())
    polygons = {label: sk.polygons.Polygon.from_json(points) for label, points in data.items()}
    masks = np.stack([polygons[label].rasterize(size) for label in mask_names])
    return mask_names, torch.tensor(masks.astype(np.float32))


def load_template_masks(
    court_template: sk.courts.BasketCourt,
    labels: tuple[str, ...],
    width: int,
) -> Float[Tensor, "N H W"]:
    masks = []
    for label in labels:
        image = court_template.get_mask_image(label, width).convert("L")
        mask = np.asarray(image, dtype=np.float32) / 255
        masks.append(torch.tensor(mask))
    return torch.stack(masks)


def iou_metrics(
    source_masks: Float[Tensor, "N H W"],
    target_masks: Float[Tensor, "N H W"],
    homography: Float[np.ndarray, "3 3"],
) -> dict[str, float]:
    output_shape = (target_masks.shape[-2], target_masks.shape[-1])
    homography_tensor = torch.tensor(homography, dtype=source_masks.dtype, device=source_masks.device)
    predictions = warp(source_masks, homography_tensor, output_shape)
    scores = []
    areas = []
    for prediction_tensor, target_tensor in zip(predictions, target_masks, strict=True):
        target = target_tensor.numpy() > 0.5
        prediction = prediction_tensor.numpy() > 0.5
        union = np.logical_or(target, prediction).sum()
        scores.append(float(np.logical_and(target, prediction).sum() / union))
        areas.append(float(target.sum()))
    weights = np.array(areas) / np.sum(areas)
    return {
        "macro_iou": float(np.mean(scores)),
        "area_weighted_iou": float(np.dot(scores, weights)),
    }


if __name__ == "__main__":
    app()
