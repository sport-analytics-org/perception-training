import json
from pathlib import Path

import numpy as np
import sportkit as sk
import torch
import typer
from jaxtyping import Float
from torch import Tensor

import perception_training as pt

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
    width, height = RASTER_SIZE
    target_masks_by_label = pt.dataset.read_polygon_masks(mask_path, width, height)
    labels = tuple(target_masks_by_label)
    target_mask_tensors = [torch.tensor(mask.astype(np.float32)) for mask in target_masks_by_label.values()]
    target_masks = torch.stack(target_mask_tensors)
    source_masks = pt.homography.template_masks(court_template, labels, width, torch.device("cpu"))

    mask_multipliers = [1.5 if "3pt_area" in label or "painted_area" in label else 1.0 for label in labels]
    multipliers = torch.tensor(mask_multipliers)
    initial = torch.tensor(pt.homography.centered_homography(), dtype=target_masks.dtype)
    homography = pt.homography.fit_homography(source_masks, target_masks, initial, multipliers)
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


def iou_metrics(
    source_masks: Float[Tensor, "N H W"],
    target_masks: Float[Tensor, "N H W"],
    homography: Float[np.ndarray, "3 3"],
) -> dict[str, float]:
    output_shape = (target_masks.shape[-2], target_masks.shape[-1])
    homography_tensor = torch.tensor(homography, dtype=source_masks.dtype, device=source_masks.device)
    predictions = pt.warp.warp(source_masks, homography_tensor, output_shape)
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
