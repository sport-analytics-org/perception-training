import json
from pathlib import Path

import numpy as np
import torch
import typer
from jaxtyping import Bool, Float, UInt8
from PIL import Image
from sportanalytics import FibaCourt, NbaCourt
from sportanalytics.court.basket import BasketCourt
from torch import Tensor

from court_training.homography import centered_homography, fit_homography, render_masks

app = typer.Typer(help="Fit a basketball court homography from raster bitfield masks.")
MASK_ARGUMENT = typer.Argument(help="Raster bitfield mask WebP.")
COURT_OPTION = typer.Option("nba", help="Court template to fit: nba or fiba.")

COURTS = {
    "nba": NbaCourt,
    "fiba": FibaCourt,
}


@app.command()
def main(
    mask: Path = MASK_ARGUMENT,
    court: str = COURT_OPTION,
) -> None:
    court_template = COURTS[court]
    mask_path = mask.expanduser().resolve()
    targets = load_targets(mask_path, court_template)
    labels = tuple(label for label in court_template.areas() if label in targets)
    target_masks = stack_targets(targets, labels)
    initial = centered_homography()
    homography = fit_homography(
        target_masks,
        court_template,
        labels,
        initial,
        boosted_masks=("3pt_area", "painted_area"),
        boost=1.5,
    )
    initial_metrics = iou_metrics(target_masks, labels, initial, court_template)
    final_metrics = iou_metrics(target_masks, labels, homography, court_template)
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


def load_targets(mask_path: Path, court_template: BasketCourt) -> dict[str, Bool[np.ndarray, "H W"]]:
    bitfield: UInt8[np.ndarray, "H W"] = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    targets = {}
    for index, name in enumerate(court_template.areas()):
        bit = np.uint8(1 << index)
        if np.any(bitfield & bit):
            targets[name] = (bitfield & bit) > 0
    if not targets:
        raise ValueError(f"No masks found in {mask_path}")
    return targets


def stack_targets(
    targets: dict[str, Bool[np.ndarray, "H W"]],
    labels: tuple[str, ...],
) -> Float[Tensor, "N H W"]:
    return torch.stack([torch.tensor(targets[label].astype(np.float32)) for label in labels])


def iou_metrics(
    target_masks: Float[Tensor, "N H W"],
    labels: tuple[str, ...],
    homography: Float[np.ndarray, "3 3"],
    court_template: BasketCourt,
) -> dict[str, float]:
    output_shape = (target_masks.shape[-2], target_masks.shape[-1])
    predictions = render_masks(homography, court_template, labels, output_shape)
    scores = []
    areas = []
    for prediction_tensor, target_tensor in zip(predictions, target_masks, strict=True):
        target = target_tensor.numpy() > 0.5
        prediction = prediction_tensor.numpy() > 0.5
        union = np.logical_or(target, prediction).sum()
        scores.append(float(np.logical_and(target, prediction).sum() / union) if union else 1.0)
        areas.append(float(target.sum()))
    weights = np.array(areas) / np.sum(areas)
    return {
        "macro_iou": float(np.mean(scores)),
        "area_weighted_iou": float(np.dot(scores, weights)),
    }


if __name__ == "__main__":
    app()
