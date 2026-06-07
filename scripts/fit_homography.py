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
from torch.nn import functional as F

from court_training.loss import dice_loss
from court_training.warp import warp

app = typer.Typer(help="Fit a basketball court homography from raster bitfield masks.")
MASK_ARGUMENT = typer.Argument(help="Raster bitfield mask WebP.")
COURT_OPTION = typer.Option("nba", help="Court template to fit: nba or fiba.")

FIT_SIZE = 384
CENTERED_SOURCE = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float64)
CENTERED_TARGET = np.array(
    [(0.20, 0.40), (0.80, 0.40), (1.05, 0.90), (-0.05, 0.90)],
    dtype=np.float64,
)


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
    initial = centered_homography()
    homography = fit_homography(targets, court_template)
    result = {
        "mask": str(mask_path),
        "court": court,
        "labels": sorted(targets),
        "initial_iou": mean_iou(targets, initial, court_template),
        "final_iou": mean_iou(targets, homography, court_template),
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


def fit_homography(
    targets: dict[str, Bool[np.ndarray, "H W"]],
    court_template: BasketCourt,
    size: int = FIT_SIZE,
) -> Float[np.ndarray, "3 3"]:
    device = torch.device("cpu")
    labels = [label for label in court_template.areas() if label in targets]
    source_masks = template_masks(court_template, labels, size, device)
    target_masks = torch.stack([resize_mask(targets[label], size, device) for label in labels])
    source_masks = torch.cat([source_masks, source_masks.amax(dim=0, keepdim=True)])
    target_masks = torch.cat([target_masks, target_masks.amax(dim=0, keepdim=True)])
    weights = torch.ones(source_masks.shape[0], device=device)
    weights /= weights.sum()

    initial_tensor = torch.tensor(centered_homography(), dtype=torch.float32)
    params = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([params], lr=0.6, max_iter=420, history_size=20, line_search_fn="strong_wolfe")

    def closure() -> Float[Tensor, ""]:
        optimizer.zero_grad(set_to_none=True)
        homography = compose_homography(initial_tensor, params)
        output_shape = (target_masks.shape[-2], target_masks.shape[-1])
        predicted = warp(source_masks, homography, output_shape)
        loss = dice_loss(predicted, target_masks, weights)
        if "left_court" in labels and "right_court" in labels:
            loss = loss + horizon_loss(homography)
        loss.backward()
        return loss

    optimizer.step(closure)
    homography = compose_homography(initial_tensor, params.detach()).numpy()
    return homography / homography[2, 2]


def template_masks(
    court_template: BasketCourt,
    labels: list[str],
    width: int,
    device: torch.device,
) -> Float[Tensor, "masks H W"]:
    masks = []
    for label in labels:
        image = court_template.get_mask_image(label, width).convert("L")
        array = np.asarray(image, dtype=np.float32) / 255
        masks.append(torch.tensor(array, device=device))
    return torch.stack(masks)


def resize_mask(mask: Bool[np.ndarray, "H W"], size: int, device: torch.device) -> Float[Tensor, "H W"]:
    tensor = torch.tensor(mask.astype(np.float32), device=device)[None, None]
    return F.interpolate(tensor, size=(size, size), mode="area")[0, 0]


def centered_homography() -> Float[np.ndarray, "3 3"]:
    return homography_from_points(CENTERED_SOURCE, CENTERED_TARGET)


def homography_from_points(
    source: Float[np.ndarray, "points 2"],
    target: Float[np.ndarray, "points 2"],
) -> Float[np.ndarray, "3 3"]:
    rows = []
    values = []
    for (x, y), (u, v) in zip(source, target, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        values.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.append(v)
    params = np.linalg.solve(np.array(rows), np.array(values))
    homography = np.array(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0],
        ],
        dtype=np.float64,
    )
    return homography / homography[2, 2]


def compose_homography(initial: Float[Tensor, "3 3"], params: Float[Tensor, "8"]) -> Float[Tensor, "3 3"]:
    update = torch.stack(
        [
            torch.stack([params[0], params[1], params[2]]),
            torch.stack([params[3], params[4], params[5]]),
            torch.stack([params[6], params[7], torch.ones_like(params[0])]),
        ]
    )
    homography = initial @ update
    return homography / homography[2, 2]


def horizon_loss(homography: Float[Tensor, "3 3"]) -> Float[Tensor, ""]:
    points = torch.tensor(
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0.5, 0, 1], [0.5, 1, 1]],
        dtype=homography.dtype,
        device=homography.device,
    )
    return torch.relu(0.05 - points @ homography[2]).pow(2).mean() * 10


def mean_iou(
    targets: dict[str, Bool[np.ndarray, "H W"]],
    homography: Float[np.ndarray, "3 3"],
    court_template: BasketCourt,
) -> float:
    size = 768
    device = torch.device("cpu")
    labels = [label for label in court_template.areas() if label in targets]
    source_masks = template_masks(court_template, labels, size, device)
    target_masks = torch.stack([resize_mask(targets[label], size, device) for label in labels])
    output_shape = (target_masks.shape[-2], target_masks.shape[-1])
    predictions = warp(source_masks, torch.tensor(homography, dtype=torch.float32), output_shape)
    scores = []
    for prediction_tensor, target_tensor in zip(predictions, target_masks, strict=True):
        target = target_tensor.numpy() > 0.5
        prediction = prediction_tensor.numpy() > 0.5
        union = np.logical_or(target, prediction).sum()
        scores.append(float(np.logical_and(target, prediction).sum() / union) if union else 1.0)
    return float(np.mean(scores))


if __name__ == "__main__":
    app()
