import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import typer
from PIL import Image, ImageDraw
from sportanalytics import NbaCourt
from torch.nn import functional as F

app = typer.Typer(help="Fit a basketball court homography from raster bitfield masks.")
DATA_ROOT_ARGUMENT = typer.Argument(help="Exported dataset root containing basket/.")
DATASET_ARGUMENT = typer.Argument(help="Dataset name under basket/.")
IMAGE_ARGUMENT = typer.Argument(help="Image path inside the dataset, for example 003/frame.jpg.")

FIT_SIZE = 384
CENTERED_SOURCE = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float64)
CENTERED_TARGET = np.array([(0.20, 0.40), (0.80, 0.40), (1.05, 0.90), (-0.05, 0.90)], dtype=np.float64)


@dataclass(frozen=True)
class Court:
    length: float
    width: float
    key_width: float
    free_throw_distance: float
    hoop_distance_from_baseline: float
    three_point_radius: float
    three_point_side_margin: float


NBA_COURT = Court(28.6512, 15.24, 4.8768, 5.7912, 1.6002, 7.239, 0.914)
FIBA_COURT = Court(28.0, 15.0, 4.9, 5.8, 1.575, 6.75, 0.9)
MASK_NAMES = tuple(NbaCourt.areas())


@app.command()
def main(
    data_root: Path = DATA_ROOT_ARGUMENT,
    dataset: str = DATASET_ARGUMENT,
    image: str = IMAGE_ARGUMENT,
) -> None:
    basket_root = data_root.expanduser().resolve() / "basket"
    targets = load_targets(basket_root, dataset, image)
    initial = centered_homography()
    homography = fit_homography(targets, dataset, initial)
    result = {
        "dataset": dataset,
        "image": image,
        "labels": sorted(targets),
        "initial_iou": mean_iou(targets, initial, dataset),
        "final_iou": mean_iou(targets, homography, dataset),
        "homography": homography.tolist(),
    }
    print(json.dumps(result, indent=2))


def load_targets(data_root: Path, dataset: str, image_name: str) -> dict[str, np.ndarray]:
    mask_path = data_root / dataset / "masks" / Path(image_name).with_suffix(".webp")
    bitfield = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    targets = {}
    for index, name in enumerate(MASK_NAMES):
        bit = np.uint8(1 << index)
        if np.any(bitfield & bit):
            targets[name] = (bitfield & bit) > 0
    if not targets:
        raise ValueError(f"No masks found in {mask_path}")
    return targets


def fit_homography(
    targets: dict[str, np.ndarray],
    dataset: str,
    initial: np.ndarray | None = None,
    size: int = FIT_SIZE,
) -> np.ndarray:
    device = torch.device("cpu")
    labels = [label for label in MASK_NAMES if label in targets]
    templates = template_polygons(dataset)
    source_masks = torch.stack([rasterize(templates[label], size, device) for label in labels])
    target_masks = torch.stack([resize_mask(targets[label], size, device) for label in labels])
    source_masks = with_union_mask(source_masks)
    target_masks = with_union_mask(target_masks)
    weights = torch.ones(source_masks.shape[0], device=device)
    weights /= weights.sum()

    initial_tensor = torch.tensor(initial if initial is not None else centered_homography(), dtype=torch.float32)
    params = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32, requires_grad=True)
    grid = normalized_grid(size, device)
    optimizer = torch.optim.LBFGS([params], lr=0.6, max_iter=420, history_size=20, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        homography = compose_homography(initial_tensor, params)
        predicted = warp_masks(source_masks, homography, grid, size)
        loss = dice_loss(predicted, target_masks, weights)
        if "left_court" in labels and "right_court" in labels:
            loss = loss + horizon_loss(homography)
        loss.backward()
        return loss

    optimizer.step(closure)
    homography = compose_homography(initial_tensor, params.detach()).numpy()
    return homography / homography[2, 2]


def template_polygons(dataset: str) -> dict[str, list[list[tuple[float, float]]]]:
    court = FIBA_COURT if dataset == "borgo" else NBA_COURT
    left_3pt = left_3pt_area(court)
    left_paint = left_painted_area(court)
    return {
        "left_court": [[(0, 0), (0.5, 0), (0.5, 1), (0, 1)]],
        "right_court": [[(0.5, 0), (1, 0), (1, 1), (0.5, 1)]],
        "left_3pt_area": [left_3pt],
        "right_3pt_area": [mirror_x(left_3pt)],
        "left_painted_area": [left_paint],
        "right_painted_area": [mirror_x(left_paint)],
    }


def resize_mask(mask: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(mask.astype(np.float32), device=device)[None, None]
    return F.interpolate(tensor, size=(size, size), mode="area")[0, 0]


def rasterize(polygons: list[list[tuple[float, float]]], size: int, device: torch.device) -> torch.Tensor:
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon([(x * size, y * size) for x, y in polygon], fill=255)
    return torch.tensor(np.asarray(image, dtype=np.float32) / 255, device=device)


def with_union_mask(masks: torch.Tensor) -> torch.Tensor:
    return torch.cat([masks, masks.amax(dim=0, keepdim=True)])


def centered_homography() -> np.ndarray:
    return homography_from_points(CENTERED_SOURCE, CENTERED_TARGET)


def homography_from_points(source: np.ndarray, target: np.ndarray) -> np.ndarray:
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


def normalized_grid(size: int, device: torch.device) -> torch.Tensor:
    axis = torch.linspace(0, 1, size, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(-1, 3)


def warp_masks(masks: torch.Tensor, homography: torch.Tensor, grid: torch.Tensor, size: int) -> torch.Tensor:
    inverse = torch.linalg.inv(homography)
    source = grid @ inverse.T
    denominator = source[:, 2:].clamp_min(1e-6)
    source = source[:, :2] / denominator
    sample_grid = source.reshape(1, size, size, 2) * 2 - 1
    sample_grid = sample_grid.expand(masks.shape[0], -1, -1, -1)
    return F.grid_sample(masks[:, None], sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]


def compose_homography(initial: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    update = torch.stack(
        [
            torch.stack([params[0], params[1], params[2]]),
            torch.stack([params[3], params[4], params[5]]),
            torch.stack([params[6], params[7], torch.ones_like(params[0])]),
        ]
    )
    homography = initial @ update
    return homography / homography[2, 2]


def dice_loss(predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    intersection = (predicted * target).sum(dim=(1, 2))
    denominator = predicted.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2 * intersection + 1) / (denominator + 1)
    return 1 - (dice * weights).sum()


def horizon_loss(homography: torch.Tensor) -> torch.Tensor:
    points = torch.tensor(
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0.5, 0, 1], [0.5, 1, 1]],
        dtype=homography.dtype,
        device=homography.device,
    )
    return torch.relu(0.05 - points @ homography[2]).pow(2).mean() * 10


def mean_iou(targets: dict[str, np.ndarray], homography: np.ndarray, dataset: str) -> float:
    size = 768
    scores = []
    for label, polygons in template_polygons(dataset).items():
        if label not in targets:
            continue
        target = resize_mask(targets[label], size, torch.device("cpu")).numpy() > 0.5
        prediction = rasterize(project_polygons(homography, polygons), size, torch.device("cpu")).numpy() > 0.5
        union = np.logical_or(target, prediction).sum()
        scores.append(float(np.logical_and(target, prediction).sum() / union) if union else 1.0)
    return float(np.mean(scores))


def project_polygons(
    homography: np.ndarray,
    polygons: list[list[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    projected_polygons = []
    for polygon in polygons:
        projected = []
        for point in polygon:
            projected_point = project_point(homography, point)
            if projected_point is not None:
                projected.append(projected_point)
        if len(projected) >= 3:
            projected_polygons.append(projected)
    return projected_polygons


def project_point(homography: np.ndarray, point: tuple[float, float]) -> tuple[float, float] | None:
    projected = homography @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    if projected[2] <= 1e-6:
        return None
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def left_3pt_area(court: Court) -> list[tuple[float, float]]:
    half_width = court.width / 2
    hoop_x = -court.length / 2 + court.hoop_distance_from_baseline
    corner_y = half_width - court.three_point_side_margin
    dx = np.sqrt(court.three_point_radius**2 - corner_y**2)
    angles = np.linspace(np.arctan2(-corner_y, dx), np.arctan2(corner_y, dx), 48)
    points = [
        court_point(court, hoop_x + court.three_point_radius * np.cos(angle), court.three_point_radius * np.sin(angle))
        for angle in angles
    ]
    return [(0, 0), (0, 1), *points]


def left_painted_area(court: Court) -> list[tuple[float, float]]:
    baseline = -court.length / 2
    free_throw_x = baseline + court.free_throw_distance
    paint_y = court.key_width / 2
    return [
        court_point(court, baseline, paint_y),
        court_point(court, free_throw_x, paint_y),
        court_point(court, free_throw_x, -paint_y),
        court_point(court, baseline, -paint_y),
    ]


def court_point(court: Court, x: float, y: float) -> tuple[float, float]:
    return (x + court.length / 2) / court.length, (court.width / 2 - y) / court.width


def mirror_x(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(1 - x, y) for x, y in reversed(points)]


if __name__ == "__main__":
    app()
