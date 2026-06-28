import html
import json
import random
from pathlib import Path

import numpy as np
import sportkit as sk
import torch
import typer
from jaxtyping import Bool, Float
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from tqdm import tqdm

import perception_training as pt
from perception_training.segmentation.model import CourtSegmenter
from perception_training.warp import warp

app = typer.Typer(help="Predict basketball masks, fit homographies to them, and write an HTML report.")

DATASET_ROOT_ARGUMENT = typer.Argument(help="Basketball image dataset root.")
CHECKPOINT_ARGUMENT = typer.Argument(help="CourtSegmenter checkpoint.")
OUTPUT_DIR_ARGUMENT = typer.Argument(help="Directory where the HTML report is written.")
DATASETS_OPTION = typer.Option(
    ["nba_mixed", "fiba_borgo", "nba_detection1"],
    "--dataset",
    help="Subdataset to sample. Can be passed multiple times.",
)
COLORS = np.array(
    [
        (58, 134, 255),
        (255, 122, 69),
        (45, 197, 244),
        (255, 183, 77),
        (105, 214, 155),
        (239, 83, 80),
    ],
    dtype=np.float32,
)


@app.command()
def main(
    dataset_root: Path = DATASET_ROOT_ARGUMENT,
    checkpoint: Path = CHECKPOINT_ARGUMENT,
    output_dir: Path = OUTPUT_DIR_ARGUMENT,
    count_per_dataset: int = typer.Option(100, help="Number of random images to sample per subdataset."),
    datasets: list[str] = DATASETS_OPTION,
    seed: int = typer.Option(7, help="Random seed."),
) -> None:
    dataset_root = dataset_root.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    image_paths = unlabelled_images(dataset_root)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = CourtSegmenter.load(checkpoint, device)
    rng = random.Random(seed)
    samples = sample_by_dataset(image_paths, count_per_dataset * 10, tuple(datasets), rng)
    expected = count_per_dataset * len(datasets)
    logger.info("Sampling up to {} candidates from {} unlabelled images", len(samples), len(image_paths))

    rows = []
    counts = dict.fromkeys(datasets, 0)
    for image_path in tqdm(samples, desc="Predicting and fitting"):
        if len(rows) == expected:
            break
        dataset = image_path.relative_to(dataset_root / "images").parts[0]
        if counts[dataset] == count_per_dataset:
            continue
        original_image = Image.open(image_path).convert("RGB")
        prediction = model.predict([original_image])
        probabilities = torch.from_numpy(prediction["masks"][0])
        keypoints = prediction["keypoints"][0]
        visibility = prediction["visibility"][0]

        is_fiba_dataset = dataset.startswith("fiba_")
        court_name = "fiba" if is_fiba_dataset else "nba"
        court = sk.courts.FibaCourt if is_fiba_dataset else sk.courts.NbaCourt
        homography_mask_names = tuple(court.planar_areas())
        homography_probabilities = probabilities[: len(homography_mask_names)]
        visible = visibility >= 0.5
        if visible.sum() < 4:
            logger.info("Skipping {}: only {} visible keypoints", image_path, visible.sum())
            continue
        matrix, fitted, score = pt.homography.fit_court(
            court, homography_mask_names, model.keypoint_names, homography_probabilities, keypoints, visible
        )
        fitted_homography = matrix.cpu().numpy()
        fitted_original = render_at_image_size(court, homography_mask_names, fitted_homography, original_image.size)
        fitted_keypoints, fitted_visibility = project_keypoints(court, model.keypoint_names, fitted_homography)

        save_labels(
            dataset_root,
            image_path,
            court_name,
            fitted_original,
            fitted_homography,
            fitted_keypoints,
            fitted_visibility,
            score,
        )
        panel_path = panel_dir / f"{len(rows):02d}_{image_path.stem}.jpg"
        image = original_image.resize((model.image_size[1], model.image_size[0]), Image.Resampling.BILINEAR)
        make_panel(
            image,
            homography_probabilities,
            fitted,
            keypoints,
            visibility,
            image_path,
            court_name,
            score,
            panel_path,
        )
        rows.append((image_path.relative_to(dataset_root), panel_path.relative_to(output_dir), court_name, score))
        counts[dataset] += 1

    missing = {dataset: count_per_dataset - count for dataset, count in counts.items() if count < count_per_dataset}
    if missing:
        raise ValueError(f"Not enough successful fits: {missing}")

    write_report(output_dir / "index.html", rows)
    logger.info("Wrote {}", output_dir / "index.html")


def unlabelled_images(dataset_root: Path) -> list[Path]:
    images_root = dataset_root / "images"
    masks_root = dataset_root / "masks"
    paths = []
    for image_path in sorted(images_root.glob("*/*/*.jpg")):
        mask_path = masks_root / image_path.relative_to(images_root).with_suffix(".json")
        if not mask_path.is_file():
            paths.append(image_path)
    return paths


def sample_by_dataset(image_paths: list[Path], count: int, datasets: tuple[str, ...], rng: random.Random) -> list[Path]:
    samples = []
    for dataset in datasets:
        paths = [path for path in image_paths if path.parent.parent.name == dataset]
        if len(paths) < count:
            raise ValueError(f"{dataset} has {len(paths)} unlabelled images, cannot sample {count}")
        samples.extend(rng.sample(paths, count))
    rng.shuffle(samples)
    return samples


def render_at_image_size(
    court: sk.courts.BasketCourt,
    mask_names: tuple[str, ...],
    matrix: Float[np.ndarray, "3 3"],
    size: tuple[int, int],
) -> Float[Tensor, "N H W"]:
    width, height = size
    source_masks = pt.homography.template_masks(court, mask_names, width, torch.device("cpu"))
    homography_tensor = torch.tensor(matrix, dtype=source_masks.dtype)
    return warp(source_masks, homography_tensor, (height, width))


def project_keypoints(
    court: sk.courts.BasketCourt,
    keypoint_names: tuple[str, ...],
    matrix: Float[np.ndarray, "3 3"],
) -> tuple[Float[np.ndarray, "K 2"], Bool[np.ndarray, "K"]]:
    points = pt.homography.normalized_keypoints(court, keypoint_names)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    projected = homogeneous @ matrix.T
    keypoints = projected[:, :2] / projected[:, 2:]
    visibility = np.logical_and.reduce(
        [
            keypoints[:, 0] >= 0,
            keypoints[:, 0] <= 1,
            keypoints[:, 1] >= 0,
            keypoints[:, 1] <= 1,
        ]
    )
    return keypoints, visibility


def save_labels(
    dataset_root: Path,
    image_path: Path,
    court_name: str,
    masks: Float[Tensor, "N H W"],
    homography: Float[np.ndarray, "3 3"],
    keypoints: Float[np.ndarray, "K 2"],
    visibility: Bool[np.ndarray, "K"],
    score: float,
) -> None:
    image_relative = image_path.relative_to(dataset_root / "images")
    dataset, shard = image_relative.parts[:2]
    image_key = str(Path(*image_relative.parts[1:]))

    mask_path = dataset_root / "masks" / image_relative.with_suffix(".json")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.write_text(json.dumps(mask_surfaces(court_name, masks), indent=2) + "\n")

    homography_path = dataset_root / "homography" / dataset / f"{shard}.json"
    update_json(
        homography_path,
        "homographies",
        image_key,
        {
            "court": court_name,
            "matrix": homography.tolist(),
            "soft_iou": score,
        },
    )

    keypoint_path = dataset_root / "keypoints" / dataset / f"{shard}.json"
    points = [
        {"position": position.tolist(), "visible": bool(visible)}
        for position, visible in zip(keypoints, visibility, strict=True)
    ]
    update_json(keypoint_path, "keypoints", image_key, {"court": court_name, "points": points})


def mask_surfaces(court_name: str, masks: Float[Tensor, "N H W"]) -> dict:
    court = sk.courts.FibaCourt if court_name == "fiba" else sk.courts.NbaCourt
    labels = tuple(court.planar_areas())
    surfaces = {}
    for label, mask in zip(labels, masks, strict=True):
        polygon = sk.polygons.trace_mask(mask.numpy() > 0.5)
        surfaces[label] = polygon.to_json()
    return surfaces


def update_json(path: Path, key: str, image_key: str, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.is_file() else {key: {}}
    data[key][image_key] = value
    path.write_text(json.dumps(data, indent=2) + "\n")


def make_panel(
    image: Image.Image,
    probabilities: Float[Tensor, "N H W"],
    fitted: Float[Tensor, "N H W"],
    keypoints: Float[np.ndarray, "K 2"],
    visibility: Float[np.ndarray, "*K"],
    image_path: Path,
    court_name: str,
    score: float,
    output_path: Path,
) -> None:
    base = image.convert("RGB")
    panels = [
        labeled(base, "image"),
        labeled(overlay(base, probabilities), "model probabilities"),
        labeled(draw_keypoints(base, keypoints, visibility), "predicted keypoints"),
        labeled(overlay(base, fitted), "fitted homography"),
        labeled(diff_image(probabilities, fitted), "absolute difference"),
    ]
    width = sum(panel.width for panel in panels)
    canvas = Image.new("RGB", (width, panels[0].height + 42), "white")
    draw = ImageDraw.Draw(canvas)
    image_name = f"{image_path.parent.parent.name}/{image_path.parent.name}/{image_path.name}"
    title = f"{image_name} | {court_name} | soft IoU {score:.3f}"
    draw.text((10, 12), title, fill=(20, 20, 20))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 42))
        x += panel.width
    canvas.save(output_path, quality=92)


def draw_keypoints(
    image: Image.Image,
    keypoints: Float[np.ndarray, "K 2"],
    visibility: Float[np.ndarray, "*K"],
    threshold: float = 0.5,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    for x, y in keypoints[visibility >= threshold]:
        center_x = float(x) * (width - 1)
        center_y = float(y) * (height - 1)
        draw.ellipse((center_x - 3, center_y - 3, center_x + 3, center_y + 3), fill=(255, 232, 64))
        draw.ellipse((center_x - 4, center_y - 4, center_x + 4, center_y + 4), outline=(20, 20, 20))
    return output


def overlay(image: Image.Image, masks: Float[Tensor, "N H W"]) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha = masks.clamp(0, 1).numpy()[..., None] * 0.45
    colors = COLORS[np.arange(len(masks)) % len(COLORS)][:, None, None, :]
    overlay_rgb = (alpha * colors).sum(axis=0)
    total_alpha = np.clip(alpha.sum(axis=0), 0, 0.75)
    result = base * (1 - total_alpha) + overlay_rgb
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def diff_image(probabilities: Float[Tensor, "N H W"], fitted: Float[Tensor, "N H W"]) -> Image.Image:
    diff = (probabilities - fitted).abs().mean(dim=0).clamp(0, 1).numpy()
    red = (diff * 255).astype(np.uint8)
    blue = ((1 - diff) * 60).astype(np.uint8)
    rgb = np.stack([red, np.zeros_like(red), blue], axis=-1)
    return Image.fromarray(rgb)


def labeled(image: Image.Image, label: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, bbox[2] + 12, bbox[3] + 10), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255), font=font)
    return image


def write_report(path: Path, rows: list[tuple[Path, Path, str, float]]) -> None:
    items = []
    for image_path, panel_path, court_name, score in rows:
        items.append(
            f"<section><h2>{html.escape(str(image_path))}</h2>"
            f"<p>{court_name} | soft IoU {score:.3f}</p>"
            f"<img src='{html.escape(str(panel_path))}'></section>"
        )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
        "margin:24px;background:#111;color:#eee}"
        "section{margin-bottom:30px}img{max-width:100%;height:auto;border:1px solid #333;background:#000}"
        "h1{font-size:24px}h2{font-size:16px;margin-bottom:4px}p{color:#bbb;margin-top:0}</style>"
        "</head><body><h1>Predicted masks fitted with homography</h1>" + "\n".join(items) + "</body></html>\n"
    )


if __name__ == "__main__":
    app()
