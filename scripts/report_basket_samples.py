import random
from pathlib import Path

import numpy as np
import typer
from PIL import Image
from tqdm import tqdm

from court_training.inference import CourtSegmenter
from court_training.masks import MASK_NAMES

app = typer.Typer(help="Render random basketball segmentation predictions as HTML.")
DATASET_ROOT_ARGUMENT = typer.Argument(help="Root of the basketball-imgs dataset.")
CHECKPOINT_ARGUMENT = typer.Argument(help="Model checkpoint.")
OUTPUT_DIR_ARGUMENT = typer.Argument(help="Report output directory.")
DATASETS = ("basketball_51", "borgo", "e_bard_detection")
COLORS = np.array(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
    ],
    dtype=np.float32,
)


@app.command()
def main(
    dataset_root: Path = DATASET_ROOT_ARGUMENT,
    checkpoint: Path = CHECKPOINT_ARGUMENT,
    output_dir: Path = OUTPUT_DIR_ARGUMENT,
    samples_per_dataset: int = typer.Option(10, help="Images sampled from each subdataset."),
    seed: int = typer.Option(79, help="Sampling seed."),
) -> None:
    output_dir = output_dir.expanduser().resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    segmenter = CourtSegmenter.from_checkpoint(checkpoint)
    samples = sample_images(dataset_root.expanduser().resolve(), samples_per_dataset, seed)
    rows = []
    for dataset_name, image_path in tqdm(samples, desc="Predicting"):
        image = Image.open(image_path).convert("RGB")
        probabilities = segmenter.predict_proba(image)
        overlay = render_overlay(image, probabilities)
        relative_path = image_path.relative_to(dataset_root)
        overlay_path = image_dir / f"{dataset_name}_{image_path.stem}.jpg"
        overlay.save(overlay_path, quality=90)
        rows.append((dataset_name, relative_path, overlay_path.relative_to(output_dir)))

    (output_dir / "index.html").write_text(render_html(rows), encoding="utf-8")


def sample_images(dataset_root: Path, count: int, seed: int) -> list[tuple[str, Path]]:
    random.seed(seed)
    samples = []
    for dataset_name in DATASETS:
        image_root = dataset_root / "images" / dataset_name
        paths = sorted(image_root.glob("*/*.jpg"))
        if len(paths) < count:
            raise ValueError(f"{dataset_name} has only {len(paths)} jpg images")
        samples.extend((dataset_name, path) for path in random.sample(paths, count))
    return samples


def render_overlay(image: Image.Image, probabilities: np.ndarray) -> Image.Image:
    image_array = np.asarray(image.resize((probabilities.shape[1], probabilities.shape[0])), dtype=np.float32)
    masks = probabilities >= 0.5
    color = np.zeros_like(image_array)
    alpha = np.zeros(probabilities.shape[:2], dtype=np.float32)
    for index, rgb in enumerate(COLORS):
        mask = masks[..., index]
        color[mask] = rgb
        alpha[mask] = 0.42
    blended = image_array * (1 - alpha[..., None]) + color * alpha[..., None]
    return Image.fromarray(blended.astype(np.uint8))


def render_html(rows: list[tuple[str, Path, Path]]) -> str:
    cards = "\n".join(
        render_card(dataset_name, source_path, overlay_path)
        for dataset_name, source_path, overlay_path in rows
    )
    legend = "\n".join(
        f'<span><i style="background: rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"></i>{name}</span>'
        for name, rgb in zip(MASK_NAMES, COLORS.astype(int), strict=True)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Basketball Court Segmentation Samples</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #101114; color: #f5f5f5; }}
    header {{ padding: 24px 28px 12px; }}
    h1 {{ margin: 0 0 12px; font-size: 24px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 12px; color: #d7d7d7; font-size: 13px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 18px;
      padding: 20px 28px 32px;
    }}
    article {{ background: #181a20; border: 1px solid #2a2d35; border-radius: 8px; overflow: hidden; }}
    img {{ display: block; width: 100%; height: auto; }}
    .meta {{ padding: 10px 12px 12px; font-size: 13px; color: #d9d9d9; }}
    .dataset {{ color: #ffffff; font-weight: 700; }}
    .path {{ margin-top: 4px; overflow-wrap: anywhere; color: #aeb4c0; }}
  </style>
</head>
<body>
  <header>
    <h1>Basketball Court Segmentation Samples</h1>
    <div class="legend">{legend}</div>
  </header>
  <main>{cards}</main>
</body>
</html>
"""


def render_card(dataset_name: str, source_path: Path, overlay_path: Path) -> str:
    return f"""<article>
  <img src="{overlay_path}" alt="{source_path}">
  <div class="meta">
    <div class="dataset">{dataset_name}</div>
    <div class="path">{source_path}</div>
  </div>
</article>"""


if __name__ == "__main__":
    app()
