# ruff: noqa: E501
import html
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import typer
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from perception_training.dataset import BASKETBALL_ATTRIBUTE_BASE_CLASSES
from perception_training.detection.model import CourtDetector

app = typer.Typer(help="Build an HTML viewer for predicted detection attributes on tar-sharded datasets.")

CHECKPOINT_ARGUMENT = typer.Argument(help="Attribute-head CourtDetector checkpoint.")
OUTPUT_HTML_ARGUMENT = typer.Argument(help="HTML report to write.")
DATASET_ROOT_OPTION = typer.Option(..., "--dataset-root", help="Tar-sharded dataset root.")


@dataclass(frozen=True)
class ImageRef:
    dataset: str
    tar_path: Path
    member: str

    @property
    def stem(self) -> str:
        return Path(self.member).stem


@dataclass(frozen=True)
class RenderedRow:
    index: int
    ref: ImageRef
    image_src: str
    width: int
    height: int
    detections: list[dict]


@app.command()
def main(
    checkpoint: Path = CHECKPOINT_ARGUMENT,
    output_html: Path = OUTPUT_HTML_ARGUMENT,
    dataset_root: list[Path] = DATASET_ROOT_OPTION,
    total_images: int = typer.Option(100, help="Total images to include across all dataset roots."),
    threshold: float = typer.Option(0.05, help="Base detection score threshold."),
    attribute_threshold: float = typer.Option(0.35, help="Probability threshold for active attribute chips."),
    nms_iou: float = typer.Option(0.6, help="Per-class NMS IoU threshold."),
    max_detections: int = typer.Option(120, help="Maximum detections per image."),
    batch_size: int = typer.Option(4, help="Images per prediction batch."),
    seed: int = typer.Option(51, help="Sampling seed."),
) -> None:
    output_html = output_html.expanduser().resolve()
    asset_dir = output_html.with_suffix("").with_name(f"{output_html.stem}_assets")
    asset_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = CourtDetector.load(checkpoint.expanduser().resolve(), device)
    dataset_roots = [root.expanduser().resolve() for root in dataset_root]
    samples = sample_refs(dataset_roots, total_images, seed)
    logger.info("Running {} on {} images using {}", checkpoint, len(samples), device)

    rows: list[RenderedRow] = []
    for batch_start in tqdm(range(0, len(samples), batch_size), desc="Predicting"):
        batch_refs = samples[batch_start : batch_start + batch_size]
        images = [load_image(ref) for ref in batch_refs]
        predictions = model.predict(
            images,
            threshold=threshold,
            nms_iou=nms_iou,
            max_detections=max_detections,
        )
        for ref, image, prediction in zip(batch_refs, images, predictions, strict=True):
            index = len(rows) + 1
            detections = summarize_detections(model, prediction, attribute_threshold)
            rendered = draw_predictions(image, detections)
            asset_name = f"{index:03d}_{safe_name(ref.dataset)}_{safe_name(ref.stem)}.jpg"
            rendered.save(asset_dir / asset_name, quality=92)
            rows.append(
                RenderedRow(
                    index=index,
                    ref=ref,
                    image_src=f"{asset_dir.name}/{asset_name}",
                    width=image.width,
                    height=image.height,
                    detections=detections,
                )
            )

    metadata = {
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "datasets": [str(root) for root in dataset_roots],
        "total_images": len(rows),
        "threshold": threshold,
        "attribute_threshold": attribute_threshold,
        "nms_iou": nms_iou,
        "max_detections": max_detections,
    }
    output_html.write_text(render_html(metadata, rows), encoding="utf-8")
    logger.info("Wrote {}", output_html)
    typer.echo(str(output_html))


def sample_refs(dataset_roots: list[Path], total_images: int, seed: int) -> list[ImageRef]:
    per_dataset = total_images // len(dataset_roots)
    remainder = total_images % len(dataset_roots)
    rng = np.random.default_rng(seed)
    samples = []
    for dataset_index, root in enumerate(dataset_roots):
        count = per_dataset + int(dataset_index < remainder)
        refs = list_image_refs(root)
        if len(refs) <= count:
            selected = refs
        else:
            indexes = np.linspace(0, len(refs) - 1, count * 3, dtype=np.int64)
            indexes = rng.choice(np.unique(indexes), size=count, replace=False)
            selected = [refs[int(index)] for index in sorted(indexes)]
        samples.extend(selected)
    return samples


def list_image_refs(root: Path) -> list[ImageRef]:
    refs = []
    for tar_path in sorted(root.glob("*.tar")):
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().startswith("images/") and is_image(member.name):
                    refs.append(ImageRef(root.name, tar_path, member.name))
    if not refs:
        raise ValueError(f"No images found in tar shards under {root}")
    return refs


def is_image(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))


def load_image(ref: ImageRef) -> Image.Image:
    with tarfile.open(ref.tar_path) as tar:
        file = tar.extractfile(ref.member)
        if file is None:
            raise FileNotFoundError(f"{ref.tar_path}:{ref.member}")
        return Image.open(io.BytesIO(file.read())).convert("RGB")


def summarize_detections(model: CourtDetector, prediction: dict, attribute_threshold: float) -> list[dict]:
    rows = []
    for index, (box, score, label) in enumerate(
        zip(prediction["boxes"], prediction["scores"], prediction["labels"], strict=True)
    ):
        class_name = model.class_names[int(label)]
        if class_name != "player":
            continue
        attributes = {}
        for attribute_index, attribute in enumerate(model.attribute_names):
            if BASKETBALL_ATTRIBUTE_BASE_CLASSES[attribute] != class_name:
                continue
            attributes[attribute] = float(prediction["attributes"][index, attribute_index])
        active = [name for name, probability in attributes.items() if probability >= attribute_threshold]
        if not active:
            continue
        rows.append(
            {
                "index": index,
                "class_name": class_name,
                "score": float(score),
                "box": [float(value) for value in box],
                "attributes": attributes,
                "active": active,
                "max_attribute": max(attributes.values(), default=0.0),
            }
        )
    rows.sort(key=lambda row: (row["max_attribute"], row["score"]), reverse=True)
    return rows


def draw_predictions(image: Image.Image, detections: list[dict]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    width, height = image.size
    for detection in sorted(detections, key=lambda row: row["score"]):
        x1, y1, x2, y2 = normalized_box_to_pixels(detection["box"], width, height)
        color = (34, 197, 94)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label = box_label(detection)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 6)
        draw.rectangle((x1, label_y, x1 + text_width + 8, label_y + text_height + 6), fill=color)
        draw.text((x1 + 4, label_y + 3), label, fill=(0, 0, 0), font=font)
    return output


def normalized_box_to_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(np.clip(x1 * width, 0, width - 1)),
        int(np.clip(y1 * height, 0, height - 1)),
        int(np.clip(x2 * width, 0, width - 1)),
        int(np.clip(y2 * height, 0, height - 1)),
    )


def box_label(detection: dict) -> str:
    prefix = f"player {detection['index']}"
    attrs = " ".join(attribute_label(name) for name in detection["active"])
    return f"{prefix} {attrs}"


def attribute_label(name: str) -> str:
    return {
        "in_possession": "in possession",
        "jump_shot": "jump shot",
        "layup_dunk": "layup/dunk",
        "shot_block": "shot block",
    }.get(name, name)


def render_html(metadata: dict, rows: list[RenderedRow]) -> str:
    settings = html.escape(json.dumps(metadata, indent=2))
    data = json.dumps([row_to_json(row) for row in rows])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Detection Attribute Predictions</title>
  <style>
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background: #101214; color: #e8edf2; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 14px 18px; background: #171a1f; border-bottom: 1px solid #2b3138; }}
    h1 {{ margin: 0 0 8px; font-size: 20px; }}
    .meta {{ color: #aab4bf; font-size: 13px; }}
    .filters {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    button {{ border: 1px solid #3a424c; background: #20252b; color: #e8edf2; padding: 7px 10px; border-radius: 6px; cursor: pointer; }}
    button.active {{ border-color: #63d471; background: #17351f; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 16px; padding: 16px; }}
    article {{ border: 1px solid #2b3138; background: #171a1f; border-radius: 8px; overflow: hidden; }}
    img {{ display: block; width: 100%; height: auto; background: #070809; }}
    .content {{ padding: 12px; }}
    h2 {{ margin: 0 0 8px; font-size: 13px; line-height: 1.35; color: #d7dde4; word-break: break-word; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }}
    .chip {{ border: 1px solid #3a424c; border-radius: 999px; padding: 3px 7px; font-size: 12px; color: #cdd6df; }}
    .chip.hot {{ border-color: #63d471; color: #91f2a0; background: #102818; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ padding: 5px 4px; border-top: 1px solid #2b3138; text-align: left; vertical-align: top; }}
    th {{ color: #9aa6b2; font-weight: 600; }}
    code, pre {{ color: #c8d2dc; }}
    pre {{ white-space: pre-wrap; margin: 0; }}
  </style>
</head>
<body>
  <header>
    <h1>Detection Attribute Predictions</h1>
    <div class="meta">{len(rows)} images | boxes are only drawn for players with an active attribute</div>
    <div class="filters" id="filters"></div>
  </header>
  <main id="grid"></main>
  <script>
    const settings = `{settings}`;
    const rows = {data};
    const attributes = ["all", "in_possession", "jump_shot", "layup_dunk", "shot_block"];
    let active = "all";
    const grid = document.querySelector("#grid");
    const filters = document.querySelector("#filters");
    filters.innerHTML = attributes.map(name => `<button data-name="${{name}}">${{label(name)}}</button>`).join("");
    filters.addEventListener("click", event => {{
      const button = event.target.closest("button");
      if (!button) return;
      active = button.dataset.name;
      render();
    }});
    function render() {{
      filters.querySelectorAll("button").forEach(button => button.classList.toggle("active", button.dataset.name === active));
      const visible = active === "all" ? rows : rows.filter(row => row.detections.some(det => det.active.includes(active)));
      grid.innerHTML = visible.map(renderRow).join("");
    }}
    function renderRow(row) {{
      return `<article>
        <img src="${{escapeHtml(row.image_src)}}" loading="lazy">
        <div class="content">
          <h2>#${{row.index}} · ${{escapeHtml(row.dataset)}} · ${{escapeHtml(row.member)}}</h2>
          <div class="chips">${{summaryChips(row)}}</div>
          <table>
            <thead><tr><th>box</th><th>score</th><th>attributes</th></tr></thead>
            <tbody>${{row.detections.slice(0, 12).map(renderDetection).join("")}}</tbody>
          </table>
        </div>
      </article>`;
    }}
    function renderDetection(det) {{
      const attrs = Object.entries(det.attributes)
        .map(([name, value]) => `<span class="chip ${{det.active.includes(name) ? "hot" : ""}}">${{label(name)}} ${{value.toFixed(2)}}</span>`)
        .join(" ");
      return `<tr><td>${{det.class_name}} ${{det.index}}</td><td>${{det.score.toFixed(2)}}</td><td>${{attrs}}</td></tr>`;
    }}
    function summaryChips(row) {{
      const counts = {{}};
      row.detections.flatMap(det => det.active).forEach(name => counts[name] = (counts[name] || 0) + 1);
      return Object.entries(counts).map(([name, count]) => `<span class="chip hot">${{label(name)}} ×${{count}}</span>`).join("") || `<span class="chip">no active attributes</span>`;
    }}
    function label(name) {{
      const labels = {{
        all: "All",
        in_possession: "In Possession",
        jump_shot: "Jump Shot",
        layup_dunk: "Layup/Dunk",
        shot_block: "Shot Block"
      }};
      return labels[name] || name;
    }}
    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, char => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[char]));
    }}
    render();
  </script>
</body>
</html>
"""


def row_to_json(row: RenderedRow) -> dict:
    return {
        "index": row.index,
        "dataset": row.ref.dataset,
        "tar": str(row.ref.tar_path),
        "member": row.ref.member,
        "image_src": row.image_src,
        "width": row.width,
        "height": row.height,
        "detections": row.detections,
    }


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)[:140]


if __name__ == "__main__":
    app()
