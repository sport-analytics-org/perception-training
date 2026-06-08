import json
from pathlib import Path
from shutil import copy2

import typer
from loguru import logger
from tqdm import tqdm

app = typer.Typer(help="Export original labelled subdatasets into flat train/val folders.")
TRAIN_DATASET_OPTION = typer.Option(..., help="Subdataset name to copy into train.")
VAL_DATASET_OPTION = typer.Option(..., help="Subdataset name to copy into val.")


@app.command()
def main(
    dataset_root: Path,
    output_root: Path,
    train_dataset: list[str] = TRAIN_DATASET_OPTION,
    val_dataset: list[str] = VAL_DATASET_OPTION,
    keypoints: bool = typer.Option(True, "--keypoints/--no-keypoints", help="Export keypoints with images and masks."),
) -> None:
    export_split(dataset_root, train_dataset, output_root / "train", keypoints)
    export_split(dataset_root, val_dataset, output_root / "val", keypoints)


def export_split(dataset_root: Path, dataset_names: list[str], output_root: Path, export_keypoints: bool) -> None:
    image_output = output_root / "images"
    mask_output = output_root / "masks"
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)
    keypoint_output = output_root / "keypoints"
    if export_keypoints:
        keypoint_output.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_names:
        keypoints = read_keypoints(dataset_root, dataset_name) if export_keypoints else {}
        pairs = image_mask_pairs(dataset_root, dataset_name)
        copied = 0
        skipped = 0
        for image_path, mask_path in tqdm(pairs, desc=dataset_name):
            name = flat_name(dataset_root, dataset_name, image_path)
            image_keypoint = keypoints.get(image_key(dataset_root, dataset_name, image_path))
            if export_keypoints and image_keypoint is None:
                skipped += 1
                continue
            copy2(image_path, image_output / f"{name}.jpg")
            copy2(mask_path, mask_output / f"{name}.webp")
            if export_keypoints:
                output = json.dumps(image_keypoint, indent=2) + "\n"
                (keypoint_output / f"{name}.json").write_text(output)
            copied += 1
        logger.info("{}: copied {} labelled images", dataset_name, copied)
        if export_keypoints:
            logger.info("{}: skipped {} labelled images without keypoints", dataset_name, skipped)


def image_mask_pairs(dataset_root: Path, dataset_name: str) -> list[tuple[Path, Path]]:
    image_root = dataset_root / "images" / dataset_name
    mask_root = dataset_root / "masks" / dataset_name
    pairs = []
    for image_path in sorted(image_root.glob("*/*.jpg")):
        mask_path = mask_root / image_path.relative_to(image_root).with_suffix(".webp")
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def read_keypoints(dataset_root: Path, dataset_name: str) -> dict[str, dict]:
    keypoints = {}
    for path in sorted((dataset_root / "keypoints" / dataset_name).glob("*.json")):
        keypoints.update(json.loads(path.read_text())["keypoints"])
    return keypoints


def image_key(dataset_root: Path, dataset_name: str, image_path: Path) -> str:
    image_root = dataset_root / "images" / dataset_name
    return str(image_path.relative_to(image_root))


def flat_name(dataset_root: Path, dataset_name: str, image_path: Path) -> str:
    image_root = dataset_root / "images" / dataset_name
    relative_stem = image_path.relative_to(image_root).with_suffix("")
    return f"{dataset_name}_{'_'.join(relative_stem.parts)}"


if __name__ == "__main__":
    app()
