from pathlib import Path
from shutil import copy2

import typer
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
) -> None:
    export_split(dataset_root, train_dataset, output_root / "train")
    export_split(dataset_root, val_dataset, output_root / "val")


def export_split(dataset_root: Path, dataset_names: list[str], output_root: Path) -> None:
    image_output = output_root / "images"
    mask_output = output_root / "masks"
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_names:
        pairs = image_mask_pairs(dataset_root, dataset_name)
        for image_path, mask_path in tqdm(pairs, desc=dataset_name):
            name = flat_name(dataset_root, dataset_name, image_path)
            copy2(image_path, image_output / f"{name}.jpg")
            copy2(mask_path, mask_output / f"{name}.webp")


def image_mask_pairs(dataset_root: Path, dataset_name: str) -> list[tuple[Path, Path]]:
    image_root = dataset_root / dataset_name / "images"
    mask_root = dataset_root / dataset_name / "masks"
    pairs = []
    for image_path in sorted(image_root.glob("*/*.jpg")):
        mask_path = mask_root / image_path.relative_to(image_root).with_suffix(".webp")
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def flat_name(dataset_root: Path, dataset_name: str, image_path: Path) -> str:
    image_root = dataset_root / dataset_name / "images"
    relative_stem = image_path.relative_to(image_root).with_suffix("")
    return f"{dataset_name}_{'_'.join(relative_stem.parts)}"


if __name__ == "__main__":
    app()
