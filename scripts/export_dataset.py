from pathlib import Path
from shutil import copy2

import typer
from tqdm import tqdm

app = typer.Typer(help="Export original labelled subdatasets into flat train/val folders.")
TRAIN_SOURCE_OPTION = typer.Option(..., help="Original subdataset folder to copy into train.")
VAL_SOURCE_OPTION = typer.Option(..., help="Original subdataset folder to copy into val.")


@app.command()
def main(
    output_root: Path,
    train_source: list[Path] = TRAIN_SOURCE_OPTION,
    val_source: list[Path] = VAL_SOURCE_OPTION,
) -> None:
    export_split(train_source, output_root / "train")
    export_split(val_source, output_root / "val")


def export_split(sources: list[Path], output_root: Path) -> None:
    image_output = output_root / "images"
    mask_output = output_root / "masks"
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    for source in sources:
        pairs = image_mask_pairs(source)
        for image_path, mask_path in tqdm(pairs, desc=source.name):
            name = flat_name(source, image_path)
            copy2(image_path, image_output / f"{name}.jpg")
            copy2(mask_path, mask_output / f"{name}.webp")


def image_mask_pairs(source: Path) -> list[tuple[Path, Path]]:
    image_root = source / "images"
    mask_root = source / "masks"
    pairs = []
    for image_path in sorted(image_root.glob("*/*.jpg")):
        mask_path = mask_root / image_path.relative_to(image_root).with_suffix(".webp")
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def flat_name(source: Path, image_path: Path) -> str:
    relative_stem = image_path.relative_to(source / "images").with_suffix("")
    return f"{source.name}_{'_'.join(relative_stem.parts)}"


if __name__ == "__main__":
    app()
