import json
import tarfile
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

app = typer.Typer(help="Export reviewed tar-sharded subdatasets into flat train/val folders.")
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
    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
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
        copied = 0
        skipped = 0
        shard_paths = sorted((dataset_root / dataset_name).glob("*.tar"))
        for shard_path in tqdm(shard_paths, desc=dataset_name):
            shard_copied, shard_skipped = export_shard(shard_path, dataset_name, output_root, export_keypoints)
            copied += shard_copied
            skipped += shard_skipped
        logger.info("{}: copied {} reviewed images", dataset_name, copied)
        if export_keypoints:
            logger.info("{}: skipped {} reviewed images without keypoints", dataset_name, skipped)


def export_shard(
    shard_path: Path,
    dataset_name: str,
    output_root: Path,
    export_keypoints: bool,
) -> tuple[int, int]:
    with tarfile.open(shard_path) as shard:
        members = {member.name: member for member in shard.getmembers() if member.isfile()}
        metadata = read_json(shard, members, f"metadata/{shard_id(shard_path)}.json")
        keypoints = read_json(shard, members, f"keypoints/{shard_id(shard_path)}.json").get("keypoints", {})
        reviewed_keys = reviewed_images(metadata)

        copied = 0
        skipped = 0
        for image_key in sorted(reviewed_keys):
            image_name = Path(image_key).name
            image_member = members[f"images/{image_name}"]
            mask_member = members[f"masks/{Path(image_name).with_suffix('.webp')}"]
            image_keypoints = keypoints.get(image_key)
            if export_keypoints and image_keypoints is None:
                skipped += 1
                continue

            name = flat_name(dataset_name, image_key)
            write_member(shard, image_member, output_root / "images" / f"{name}.jpg")
            write_member(shard, mask_member, output_root / "masks" / f"{name}.webp")
            if export_keypoints:
                output = json.dumps(image_keypoints, indent=2) + "\n"
                (output_root / "keypoints" / f"{name}.json").write_text(output)
            copied += 1

    return copied, skipped


def shard_id(path: Path) -> str:
    return path.stem[-3:]


def read_json(shard: tarfile.TarFile, members: dict[str, tarfile.TarInfo], name: str) -> dict:
    member = members.get(name)
    if member is None:
        return {}
    file = shard.extractfile(member)
    return json.load(file) if file else {}


def reviewed_images(metadata: dict) -> set[str]:
    reviews = metadata.get("reviews", {})
    return {image_key for image_key, masks in reviews.items() if all(masks.values())}


def write_member(shard: tarfile.TarFile, member: tarfile.TarInfo, output_path: Path) -> None:
    file = shard.extractfile(member)
    output_path.write_bytes(file.read() if file else b"")


def flat_name(dataset_name: str, image_key: str) -> str:
    relative_stem = Path(image_key).with_suffix("")
    return f"{dataset_name}_{'_'.join(relative_stem.parts)}"


if __name__ == "__main__":
    app()
