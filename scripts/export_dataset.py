import dataclasses
import json
import tarfile
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from perception_training.dataset import BASKETBALL_DETECTION_ATTRIBUTES

app = typer.Typer(help="Export tar-sharded subdatasets into flat train/val folders.")
TRAIN_DATASET_OPTION = typer.Option(None, help="Subdataset name to copy into train.")
VAL_DATASET_OPTION = typer.Option(None, help="Subdataset name to copy into val.")


@dataclasses.dataclass(frozen=True)
class ExportOptions:
    masks: bool
    keypoints: bool
    detections: bool


@app.command()
def main(
    dataset_root: Path,
    output_root: Path,
    train_dataset: list[str] | None = TRAIN_DATASET_OPTION,
    val_dataset: list[str] | None = VAL_DATASET_OPTION,
    masks: bool = typer.Option(True, "--masks/--no-masks", help="Export reviewed segmentation masks."),
    keypoints: bool = typer.Option(True, "--keypoints/--no-keypoints", help="Export keypoints with masks."),
    detections: bool = typer.Option(
        True,
        "--detections/--no-detections",
        help="Export object detections as NPZ files.",
    ),
) -> None:
    if not masks and not detections:
        raise typer.BadParameter("At least one of --masks or --detections must be enabled.")
    if not train_dataset and not val_dataset:
        raise typer.BadParameter("Select at least one subdataset with --train-dataset or --val-dataset.")

    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    options = ExportOptions(masks=masks, keypoints=keypoints, detections=detections)

    if train_dataset:
        export_split(dataset_root, output_root / "train", train_dataset, options)
    if val_dataset:
        export_split(dataset_root, output_root / "val", val_dataset, options)


def export_split(
    dataset_root: Path,
    output_root: Path,
    dataset_names: list[str],
    options: ExportOptions,
) -> None:
    make_output_dirs(output_root, options)

    for dataset_name in dataset_names:
        copied = 0
        skipped = 0
        shard_paths = sorted((dataset_root / dataset_name).glob("*.tar"))
        for shard_path in tqdm(shard_paths, desc=dataset_name):
            shard_copied, shard_skipped = export_shard(shard_path, dataset_name, output_root, options)
            copied += shard_copied
            skipped += shard_skipped
        logger.info("{}: copied {} images", dataset_name, copied)
        if options.masks and options.keypoints:
            logger.info("{}: skipped {} reviewed images without keypoints", dataset_name, skipped)


def export_shard(
    shard_path: Path,
    dataset_name: str,
    output_root: Path,
    options: ExportOptions,
) -> tuple[int, int]:
    with tarfile.open(shard_path) as shard:
        members = {member.name: member for member in shard.getmembers() if member.isfile()}
        metadata = read_json(shard, members, f"metadata/{shard_id(shard_path)}.json")
        keypoints = read_json(shard, members, f"keypoints/{shard_id(shard_path)}.json").get("keypoints", {})
        reviewed_keys = reviewed_images(metadata, members)
        detection_keys = {Path(name).with_suffix(".jpg").name for name in members if name.startswith("detections/")}
        export_keys = set()
        if options.masks:
            export_keys |= reviewed_keys
        if options.detections:
            export_keys |= detection_keys

        copied = 0
        skipped = 0
        for image_key in sorted(export_keys):
            image_name = Path(image_key).name
            image_member = require_member(members, f"images/{image_name}")
            mask_name = f"masks/{Path(image_name).with_suffix('.webp')}"
            keypoint_data = keypoints.get(image_key)
            should_export_mask = options.masks and image_key in reviewed_keys
            if should_export_mask and options.keypoints and keypoint_data is None:
                should_export_mask = False
                skipped += 1
                if not options.detections:
                    continue

            name = flat_name(dataset_name, image_key)
            write_member(shard, image_member, output_root / "images" / f"{name}.jpg")
            if should_export_mask:
                mask_member = require_member(members, mask_name)
                write_member(shard, mask_member, output_root / "masks" / f"{name}.webp")
            if should_export_mask and options.keypoints:
                output = json.dumps(keypoint_data, indent=2) + "\n"
                (output_root / "keypoints" / f"{name}.json").write_text(output)
            if options.detections:
                detections = read_json(shard, members, f"detections/{Path(image_name).with_suffix('.json')}")
                write_detections_npz(detections, output_root / "detections" / f"{name}.npz")
            copied += 1

    return copied, skipped


def make_output_dirs(output_root: Path, options: ExportOptions) -> None:
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    if options.masks:
        (output_root / "masks").mkdir(parents=True, exist_ok=True)
    if options.masks and options.keypoints:
        (output_root / "keypoints").mkdir(parents=True, exist_ok=True)
    if options.detections:
        (output_root / "detections").mkdir(parents=True, exist_ok=True)


def shard_id(path: Path) -> str:
    return path.stem[-3:]


def read_json(shard: tarfile.TarFile, members: dict[str, tarfile.TarInfo], name: str) -> dict:
    member = members.get(name)
    if member is None:
        return {}
    file = shard.extractfile(member)
    if file is None:
        raise FileNotFoundError(name)
    return json.load(file)


def reviewed_images(metadata: dict, members: dict[str, tarfile.TarInfo]) -> set[str]:
    reviews = metadata.get("reviews", {})
    if reviews:
        return {image_key for image_key, masks in reviews.items() if all(masks.values())}
    if metadata.get("approvals", {}).get("masks"):
        return {Path(member).with_suffix(".jpg").name for member in members if member.startswith("masks/")}
    return set()


def require_member(members: dict[str, tarfile.TarInfo], name: str) -> tarfile.TarInfo:
    member = members.get(name)
    if member is None:
        raise FileNotFoundError(name)
    return member


def write_member(shard: tarfile.TarFile, member: tarfile.TarInfo, output_path: Path) -> None:
    file = shard.extractfile(member)
    if file is None:
        raise FileNotFoundError(member.name)
    output_path.write_bytes(file.read())


def write_detections_npz(data: dict, output_path: Path) -> None:
    detections = data.get("detections", [])
    boxes_xywh = np.array([detection["bbox_xywh"] for detection in detections], dtype=np.float32).reshape(-1, 4)
    category_names = np.array([detection["category_name"] for detection in detections], dtype=str)
    attributes = np.array(
        [
            [bool(detection.get("attributes", {}).get(name, False)) for name in BASKETBALL_DETECTION_ATTRIBUTES]
            for detection in detections
        ],
        dtype=np.bool_,
    ).reshape(-1, len(BASKETBALL_DETECTION_ATTRIBUTES))
    np.savez_compressed(
        output_path,
        boxes_xywh=boxes_xywh,
        category_names=category_names,
        attributes=attributes,
        attribute_names=np.array(BASKETBALL_DETECTION_ATTRIBUTES, dtype=str),
    )


def flat_name(dataset_name: str, image_key: str) -> str:
    relative_stem = Path(image_key).with_suffix("")
    return f"{dataset_name}_{'_'.join(relative_stem.parts)}"


if __name__ == "__main__":
    app()
