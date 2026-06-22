import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from perception_training.dataset import BASKETBALL_DETECTION_ATTRIBUTES

app = typer.Typer(help="Build similarity-clustered attribute and classic detection splits.")

SOURCE_ROOT_ARGUMENT = typer.Argument(help="Flat exported detection dataset with images/ and detections/.")
OUTPUT_ROOT_ARGUMENT = typer.Argument(help="Output root that will receive attributes/ and classic/ splits.")

CLASSIC_CLASSES = (
    "ball",
    "ball-in-basket",
    "number",
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "referee",
    "rim",
)
ATTRIBUTE_TO_CLASS = {
    "in_basket": "ball-in-basket",
    "in_possession": "player-in-possession",
    "jump_shot": "player-jump-shot",
    "layup_dunk": "player-layup-dunk",
    "shot_block": "player-shot-block",
}


@app.command()
def main(
    source_root: Path = SOURCE_ROOT_ARGUMENT,
    output_root: Path = OUTPUT_ROOT_ARGUMENT,
    clusters: int = typer.Option(5, help="Number of visual clusters."),
    target_val_fraction: float = typer.Option(0.2, help="Preferred validation cluster size."),
    seed: int = typer.Option(51, help="Random seed for k-means."),
) -> None:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    image_paths = sorted((source_root / "images").glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No images found under {source_root / 'images'}")
    features = np.stack([image_features(path) for path in tqdm(image_paths, desc="Embedding images")])
    features = pca(normalize(features), dims=min(32, len(image_paths) - 1, features.shape[1]))
    labels = kmeans(features, clusters=clusters, seed=seed)
    selected_cluster = choose_validation_cluster(source_root, image_paths, labels, target_val_fraction)
    split_names = {
        path.stem: "val" if int(label) == selected_cluster else "train"
        for path, label in zip(image_paths, labels, strict=True)
    }

    for format_name in ("attributes", "classic"):
        build_format_split(source_root, output_root / format_name, split_names, classic=format_name == "classic")
    report = cluster_report(source_root, image_paths, labels, selected_cluster, split_names)
    report["classic_classes"] = CLASSIC_CLASSES
    report["attribute_names"] = BASKETBALL_DETECTION_ATTRIBUTES
    (output_root / "cluster_split_report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Selected validation cluster {} with {} images", selected_cluster, report["splits"]["val"]["images"])
    logger.info("Wrote {}", output_root)


def image_features(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    return np.concatenate([rgb.reshape(-1), hist])


def normalize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-6)


def pca(features: np.ndarray, dims: int) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:dims].T


def kmeans(features: np.ndarray, clusters: int, seed: int, iterations: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = [features[rng.integers(len(features))]]
    for _ in range(1, clusters):
        distances = squared_distance_to_nearest_center(features, np.stack(centers))
        probabilities = distances / distances.sum()
        centers.append(features[rng.choice(len(features), p=probabilities)])
    centers_array = np.stack(centers)
    labels = np.zeros(len(features), dtype=np.int64)
    for _ in range(iterations):
        distances = ((features[:, None, :] - centers_array[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster_index in range(clusters):
            members = features[labels == cluster_index]
            if len(members):
                centers_array[cluster_index] = members.mean(axis=0)
    return labels


def squared_distance_to_nearest_center(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.maximum(distances.min(axis=1), 1e-12)


def choose_validation_cluster(
    source_root: Path,
    image_paths: list[Path],
    labels: np.ndarray,
    target_val_fraction: float,
) -> int:
    target_size = len(image_paths) * target_val_fraction
    candidates = []
    for cluster_index in sorted(set(labels.tolist())):
        stems = [path.stem for path, label in zip(image_paths, labels, strict=True) if label == cluster_index]
        counts = attribute_counts(source_root, stems)
        non_empty_attrs = sum(count > 0 for count in counts.values())
        candidates.append(
            (
                -non_empty_attrs,
                abs(len(stems) - target_size),
                cluster_index,
            )
        )
    return min(candidates)[2]


def build_format_split(source_root: Path, output_root: Path, split_names: dict[str, str], classic: bool) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "detections").mkdir(parents=True, exist_ok=True)
    for stem, split in tqdm(split_names.items(), desc=f"Writing {output_root.name}"):
        shutil.copy2(source_root / "images" / f"{stem}.jpg", output_root / split / "images" / f"{stem}.jpg")
        source_npz = source_root / "detections" / f"{stem}.npz"
        target_npz = output_root / split / "detections" / f"{stem}.npz"
        if classic:
            write_classic_npz(source_npz, target_npz)
        else:
            shutil.copy2(source_npz, target_npz)


def write_classic_npz(source_npz: Path, target_npz: Path) -> None:
    data = np.load(source_npz)
    boxes = data["boxes_xywh"].astype(np.float32).reshape(-1, 4)
    category_names = data["category_names"].astype(str)
    attributes = data["attributes"].astype(np.bool_)
    attribute_names = data["attribute_names"].astype(str).tolist()
    classic_boxes = []
    classic_categories = []
    for box, category_name, row in zip(boxes, category_names, attributes, strict=True):
        emitted_special = False
        for attribute_name, enabled in zip(attribute_names, row, strict=True):
            if not enabled:
                continue
            special_class = ATTRIBUTE_TO_CLASS[attribute_name]
            classic_boxes.append(box)
            classic_categories.append(special_class)
            emitted_special = True
        if not emitted_special:
            classic_boxes.append(box)
            classic_categories.append(category_name)
    np.savez_compressed(
        target_npz,
        boxes_xywh=np.array(classic_boxes, dtype=np.float32).reshape(-1, 4),
        category_names=np.array(classic_categories, dtype=str),
    )


def cluster_report(
    source_root: Path,
    image_paths: list[Path],
    labels: np.ndarray,
    selected_cluster: int,
    split_names: dict[str, str],
) -> dict:
    clusters = {}
    for cluster_index in sorted(set(labels.tolist())):
        stems = [path.stem for path, label in zip(image_paths, labels, strict=True) if label == cluster_index]
        clusters[str(cluster_index)] = {
            "images": len(stems),
            "split": "val" if cluster_index == selected_cluster else "train",
            "attributes": attribute_counts(source_root, stems),
            "classes": class_counts(source_root, stems),
        }
    splits = {}
    for split in ("train", "val"):
        stems = [stem for stem, split_name in split_names.items() if split_name == split]
        splits[split] = {
            "images": len(stems),
            "attributes": attribute_counts(source_root, stems),
            "classes": class_counts(source_root, stems),
        }
    return {"selected_validation_cluster": selected_cluster, "clusters": clusters, "splits": splits}


def attribute_counts(source_root: Path, stems: list[str]) -> dict[str, int]:
    counts = dict.fromkeys(BASKETBALL_DETECTION_ATTRIBUTES, 0)
    for stem in stems:
        data = np.load(source_root / "detections" / f"{stem}.npz")
        attributes = data["attributes"].astype(np.bool_)
        names = data["attribute_names"].astype(str).tolist()
        for index, name in enumerate(names):
            counts[name] += int(attributes[:, index].sum())
    return counts


def class_counts(source_root: Path, stems: list[str]) -> dict[str, int]:
    counts = {}
    for stem in stems:
        data = np.load(source_root / "detections" / f"{stem}.npz")
        for name in data["category_names"].astype(str):
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    app()
