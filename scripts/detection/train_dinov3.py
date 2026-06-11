import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import typer
from jaxtyping import Float
from loguru import logger
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from court_training.constants import IMAGE_MEAN, IMAGE_STD
from court_training.detection.data import (
    BASKETBALL_DETECTION_CLASSES,
    canonical_classes,
    class_id,
    load_split,
    parse_classes,
)
from court_training.detection.model import DinoDetector

app = typer.Typer(help="Fine-tune a DINOv3-backbone detector on exported basketball detections.")


@dataclass(frozen=True)
class DetectionMetrics:
    map50: float
    class_ap50: tuple[float, ...]


class DetectionDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_size: tuple[int, int],
        class_names: tuple[str, ...] = BASKETBALL_DETECTION_CLASSES,
    ) -> None:
        self.samples = load_split(root)
        self.image_size = image_size
        self.class_names = canonical_classes(class_names)
        self.selected_classes = set(self.class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self.samples[index]
        height, width = self.image_size
        image = Image.open(sample.image_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGE_MEAN) / IMAGE_STD
        boxes = []
        labels = []
        for box, name in zip(sample.boxes_xywh, sample.category_names, strict=True):
            if name not in self.selected_classes:
                continue
            boxes.append(box)
            labels.append(class_id(name, self.class_names))
        boxes_xywh = np.array(boxes, dtype=np.float32).reshape((-1, 4))
        return {
            "image": image_tensor,
            "boxes_xywh": torch.tensor(boxes_xywh, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@app.command()
def main(
    train_root: Path,
    val_root: Path,
    output_dir: Path,
    backbone: str = typer.Option("vit_large_patch16_dinov3", help="timm backbone name."),
    epochs: int = typer.Option(50, help="Training epochs."),
    batch_size: int = typer.Option(4, help="Training batch size."),
    learning_rate: float = typer.Option(3e-5, help="AdamW learning rate."),
    num_workers: int = typer.Option(2, help="DataLoader workers."),
    image_height: int = typer.Option(448, help="Training image height."),
    image_width: int = typer.Option(448, help="Training image width."),
    num_queries: int = typer.Option(64, help="Object queries."),
    seed: int = typer.Option(79, help="Random seed."),
    classes: str | None = typer.Option(None, help="Comma-separated class names to train and evaluate."),
) -> None:
    set_seed(seed)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = (image_height, image_width)
    class_names = parse_classes(classes)

    train_data = DetectionDataset(train_root.expanduser().resolve(), image_size, class_names)
    val_data = DetectionDataset(val_root.expanduser().resolve(), image_size, class_names)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_detection_batch,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_detection_batch,
    )

    device = training_device()
    model = DinoDetector(
        num_classes=len(class_names),
        num_queries=num_queries,
        backbone=backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    logger.info("Training {} detector on {}", backbone, device)
    logger.info("Train images: {} | Eval images: {}", len(train_data), len(val_data))
    best_map50 = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device, len(class_names))
        logger.info("Epoch {}/{} train_loss={:.4f}", epoch, epochs, train_loss)
        logger.info("Eval AP50={:.4f} class_ap50={}", metrics.map50, format_scores(metrics.class_ap50))
        if metrics.map50 >= best_map50:
            best_map50 = metrics.map50
            torch.save(model.state_dict(), output_dir / "best.pt")
            logger.info("Saved {} with AP50={:.4f}", output_dir / "best.pt", best_map50)


def train_epoch(
    model: DinoDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_images = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = to_device(batch, device)
        prediction = model(batch["image"])
        loss = detection_loss(prediction, batch["boxes_xywh"], batch["labels"], model.no_object_class)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = batch["image"].shape[0]
        total_loss += loss.item() * batch_size
        total_images += batch_size
    return total_loss / total_images


def detection_loss(
    prediction: dict[str, Tensor],
    boxes: list[Tensor],
    labels: list[Tensor],
    no_object_class: int,
) -> Tensor:
    logits = prediction["logits"]
    predicted_boxes = prediction["boxes_xywh"]
    target_classes = torch.full(logits.shape[:2], no_object_class, dtype=torch.long, device=logits.device)
    box_losses = []
    for batch_index, target_boxes in enumerate(boxes):
        target_labels = labels[batch_index]
        matches = greedy_matches(predicted_boxes[batch_index], logits[batch_index], target_boxes, target_labels)
        for query_index, target_index in matches:
            target_classes[batch_index, query_index] = target_labels[target_index]
            box_losses.append(F.l1_loss(predicted_boxes[batch_index, query_index], target_boxes[target_index]))
    class_loss = F.cross_entropy(logits.flatten(0, 1), target_classes.flatten(0, 1))
    if not box_losses:
        return class_loss
    return class_loss + 5 * torch.stack(box_losses).mean()


def greedy_matches(
    predicted_boxes: Float[Tensor, "Q 4"],
    logits: Float[Tensor, "Q C"],
    target_boxes: Float[Tensor, "N 4"],
    target_labels: Tensor,
) -> list[tuple[int, int]]:
    available = set(range(predicted_boxes.shape[0]))
    matches = []
    probabilities = logits.softmax(dim=-1)
    for target_index, target_box in enumerate(target_boxes):
        if not available:
            break
        label = target_labels[target_index]
        costs = []
        for query_index in available:
            box_cost = F.l1_loss(predicted_boxes[query_index], target_box, reduction="sum")
            class_cost = -probabilities[query_index, label]
            costs.append((float(box_cost + class_cost), query_index))
        _, query_index = min(costs)
        available.remove(query_index)
        matches.append((query_index, target_index))
    return matches


@torch.inference_mode()
def evaluate(
    model: DinoDetector,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> DetectionMetrics:
    model.eval()
    predictions_by_class = {index: [] for index in range(num_classes)}
    targets_by_class = {index: {} for index in range(num_classes)}
    image_offset = 0
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = to_device(batch, device)
        output = model(batch["image"])
        scores, labels = output["logits"].softmax(dim=-1)[..., :-1].max(dim=-1)
        for batch_index in range(batch["image"].shape[0]):
            image_id = image_offset + batch_index
            for class_index in targets_by_class:
                mask = batch["labels"][batch_index] == class_index
                targets_by_class[class_index][image_id] = batch["boxes_xywh"][batch_index][mask]
            for box, label, score in zip(
                output["boxes_xywh"][batch_index],
                labels[batch_index],
                scores[batch_index],
                strict=True,
            ):
                predictions_by_class[int(label)].append((image_id, float(score), box.detach().cpu()))
        image_offset += batch["image"].shape[0]

    ap50 = []
    for class_index in range(num_classes):
        ap50.append(average_precision_50(predictions_by_class[class_index], targets_by_class[class_index]))
    valid_ap50 = [score for score in ap50 if not np.isnan(score)]
    return DetectionMetrics(map50=float(np.mean(valid_ap50)), class_ap50=tuple(ap50))


def average_precision_50(predictions: list[tuple[int, float, Tensor]], targets: dict[int, Tensor]) -> float:
    total_targets = sum(len(boxes) for boxes in targets.values())
    if total_targets == 0:
        return float("nan")
    predictions = sorted(predictions, key=lambda item: item[1], reverse=True)
    matched = {image_id: set() for image_id in targets}
    true_positive = []
    false_positive = []
    for image_id, _, box in predictions:
        target_boxes = targets[image_id].cpu()
        ious = box_iou_xywh(box.unsqueeze(0), target_boxes).squeeze(0)
        best_iou, target_index = ious.max(dim=0) if len(ious) else (torch.tensor(0.0), torch.tensor(-1))
        if best_iou >= 0.5 and int(target_index) not in matched[image_id]:
            matched[image_id].add(int(target_index))
            true_positive.append(1.0)
            false_positive.append(0.0)
        else:
            true_positive.append(0.0)
            false_positive.append(1.0)
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recall = tp / total_targets
    precision = tp / np.maximum(tp + fp, 1e-9)
    return float(np.trapezoid(precision, recall))


def box_iou_xywh(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    if boxes_b.numel() == 0:
        return torch.zeros((boxes_a.shape[0], 0))
    a = xywh_to_xyxy(boxes_a)
    b = xywh_to_xyxy(boxes_b)
    top_left = torch.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(dim=-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(dim=-1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp_min(1e-9)


def xywh_to_xyxy(boxes: Tensor) -> Tensor:
    center = boxes[..., :2]
    size = boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def collate_detection_batch(batch: list[dict[str, Tensor]]) -> dict[str, Tensor | list[Tensor]]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "boxes_xywh": [item["boxes_xywh"] for item in batch],
        "labels": [item["labels"] for item in batch],
    }


def to_device(batch: dict[str, Tensor | list[Tensor]], device: torch.device) -> dict[str, Tensor | list[Tensor]]:
    output = {}
    for key, value in batch.items():
        if isinstance(value, list):
            output[key] = [tensor.to(device=device) for tensor in value]
        else:
            output[key] = value.to(device=device)
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_scores(scores: tuple[float, ...]) -> str:
    return "[" + ", ".join("nan" if np.isnan(score) else f"{score:.3f}" for score in scores) + "]"


if __name__ == "__main__":
    app()
