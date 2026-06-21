import json
import random
from pathlib import Path

import numpy as np
import torch
import typer
from loguru import logger
from torch.utils.data import DataLoader, Subset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_convert
from tqdm import tqdm

from perception_training import image_io
from perception_training.augment import CourtAugment
from perception_training.dataset import BASKETBALL_DETECTION_CLASSES, CourtDataset, collate
from perception_training.detection import metrics
from perception_training.detection.model import CourtDetector

app = typer.Typer(help="Fine-tune RF-DETR Large on basketball detections.")

TRAIN_ROOT_ARGUMENT = typer.Argument(help="Flat exported training dataset root.")
OUTPUT_DIR_ARGUMENT = typer.Argument(help="Directory where checkpoints are written.")
VAL_ROOT_OPTION = typer.Option(None, help="Optional flat exported validation dataset root.")

CLIP_MAX_NORM = 0.1


@app.command()
def main(
    train_root: Path = TRAIN_ROOT_ARGUMENT,
    output_dir: Path = OUTPUT_DIR_ARGUMENT,
    val_root: Path | None = VAL_ROOT_OPTION,
    epochs: int = typer.Option(6, help="Training epochs."),
    batch_size: int = typer.Option(8, help="Training batch size."),
    learning_rate: float = typer.Option(1e-4, help="Detector learning rate."),
    lr_encoder: float = typer.Option(1.5e-4, help="Backbone encoder learning rate."),
    lr_drop: int = typer.Option(5, help="Epoch after which the learning rate drops by 10x."),
    warmup_epochs: float = typer.Option(0.5, help="Linear warmup duration in epochs."),
    weight_decay: float = typer.Option(1e-4, help="Weight decay."),
    num_workers: int = typer.Option(8, help="DataLoader workers."),
    resolution: int = typer.Option(640, help="Square training resolution."),
    val_max_samples: int = typer.Option(800, help="Use at most this many validation images during training."),
    seed: int = typer.Option(51, help="Random seed."),
) -> None:
    set_seed(seed)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(output_dir, resolution)
    image_size = (resolution, resolution)

    train_data = CourtDataset(
        train_root.expanduser().resolve(),
        image_size,
        load_bbox=True,
        transform=CourtAugment(image_size=image_size, crop_cutout=False),
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )
    val_loader = None
    val_count = 0
    if val_root is not None:
        val_data = CourtDataset(val_root.expanduser().resolve(), image_size, load_bbox=True)
        if len(val_data) > val_max_samples:
            generator = np.random.default_rng(seed)
            indexes = generator.choice(len(val_data), size=val_max_samples, replace=False)
            val_data = Subset(val_data, sorted(indexes.tolist()))
        val_count = len(val_data)
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        )

    device = training_device()
    model = CourtDetector(BASKETBALL_DETECTION_CLASSES, image_size).to(device)
    optimizer = torch.optim.AdamW(model.param_groups(learning_rate, lr_encoder, weight_decay))
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    logger.info("Training RF-DETR Large on {}", device)
    logger.info("Train images: {} | Eval images: {}", len(train_data), val_count)
    logger.info("Classes: {}", BASKETBALL_DETECTION_CLASSES)
    warmup_steps = max(1, round(warmup_epochs * len(train_loader)))
    train(model, optimizer, train_loader, val_loader, device, output_dir, epochs, lr_drop, warmup_steps)


def train(
    model: CourtDetector,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    eval_loader: DataLoader | None,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    lr_drop: int,
    warmup_steps: int,
) -> None:
    best_map = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, lr_drop, warmup_steps)
        logger.info("Epoch {}/{} train_loss={:.4f}", epoch, epochs, train_loss)
        if eval_loader is not None:
            eval_metrics = evaluate(model, eval_loader, device)
            map50_95 = eval_metrics["map50_95"]
            map50 = eval_metrics["map50"]
            map75 = eval_metrics["map75"]
            per_class_map = eval_metrics["per_class_map"]
            logger.info("Eval mAP50_95={:.4f} mAP50={:.4f} mAP75={:.4f}", map50_95, map50, map75)
            logger.info("Eval per-class mAP={}", per_class_map)
            if map50_95 >= best_map:
                best_map = map50_95
                torch.save(model.state_dict(), output_dir / "best.pt")
                logger.info("Saved {} with eval_mAP50_95={:.4f}", output_dir / "best.pt", best_map)
    torch.save(model.state_dict(), output_dir / "final.pt")
    logger.info("Saved final checkpoint to {}", output_dir / "final.pt")


def train_epoch(
    model: CourtDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    lr_drop: int,
    warmup_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_images = 0
    for batch_index, (images, targets) in enumerate(tqdm(loader, desc="Training", leave=False)):
        step = (epoch - 1) * len(loader) + batch_index
        set_lr(optimizer, lr_factor(step, warmup_steps, epoch, lr_drop))
        images = images.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = model.loss(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
        optimizer.step()
        total_loss += loss.item() * len(targets)
        total_images += len(targets)
    return total_loss / total_images


def lr_factor(step: int, warmup_steps: int, epoch: int, lr_drop: int) -> float:
    warmup = min(1.0, (step + 1) / warmup_steps)
    drop = 0.1 if epoch > lr_drop else 1.0
    return warmup * drop


def set_lr(optimizer: torch.optim.Optimizer, factor: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor


@torch.inference_mode()
def evaluate(model: CourtDetector, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    metric = MeanAveragePrecision(box_format="xywh", class_metrics=True)
    for images, targets in tqdm(loader, desc="Evaluating", leave=False):
        batch_images = [image_io.tensor2image(image) for image in images.to(device)]
        detections = model.predict(batch_images)
        predictions = [detections_to_torchmetrics(detection) for detection in detections]
        # TorchMetrics stores targets until compute(); clone off DataLoader shared-memory storage.
        ground_truth = []
        for target in targets:
            boxes = target["boxes_xywh"].clone()
            labels = target["labels"].clone()
            ground_truth.append({"boxes": boxes, "labels": labels})
        metric.update(predictions, ground_truth)
    return metrics.summarize(metric, model.class_names)


def detections_to_torchmetrics(detections) -> dict[str, torch.Tensor]:
    boxes = torch.from_numpy(detections["boxes"]).to(dtype=torch.float32)
    return {
        "boxes": box_convert(boxes, "xyxy", "xywh"),
        "scores": torch.from_numpy(detections["scores"]).to(dtype=torch.float32),
        "labels": torch.from_numpy(detections["labels"]).to(dtype=torch.long),
    }


def write_metadata(output_dir: Path, resolution: int) -> None:
    """Sidecar read by CourtDetector.load."""
    metadata = {
        "architecture": "RF-DETR Large",
        "image_size": {"height": resolution, "width": resolution},
        "classes": list(BASKETBALL_DETECTION_CLASSES),
    }
    (output_dir / "args.json").write_text(json.dumps(metadata, indent=2) + "\n")


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


if __name__ == "__main__":
    app()
