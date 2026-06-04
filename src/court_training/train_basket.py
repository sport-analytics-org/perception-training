import random
from pathlib import Path

import numpy as np
import torch
import typer
from loguru import logger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from court_training.augment import BasketballAugment
from court_training.constants import BACKBONE, MASK_NAMES, TTA_SCALES
from court_training.data import MaskDataset
from court_training.model import DinoSegmenter, predict_multiscale

app = typer.Typer(help="Train a basketball court-mask segmenter.")

TRAIN_DATASETS = ("basketball_51", "borgo")
EVAL_DATASET = "e_bard_detection"
DATA_ROOT_ARGUMENT = typer.Argument(help="Exported dataset root containing basket/.")
OUTPUT_DIR_ARGUMENT = typer.Argument(help="Directory where checkpoints are written.")


@app.command()
def main(
    data_root: Path = DATA_ROOT_ARGUMENT,
    output_dir: Path = OUTPUT_DIR_ARGUMENT,
    epochs: int = typer.Option(140, help="Training epochs."),
    batch_size: int = typer.Option(2, help="Training batch size."),
    learning_rate: float = typer.Option(3e-5, help="AdamW learning rate."),
    num_workers: int = typer.Option(2, help="DataLoader workers."),
    seed: int = typer.Option(79, help="Random seed."),
    crop_cutout: bool = typer.Option(True, help="Train with random crop and image cutout augmentation."),
) -> None:
    set_seed(seed)
    basket_root = data_root.expanduser().resolve() / "basket"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = MaskDataset(basket_root, TRAIN_DATASETS, transform=BasketballAugment(crop_cutout))
    eval_data = MaskDataset(basket_root, (EVAL_DATASET,))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=num_workers)

    device = training_device()
    model = DinoSegmenter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    logger.info("Training {} on {}", BACKBONE, device)
    logger.info("Train images: {} | Eval images: {} | crop_cutout={}", len(train_data), len(eval_data), crop_cutout)
    train(model, optimizer, train_loader, eval_loader, device, output_dir, epochs)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    eval_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    output_dir: Path,
    epochs: int,
) -> None:
    best_miou = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        eval_loss, miou, class_iou = evaluate(model, eval_loader, device)
        logger.info(
            "Epoch {}/{} train_loss={:.4f} eval_loss={:.4f} eval_mIoU={:.4f} class_iou={}",
            epoch,
            epochs,
            train_loss,
            eval_loss,
            miou,
            format_scores(class_iou),
        )
        if miou >= best_miou:
            best_miou = miou
            torch.save(model.state_dict(), output_dir / "best.pt")
            logger.info("Saved {} with eval_mIoU={:.4f}", output_dir / "best.pt", best_miou)


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = segmentation_loss(model(images), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.shape[0]
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    intersection = torch.zeros(len(MASK_NAMES), device=device)
    union = torch.zeros(len(MASK_NAMES), device=device)

    for images, masks in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = predict_multiscale(model, images, TTA_SCALES)
        total_loss += segmentation_loss(logits, masks).item() * images.shape[0]

        predictions = logits.sigmoid() > 0.5
        targets = masks > 0.5
        intersection += (predictions & targets).sum(dim=(0, 2, 3))
        union += (predictions | targets).sum(dim=(0, 2, 3))

    class_iou = intersection / union.clamp_min(1)
    miou = class_iou[union > 0].mean().item()
    return total_loss / len(loader.dataset), miou, class_iou


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    probabilities = logits.sigmoid()
    numerator = 2 * (probabilities * masks).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3))
    dice = 1 - ((numerator + 1) / (denominator + 1)).mean()
    return bce + dice


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


def format_scores(scores: torch.Tensor) -> str:
    return "[" + ", ".join(f"{score:.3f}" for score in scores.tolist()) + "]"
