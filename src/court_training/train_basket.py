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
from court_training.constants import BACKBONE, IMAGE_SIZE, TTA_SCALES
from court_training.data import MaskDataset
from court_training.masks import MASK_NAMES
from court_training.model import DinoSegmenter, predict_multiscale

app = typer.Typer(help="Train a basketball court-mask segmenter.")

TRAIN_DATASETS = ("basketball_51", "borgo")
EVAL_DATASET = "e_bard_detection"
EVAL_IMAGES = {
    Path("000/all_00000001.jpg"),
    Path("001/all_00001067.jpg"),
    Path("001/all_00001105.jpg"),
    Path("001/all_00001190.jpg"),
    Path("001/all_00001266.jpg"),
    Path("001/all_00001289.jpg"),
    Path("001/all_00001331.jpg"),
    Path("001/all_00001726.jpg"),
    Path("002/all_00002178.jpg"),
    Path("002/all_00002544.jpg"),
    Path("003/all_00003983.jpg"),
    Path("004/all_00004638.jpg"),
}
DATA_ROOT_ARGUMENT = typer.Argument(help="Exported dataset root containing basket/.")
OUTPUT_DIR_ARGUMENT = typer.Argument(help="Directory where checkpoints are written.")


@app.command()
def main(
    data_root: Path = DATA_ROOT_ARGUMENT,
    output_dir: Path = OUTPUT_DIR_ARGUMENT,
    epochs: int = typer.Option(90, help="Training epochs."),
    batch_size: int = typer.Option(2, help="Training batch size."),
    learning_rate: float = typer.Option(3e-5, help="AdamW learning rate."),
    weight_decay: float = typer.Option(1e-4, help="AdamW weight decay."),
    num_workers: int = typer.Option(2, help="DataLoader workers."),
    seed: int = typer.Option(79, help="Random seed."),
    backbone: str = typer.Option(BACKBONE, help="timm backbone name."),
    image_width: int = typer.Option(IMAGE_SIZE[0], help="Training image width."),
    image_height: int = typer.Option(IMAGE_SIZE[1], help="Training image height."),
    eval_interval: int = typer.Option(1, help="Evaluate every N epochs, always including the final epoch."),
) -> None:
    set_seed(seed)
    basket_root = data_root.expanduser().resolve() / "basket"
    image_size = (image_width, image_height)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_items, eval_items = split_items(basket_root)
    train_data = MaskDataset(train_items, image_size, transform=BasketballAugment())
    eval_data = MaskDataset(eval_items, image_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=num_workers)

    device = training_device()
    model = DinoSegmenter(backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    logger.info("Training {} on {}", backbone, device)
    logger.info(
        "Train images: {} | Eval images: {} | size={} | augmentation=appearance | lr={} | weight_decay={}",
        len(train_data),
        len(eval_data),
        image_size,
        learning_rate,
        weight_decay,
    )
    train(
        model,
        optimizer,
        train_loader,
        eval_loader,
        device,
        output_dir,
        epochs,
        eval_interval,
    )


def split_items(basket_root: Path) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    train_items = []
    for dataset_name in TRAIN_DATASETS:
        train_items.extend(MaskDataset.items_for(basket_root, dataset_name))

    eval_items = []
    for image_path, mask_path in MaskDataset.items_for(basket_root, EVAL_DATASET):
        image_name = image_path.relative_to(basket_root / EVAL_DATASET / "images")
        if image_name in EVAL_IMAGES:
            eval_items.append((image_path, mask_path))
        else:
            train_items.append((image_path, mask_path))
    return train_items, eval_items


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    eval_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    output_dir: Path,
    epochs: int,
    eval_interval: int,
) -> None:
    best_miou = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        if epoch % eval_interval != 0 and epoch != epochs:
            logger.info("Epoch {}/{} train_loss={:.4f} lr={:.2e}", epoch, epochs, train_loss, current_lr(optimizer))
            continue

        eval_loss, miou, class_iou = evaluate(model, eval_loader, device)
        logger.info(
            "Epoch {}/{} train_loss={:.4f} lr={:.2e} eval_loss={:.4f} eval_mIoU={:.4f} class_iou={}",
            epoch,
            epochs,
            train_loss,
            current_lr(optimizer),
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


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


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
