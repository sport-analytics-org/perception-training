import random
from pathlib import Path

import numpy as np
import torch
import typer
from jaxtyping import Float, UInt8
from loguru import logger
from sportanalytics import BasketCourt, NbaCourt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from court_training.augment import CourtAugment
from court_training.constants import TTA_SCALES
from court_training.dataset import MaskDataset, MaskSample
from court_training.model import DinoSegmenter, resize_images

app = typer.Typer(help="Train a basketball court-mask segmenter.")

BACKBONE = "vit_large_patch16_dinov3"
BASKETBALL_AREA_ORDER = ("court", "3pt_area", "painted_area")


def court_mask_names(court: BasketCourt, area_order: tuple[str, ...]) -> tuple[str, ...]:
    area_names = set(court.areas())
    names = []
    for area in area_order:
        for side in ("left", "right"):
            name = f"{side}_{area}"
            if name in area_names:
                names.append(name)
    return tuple(names)


def side_pairs(names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    by_name = {name: index for index, name in enumerate(names)}
    pairs = []
    for left_name, left_index in by_name.items():
        if left_name.startswith("left_"):
            right_name = f"right_{left_name.removeprefix('left_')}"
            if right_name in by_name:
                pairs.append((left_index, by_name[right_name]))
    return tuple(pairs)


BASKETBALL_MASK_NAMES = court_mask_names(NbaCourt, BASKETBALL_AREA_ORDER)
BASKETBALL_LEFT_RIGHT_PAIRS = side_pairs(BASKETBALL_MASK_NAMES)
OUTPUT_DIR_ARGUMENT = typer.Argument(help="Directory where checkpoints are written.")
TRAIN_ROOT_ARGUMENT = typer.Argument(help="Flat exported training dataset root.")
VAL_ROOT_ARGUMENT = typer.Argument(help="Flat exported validation dataset root.")


@app.command()
def main(
    train_root: Path = TRAIN_ROOT_ARGUMENT,
    val_root: Path = VAL_ROOT_ARGUMENT,
    output_dir: Path = OUTPUT_DIR_ARGUMENT,
    backbone: str = typer.Option(BACKBONE, help="timm backbone name."),
    epochs: int = typer.Option(140, help="Training epochs."),
    batch_size: int = typer.Option(2, help="Training batch size."),
    learning_rate: float = typer.Option(3e-5, help="AdamW learning rate."),
    num_workers: int = typer.Option(2, help="DataLoader workers."),
    seed: int = typer.Option(79, help="Random seed."),
    crop_cutout: bool = typer.Option(True, help="Train with random crop and image cutout augmentation."),
) -> None:
    set_seed(seed)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = MaskDataset(
        train_root.expanduser().resolve(),
        load_mask=load_mask,
        transform=CourtAugment(BASKETBALL_LEFT_RIGHT_PAIRS, crop_cutout),
    )
    eval_data = MaskDataset(val_root.expanduser().resolve(), load_mask=load_mask)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=num_workers)

    device = training_device()
    model = DinoSegmenter(
        num_masks=len(BASKETBALL_MASK_NAMES),
        left_right_pairs=BASKETBALL_LEFT_RIGHT_PAIRS,
        backbone=backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    logger.info("Training {} on {}", backbone, device)
    logger.info("Train images: {} | Eval images: {} | crop_cutout={}", len(train_data), len(eval_data), crop_cutout)
    train(model, optimizer, train_loader, eval_loader, device, output_dir, epochs, BASKETBALL_MASK_NAMES)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader[MaskSample],
    eval_loader: DataLoader[MaskSample],
    device: torch.device,
    output_dir: Path,
    epochs: int,
    mask_names: tuple[str, ...],
) -> None:
    best_miou = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        eval_loss, miou, class_iou = evaluate(model, eval_loader, device, mask_names)
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
    loader: DataLoader[MaskSample],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_images = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = segmentation_loss(model(images), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.shape[0]
        total_images += images.shape[0]
    return total_loss / total_images


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[MaskSample],
    device: torch.device,
    mask_names: tuple[str, ...],
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_images = 0
    intersection = torch.zeros(len(mask_names), device=device)
    union = torch.zeros(len(mask_names), device=device)

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = predict_multiscale(model, images, TTA_SCALES)
        total_loss += segmentation_loss(logits, masks).item() * images.shape[0]

        predictions = logits.sigmoid() > 0.5
        targets = masks > 0.5
        intersection += (predictions & targets).sum(dim=(0, 2, 3))
        union += (predictions | targets).sum(dim=(0, 2, 3))
        total_images += images.shape[0]

    class_iou = intersection / union.clamp_min(1)
    miou = class_iou[union > 0].mean().item()
    return total_loss / total_images, miou, class_iou


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    probabilities = logits.sigmoid()
    numerator = 2 * (probabilities * masks).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3))
    dice = 1 - ((numerator + 1) / (denominator + 1)).mean()
    return bce + dice


def load_mask(bitfield: UInt8[np.ndarray, "H W"]) -> Float[torch.Tensor, "N H W"]:
    masks = [(bitfield & (1 << bit)) > 0 for bit in range(len(BASKETBALL_MASK_NAMES))]
    return torch.from_numpy(np.stack(masks).astype(np.float32))


def predict_multiscale(model: nn.Module, images: torch.Tensor, scales: tuple[float, ...]) -> torch.Tensor:
    output_size = images.shape[-2:]
    logits_by_scale = []
    for scale in scales:
        logits = model(resize_images(images, scale))
        logits_by_scale.append(F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False))
    return torch.stack(logits_by_scale).mean(dim=0)


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


if __name__ == "__main__":
    app()
