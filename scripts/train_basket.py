import random
from pathlib import Path

import numpy as np
import torch
import typer
from jaxtyping import Float, UInt8
from loguru import logger
from sportanalytics import NbaCourt
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from court_training.augment import CourtAugment
from court_training.constants import IMAGE_MEAN, IMAGE_STD, TTA_SCALES
from court_training.dataset import MaskDataset
from court_training.model import DinoSegmenter, resize_images

app = typer.Typer(help="Train a basketball court-mask segmenter.")


def side_pairs(names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    index_by_name = {name: index for index, name in enumerate(names)}
    left_names = [name for name in names if name.startswith("left_")]
    pairs = []
    for left_name in left_names:
        right_name = f"right_{left_name.removeprefix('left_')}"
        pairs.append((index_by_name[left_name], index_by_name[right_name]))
    return tuple(pairs)


BASKETBALL_MASK_NAMES = tuple(NbaCourt.areas())
BASKETBALL_LEFT_RIGHT_PAIRS = side_pairs(BASKETBALL_MASK_NAMES)
TRAIN_ROOT_ARGUMENT = typer.Argument(help="Flat exported training dataset root.")
VAL_ROOT_ARGUMENT = typer.Argument(help="Flat exported validation dataset root.")
OUTPUT_DIR_ARGUMENT = typer.Argument(help="Directory where checkpoints are written.")


@app.command()
def main(
    train_root: Path = TRAIN_ROOT_ARGUMENT,
    val_root: Path = VAL_ROOT_ARGUMENT,
    output_dir: Path = OUTPUT_DIR_ARGUMENT,
    backbone: str = typer.Option("vit_large_patch16_dinov3", help="timm backbone name."),
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
    train(model, optimizer, train_loader, eval_loader, device, output_dir, epochs, len(BASKETBALL_MASK_NAMES))


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    num_masks: int,
) -> None:
    best_miou = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        eval_loss, miou, class_iou = evaluate(model, eval_loader, device, num_masks)
        scores = format_scores(class_iou)
        message = "Epoch {}/{} train_loss={:.4f} eval_loss={:.4f} eval_mIoU={:.4f} class_iou={}"
        logger.info(message, epoch, epochs, train_loss, eval_loss, miou, scores)
        if miou >= best_miou:
            best_miou = miou
            torch.save(model.state_dict(), output_dir / "best.pt")
            logger.info("Saved {} with eval_mIoU={:.4f}", output_dir / "best.pt", best_miou)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_images = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        images, masks = batch_to_tensors(batch, device)
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
    loader: DataLoader,
    device: torch.device,
    num_masks: int,
) -> tuple[float, float, Tensor]:
    model.eval()
    total_loss = 0.0
    total_images = 0
    intersection = torch.zeros(num_masks, device=device)
    union = torch.zeros(num_masks, device=device)

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images, masks = batch_to_tensors(batch, device)
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


def segmentation_loss(logits: Float[Tensor, "B N H W"], masks: Float[Tensor, "B N H W"]) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    probabilities = logits.sigmoid()
    numerator = 2 * (probabilities * masks).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3))
    dice = 1 - ((numerator + 1) / (denominator + 1)).mean()
    return bce + dice


def load_mask(bitfield: UInt8[np.ndarray, "H W"]) -> Float[np.ndarray, "H W N"]:
    masks = [(bitfield & (1 << bit)) > 0 for bit in range(len(BASKETBALL_MASK_NAMES))]
    return np.stack(masks, axis=-1).astype(np.float32)


def batch_to_tensors(
    batch: dict[str, Tensor],
    device: torch.device,
) -> tuple[Float[Tensor, "B 3 H W"], Float[Tensor, "B N H W"]]:
    image_batch = batch["image"].to(device=device, dtype=torch.float32)
    mask_batch = batch["mask"].to(device=device, dtype=torch.float32)
    images = image_batch.permute(0, 3, 1, 2) / 255.0
    masks = mask_batch.permute(0, 3, 1, 2)
    images = (images - IMAGE_MEAN.to(device)) / IMAGE_STD.to(device)
    return images, masks


def predict_multiscale(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    scales: tuple[float, ...],
) -> Float[Tensor, "B N H W"]:
    output_size = images.shape[-2:]
    logits_by_scale = []
    for scale in scales:
        logits = model(resize_images(images, scale))
        resized_logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        logits_by_scale.append(resized_logits)
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


def format_scores(scores: Tensor) -> str:
    return "[" + ", ".join(f"{score:.3f}" for score in scores.tolist()) + "]"


if __name__ == "__main__":
    app()
