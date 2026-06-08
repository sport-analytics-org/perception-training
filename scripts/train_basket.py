import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import typer
from jaxtyping import Float, UInt8
from loguru import logger
from sportanalytics import NbaCourt
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from court_training.augment import CourtAugment
from court_training.constants import TTA_SCALES
from court_training.dataset import MaskDataset
from court_training.model import CourtSegmenter

app = typer.Typer(help="Train a basketball court-mask segmenter.")


BASKETBALL_MASK_NAMES = tuple(NbaCourt.areas())
BASKETBALL_KEYPOINT_NAMES = tuple(NbaCourt.keypoints())
TRAIN_ROOT_ARGUMENT = typer.Argument(help="Flat exported training dataset root.")
VAL_ROOT_ARGUMENT = typer.Argument(help="Flat exported validation dataset root.")
OUTPUT_DIR_ARGUMENT = typer.Argument(help="Directory where checkpoints are written.")


@dataclass(frozen=True)
class EvalMetrics:
    miou: float
    class_iou: tuple[float, ...]
    keypoint_error: float
    visibility_accuracy: float


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
    image_height: int = typer.Option(360, help="Training image height."),
    image_width: int = typer.Option(480, help="Training image width."),
    crop_cutout: bool = typer.Option(True, help="Train with random crop and image cutout augmentation."),
    keypoint_loss_weight: float = typer.Option(0.2, help="Weight for keypoint coordinate and visibility losses."),
) -> None:
    set_seed(seed)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = (image_height, image_width)

    train_data = MaskDataset(
        train_root.expanduser().resolve(),
        load_mask=load_mask,
        image_size=image_size,
        transform=CourtAugment(BASKETBALL_MASK_NAMES, image_size, BASKETBALL_KEYPOINT_NAMES, crop_cutout),
    )
    eval_data = MaskDataset(val_root.expanduser().resolve(), load_mask=load_mask, image_size=image_size)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    eval_loader = DataLoader(
        eval_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = training_device()
    num_masks = len(BASKETBALL_MASK_NAMES)
    num_keypoints = len(BASKETBALL_KEYPOINT_NAMES)
    model = CourtSegmenter(
        num_masks=num_masks,
        num_keypoints=num_keypoints,
        mask_names=BASKETBALL_MASK_NAMES,
        keypoint_names=BASKETBALL_KEYPOINT_NAMES,
        backbone=backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    logger.info("Training {} on {}", backbone, device)
    logger.info("Train images: {} | Eval images: {} | crop_cutout={}", len(train_data), len(eval_data), crop_cutout)
    logger.info("Masks: {}", num_masks)
    logger.info("Keypoints: {}", num_keypoints)
    train(model, optimizer, train_loader, eval_loader, device, output_dir, epochs, num_masks, keypoint_loss_weight)


def train(
    model: CourtSegmenter,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    num_masks: int,
    keypoint_loss_weight: float,
) -> None:
    best_miou = 0.0
    for epoch in range(1, epochs + 1):
        train_seg_loss, train_keypoint_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            keypoint_loss_weight,
        )
        metrics = evaluate(model, eval_loader, device, num_masks)
        logger.info("Epoch {}/{} train_seg_loss={:.4f}", epoch, epochs, train_seg_loss)
        logger.info("Epoch {}/{} train_keypoint_loss={:.4f}", epoch, epochs, train_keypoint_loss)
        logger.info("Eval mIoU={:.4f}", metrics.miou)
        logger.info("Eval class_iou={}", format_scores(metrics.class_iou))
        logger.info("Keypoints error={:.4f}", metrics.keypoint_error)
        logger.info("Visibility acc={:.3f}", metrics.visibility_accuracy)
        if metrics.miou >= best_miou:
            best_miou = metrics.miou
            torch.save(model.state_dict(), output_dir / "best.pt")
            logger.info("Saved {} with eval_mIoU={:.4f}", output_dir / "best.pt", best_miou)


def train_epoch(
    model: CourtSegmenter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    keypoint_loss_weight: float,
) -> tuple[float, float]:
    model.train()
    total_segmentation_loss = 0.0
    total_keypoint_loss = 0.0
    total_images = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        tensors = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(tensors["images"])
        mask_loss = segmentation_loss(prediction["masks"], tensors["masks"])
        point_loss = keypoint_loss(
            prediction["keypoints"],
            prediction["visibility"],
            prediction["heatmaps"],
            tensors["keypoints"],
            tensors["visibility"],
        )
        loss = mask_loss + keypoint_loss_weight * point_loss
        loss.backward()
        optimizer.step()
        batch_size = tensors["images"].shape[0]
        total_segmentation_loss += mask_loss.item() * batch_size
        total_keypoint_loss += point_loss.item() * batch_size
        total_images += batch_size
    return total_segmentation_loss / total_images, total_keypoint_loss / total_images


@torch.inference_mode()
def evaluate(
    model: CourtSegmenter,
    loader: DataLoader,
    device: torch.device,
    num_masks: int,
) -> EvalMetrics:
    model.eval()
    intersection = torch.zeros(num_masks, device=device)
    union = torch.zeros(num_masks, device=device)
    total_keypoint_error = 0.0
    total_visible_keypoints = 0
    total_visibility_correct = 0
    total_visibility = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        tensors = to_device(batch, device)
        prediction = model.predict(tensors["images"], TTA_SCALES)

        visible = tensors["visibility"] > 0.5
        error = (prediction["keypoints"][visible] - tensors["keypoints"][visible]).norm(dim=-1)
        total_keypoint_error += error.sum().item()
        total_visible_keypoints += int(visible.sum().item())
        total_visibility_correct += int(((prediction["visibility"].sigmoid() > 0.5) == visible).sum().item())
        total_visibility += tensors["visibility"].numel()

        predictions = prediction["masks"].sigmoid() > 0.5
        targets = tensors["masks"] > 0.5
        intersection += (predictions & targets).sum(dim=(0, 2, 3))
        union += (predictions | targets).sum(dim=(0, 2, 3))

    class_iou = intersection / union.clamp_min(1)
    miou = class_iou[union > 0].mean().item()
    keypoint_error = total_keypoint_error / max(total_visible_keypoints, 1)
    visibility_accuracy = total_visibility_correct / total_visibility
    return EvalMetrics(
        miou=miou,
        class_iou=tuple(class_iou.tolist()),
        keypoint_error=keypoint_error,
        visibility_accuracy=visibility_accuracy,
    )


def segmentation_loss(logits: Float[Tensor, "B N H W"], masks: Float[Tensor, "B N H W"]) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    probabilities = logits.sigmoid()
    numerator = 2 * (probabilities * masks).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3))
    dice = 1 - ((numerator + 1) / (denominator + 1)).mean()
    return bce + dice


def keypoint_loss(
    predicted_keypoints: Float[Tensor, "B K 2"],
    visibility_logits: Float[Tensor, "B K"],
    heatmaps: Float[Tensor, "B K H W"],
    keypoints: Float[Tensor, "B K 2"],
    visibility: Float[Tensor, "B K"],
) -> Tensor:
    coordinate_errors = F.smooth_l1_loss(predicted_keypoints, keypoints, reduction="none").sum(dim=-1)
    coordinate_loss = (coordinate_errors * visibility).sum() / visibility.sum().clamp_min(1)
    heatmap_targets = gaussian_heatmaps(keypoints, visibility, heatmaps.shape[-2:])
    heatmap_weights = 0.1 + 20 * heatmap_targets
    heatmap_bce = F.binary_cross_entropy_with_logits(heatmaps, heatmap_targets, reduction="none")
    heatmap_loss = (heatmap_bce * heatmap_weights).mean()
    objectness_loss = F.binary_cross_entropy_with_logits(visibility_logits, visibility)
    return 10 * coordinate_loss + 0.1 * heatmap_loss + objectness_loss


def gaussian_heatmaps(
    keypoints: Float[Tensor, "B K 2"],
    visibility: Float[Tensor, "B K"],
    size: tuple[int, int],
    sigma: float = 1.5,
) -> Float[Tensor, "B K H W"]:
    height, width = size
    x = torch.linspace(0, width - 1, width, device=keypoints.device, dtype=keypoints.dtype)
    y = torch.linspace(0, height - 1, height, device=keypoints.device, dtype=keypoints.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    point_x = keypoints[..., 0, None, None] * (width - 1)
    point_y = keypoints[..., 1, None, None] * (height - 1)
    distance = (grid_x - point_x).square() + (grid_y - point_y).square()
    heatmaps = torch.exp(-distance / (2 * sigma**2))
    return heatmaps * visibility[..., None, None]


def load_mask(bitfield: UInt8[np.ndarray, "H W"]) -> Float[np.ndarray, "H W N"]:
    masks = [(bitfield & (1 << bit)) > 0 for bit in range(len(BASKETBALL_MASK_NAMES))]
    return np.stack(masks, axis=-1).astype(np.float32)


def to_device(
    batch: dict[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    names = {
        "image": "images",
        "mask": "masks",
        "keypoints": "keypoints",
        "keypoint_visibility": "visibility",
    }
    return {output: batch[input_].to(device=device) for input_, output in names.items()}


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
    return "[" + ", ".join(f"{score:.3f}" for score in scores) + "]"


if __name__ == "__main__":
    app()
