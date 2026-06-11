import json
from pathlib import Path

import typer
from rfdetr import RFDETRLarge

from court_training.detection import data

app = typer.Typer(help="Fine-tune RF-DETR Large on basketball detections.")
DEFAULT_CLASSES = "ball,player,referee"
COURT_AUGMENTATION = {
    "HorizontalFlip": {"p": 0.5},
    "Affine": {
        "scale": (0.85, 1.2),
        "translate_percent": (-0.08, 0.08),
        "rotate": (-6, 6),
        "shear": (-3, 3),
        "p": 0.45,
    },
    "ColorJitter": {
        "brightness": 0.18,
        "contrast": 0.18,
        "saturation": 0.12,
        "hue": 0.03,
        "p": 0.5,
    },
    "GaussianBlur": {"blur_limit": 3, "p": 0.12},
}


@app.command()
def main(
    train_root: Path,
    val_root: Path,
    output_dir: Path,
    epochs: int = typer.Option(6, help="Training epochs."),
    batch_size: int = typer.Option(8, help="Training batch size."),
    learning_rate: float = typer.Option(1e-4, help="RF-DETR learning rate."),
    lr_encoder: float = typer.Option(1.5e-4, help="RF-DETR encoder learning rate."),
    lr_drop: int = typer.Option(5, help="Epoch for step LR decay."),
    warmup_epochs: float = typer.Option(0.5, help="Warmup epochs."),
    weight_decay: float = typer.Option(1e-4, help="Weight decay."),
    num_workers: int = typer.Option(8, help="Dataloader workers."),
    resolution: int = typer.Option(704, help="Square training/inference resolution."),
    val_max_samples: int = typer.Option(800, help="Use at most this many validation images during training."),
    seed: int = typer.Option(51, help="Training seed."),
    train_box_scale: str = typer.Option("ball=1.35", help="Comma-separated class=scale training box overrides."),
    classes: str = typer.Option(DEFAULT_CLASSES, help="Comma-separated class names to train and evaluate."),
) -> None:
    if resolution <= 0:
        raise typer.BadParameter("Resolution must be positive.")

    output_dir = output_dir.expanduser().resolve()
    class_names = data.parse_classes(classes)
    train_box_scales = parse_box_scales(train_box_scale)

    dataset_dir = data.write_coco_dataset(
        train_root.expanduser().resolve(),
        val_root.expanduser().resolve(),
        output_dir / "dataset-coco",
        class_names=class_names,
        val_max_samples=val_max_samples,
        sample_seed=seed,
        train_box_scales=train_box_scales,
    )

    notes = {
        "classes": class_names,
        "val_max_samples": val_max_samples,
        "train_box_scales": train_box_scales,
        "resolution": resolution,
        "use_ema": False,
    }
    (output_dir / "experiment.json").write_text(json.dumps(notes, indent=2) + "\n")

    model = RFDETRLarge(fused_optimizer=True)
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir / "rfdetr-large"),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=1,
        lr=learning_rate,
        lr_encoder=lr_encoder,
        lr_drop=lr_drop,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        num_workers=num_workers,
        eval_interval=1,
        use_ema=False,
        resolution=resolution,
        square_resize_div_64=True,
        multi_scale=False,
        expanded_scales=False,
        aug_config=COURT_AUGMENTATION,
        augmentation_backend="auto",
        checkpoint_interval=3,
        skip_best_epochs=1,
        seed=seed,
        notes=notes,
    )


def parse_box_scales(value: str) -> dict[str, float]:
    scales = {}
    for entry in value.split(","):
        class_name, scale = entry.split("=")
        canonical_name = data.canonical_category(class_name.strip())
        scales[canonical_name] = float(scale)
    return scales


if __name__ == "__main__":
    app()
