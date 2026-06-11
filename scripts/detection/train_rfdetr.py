import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.datasets.aug_config import AUG_AGGRESSIVE, AUG_CONSERVATIVE

from court_training.detection import data

app = typer.Typer(help="Fine-tune an RF-DETR detector on exported basketball detections.")
Variant = Annotated[Literal["nano", "small", "medium", "base", "large"], typer.Option(help="RF-DETR model variant.")]
AugPreset = Annotated[
    Literal["none", "flip", "court", "strong", "conservative", "aggressive"],
    typer.Option(help="Training augmentation preset."),
]
VARIANTS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}
AUG_PRESETS = {
    "none": {},
    "flip": {"HorizontalFlip": {"p": 0.5}},
    "court": {
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
    },
    "strong": {
        "HorizontalFlip": {"p": 0.5},
        "Affine": {
            "scale": (0.75, 1.3),
            "translate_percent": (-0.12, 0.12),
            "rotate": (-10, 10),
            "shear": (-5, 5),
            "p": 0.6,
        },
        "ColorJitter": {
            "brightness": 0.25,
            "contrast": 0.25,
            "saturation": 0.2,
            "hue": 0.04,
            "p": 0.6,
        },
        "GaussianBlur": {"blur_limit": 3, "p": 0.18},
        "GaussNoise": {"std_range": (0.01, 0.04), "p": 0.18},
    },
    "conservative": AUG_CONSERVATIVE,
    "aggressive": AUG_AGGRESSIVE,
}


@app.command()
def main(
    train_root: Path,
    val_root: Path,
    output_dir: Path,
    variant: Variant = "large",
    epochs: int = typer.Option(100, help="Training epochs."),
    batch_size: int = typer.Option(4, help="Training batch size."),
    grad_accum_steps: int = typer.Option(4, help="RF-DETR gradient accumulation steps."),
    learning_rate: float = typer.Option(1e-4, help="RF-DETR learning rate."),
    lr_encoder: float = typer.Option(1.5e-4, help="RF-DETR encoder learning rate."),
    lr_drop: int = typer.Option(100, help="Epoch for step LR decay."),
    warmup_epochs: float = typer.Option(0.0, help="Warmup epochs."),
    weight_decay: float = typer.Option(1e-4, help="Weight decay."),
    num_workers: int = typer.Option(8, help="Dataloader workers."),
    eval_interval: int = typer.Option(1, help="Evaluate every N epochs."),
    resolution: int = typer.Option(704, help="Input resolution. Large expects multiples of 32."),
    multi_scale: bool = typer.Option(True, "--multi-scale/--no-multi-scale", help="Use RF-DETR multi-scale training."),
    expanded_scales: bool = typer.Option(
        True,
        "--expanded-scales/--no-expanded-scales",
        help="Use RF-DETR expanded scale augmentation.",
    ),
    random_resize_padding: bool = typer.Option(
        False,
        "--random-resize-padding/--no-random-resize-padding",
        help="Use RF-DETR random resize via padding.",
    ),
    augmentation: AugPreset = "court",
    augmentation_backend: Literal["cpu", "auto", "gpu"] = typer.Option("auto", help="RF-DETR augmentation backend."),
    fused_optimizer: bool = typer.Option(
        True,
        "--fused-optimizer/--no-fused-optimizer",
        help="Use RF-DETR's fused AdamW optimizer when supported.",
    ),
    auto_batch: bool = typer.Option(False, "--auto-batch", help="Let RF-DETR find the largest safe micro-batch."),
    checkpoint_interval: int = typer.Option(10, help="Save a checkpoint every N epochs."),
    skip_best_epochs: int = typer.Option(0, help="Do not save best checkpoints before this epoch."),
    seed: int | None = typer.Option(42, help="Training seed."),
    compile_model: bool = typer.Option(
        False,
        "--compile/--no-compile",
        help="Use torch compile for the RF-DETR model.",
    ),
    gradient_checkpointing: bool = typer.Option(
        False,
        "--gradient-checkpointing/--no-gradient-checkpointing",
        help="Trade compute for memory in the RF-DETR backbone.",
    ),
    train_max_samples: int = typer.Option(0, help="Use at most this many training images. Zero keeps all images."),
    val_max_samples: int = typer.Option(0, help="Use at most this many validation images during training."),
    train_box_scale: str | None = typer.Option(
        None,
        help="Comma-separated class=scale entries applied only to training boxes, e.g. ball=1.35.",
    ),
    classes: str | None = typer.Option(None, help="Comma-separated class names to train and evaluate."),
) -> None:
    output_dir = output_dir.expanduser().resolve()
    class_names = data.parse_classes(classes)
    train_box_scales = parse_box_scales(train_box_scale)
    dataset_dir = data.write_coco_dataset(
        train_root.expanduser().resolve(),
        val_root.expanduser().resolve(),
        output_dir / "dataset-coco",
        class_names=class_names,
        train_max_samples=train_max_samples,
        val_max_samples=val_max_samples,
        sample_seed=seed or 42,
        train_box_scales=train_box_scales,
    )
    model_class = VARIANTS[variant]
    model = model_class(
        compile=compile_model,
        fused_optimizer=fused_optimizer,
        gradient_checkpointing=gradient_checkpointing,
    )
    batch_size_value: int | Literal["auto"] = "auto" if auto_batch else batch_size
    notes = {
        "classes": class_names,
        "augmentation": augmentation,
        "train_max_samples": train_max_samples,
        "val_max_samples": val_max_samples,
        "train_box_scales": train_box_scales,
    }
    (output_dir / "experiment.json").write_text(json.dumps(notes, indent=2) + "\n")
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir / f"rfdetr-{variant}"),
        epochs=epochs,
        batch_size=batch_size_value,
        grad_accum_steps=grad_accum_steps,
        lr=learning_rate,
        lr_encoder=lr_encoder,
        lr_drop=lr_drop,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        num_workers=num_workers,
        eval_interval=eval_interval,
        resolution=resolution,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        do_random_resize_via_padding=random_resize_padding,
        aug_config=AUG_PRESETS[augmentation],
        augmentation_backend=augmentation_backend,
        checkpoint_interval=checkpoint_interval,
        skip_best_epochs=skip_best_epochs,
        seed=seed,
        notes=notes,
    )


def parse_box_scales(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    scales = {}
    for entry in value.split(","):
        class_name, _, scale = entry.partition("=")
        if not scale:
            raise typer.BadParameter("box scales must use class=scale entries")
        class_name = data.canonical_category(class_name.strip())
        if class_name not in data.BASKETBALL_DETECTION_CLASSES:
            raise typer.BadParameter(f"unknown detection class: {class_name}")
        scales[class_name] = float(scale)
    return scales


if __name__ == "__main__":
    app()
