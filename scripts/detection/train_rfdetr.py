from pathlib import Path
from typing import Annotated, Literal

import typer
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

from court_training.detection import data

app = typer.Typer(help="Fine-tune an RF-DETR detector on exported basketball detections.")
Variant = Annotated[Literal["nano", "small", "medium", "base", "large"], typer.Option(help="RF-DETR model variant.")]
VARIANTS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
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
    fused_optimizer: bool = typer.Option(
        True,
        "--fused-optimizer/--no-fused-optimizer",
        help="Use RF-DETR's fused AdamW optimizer when supported.",
    ),
    classes: str | None = typer.Option(None, help="Comma-separated class names to train and evaluate."),
) -> None:
    output_dir = output_dir.expanduser().resolve()
    dataset_dir = data.write_coco_dataset(
        train_root.expanduser().resolve(),
        val_root.expanduser().resolve(),
        output_dir / "dataset-coco",
        class_names=data.parse_classes(classes),
    )
    model_class = VARIANTS[variant]
    model = model_class(fused_optimizer=fused_optimizer)
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir / f"rfdetr-{variant}"),
        epochs=epochs,
        batch_size=batch_size,
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
    )


if __name__ == "__main__":
    app()
