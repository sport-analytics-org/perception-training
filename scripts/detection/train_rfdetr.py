from pathlib import Path

import typer
from rfdetr import RFDETRBase

from court_training.detection import data

app = typer.Typer(help="Fine-tune an RF-DETR detector on exported basketball detections.")


@app.command()
def main(
    train_root: Path,
    val_root: Path,
    output_dir: Path,
    epochs: int = typer.Option(50, help="Training epochs."),
    batch_size: int = typer.Option(4, help="Training batch size."),
    grad_accum_steps: int = typer.Option(4, help="RF-DETR gradient accumulation steps."),
    learning_rate: float = typer.Option(1e-4, help="RF-DETR learning rate."),
    resolution: int = typer.Option(560, help="Input resolution. RF-DETR expects this to be divisible by 56."),
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
    model = RFDETRBase(fused_optimizer=fused_optimizer)
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir / "rfdetr"),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=learning_rate,
        resolution=resolution,
    )


if __name__ == "__main__":
    app()
