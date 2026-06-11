from pathlib import Path

import typer

from court_training.detection.data import parse_classes, write_yolo_dataset

app = typer.Typer(help="Fine-tune an Ultralytics YOLO detector on exported basketball detections.")


@app.command()
def main(
    train_root: Path,
    val_root: Path,
    output_dir: Path,
    model: str = typer.Option("yolo11n.pt", help="Ultralytics model checkpoint or config."),
    epochs: int = typer.Option(50, help="Training epochs."),
    image_size: int = typer.Option(640, help="Square training image size."),
    batch_size: int = typer.Option(16, help="Training batch size."),
    classes: str | None = typer.Option(None, help="Comma-separated class names to train and evaluate."),
    learning_rate: float | None = typer.Option(None, help="Initial YOLO learning rate."),
    final_lr_fraction: float = typer.Option(0.01, help="Final LR as a fraction of the initial LR."),
    patience: int = typer.Option(100, help="Early-stopping patience in epochs."),
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install YOLO support with `uv pip install ultralytics`.") from error

    output_dir = output_dir.expanduser().resolve()
    dataset_yaml = write_yolo_dataset(
        train_root.expanduser().resolve(),
        val_root.expanduser().resolve(),
        output_dir / "dataset-yolo",
        class_names=parse_classes(classes),
    )
    yolo = YOLO(model)
    train_kwargs = {}
    if learning_rate is not None:
        train_kwargs["lr0"] = learning_rate
        train_kwargs["lrf"] = final_lr_fraction
    yolo.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project=str(output_dir),
        name="yolo",
        patience=patience,
        **train_kwargs,
    )


if __name__ == "__main__":
    app()
