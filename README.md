# Court Training

Training and inference code for sport court perception models.

## Dataset Export

Input datasets are expected to be flat train/val folders:

```text
dataset-root/
  train/
    images/*.jpg
    masks/*.webp
    keypoints/*.json
    detections/*.npz
  val/
    images/*.jpg
    masks/*.webp
    keypoints/*.json
    detections/*.npz
```

The `masks`, `keypoints`, and `detections` directories are only created when those annotations are exported.
Mask files are grayscale WebP bitfields. Bit `0..5` maps to the mask order above.
Detection files are compressed NumPy archives with normalized top-left `boxes_xywh`
and per-box `category_names` arrays. Samples without detection annotations get empty arrays.

To export original labelled subdatasets into that layout:

```bash
uv run python scripts/export_dataset.py /path/to/basketball-imgs /path/to/output \
  --train-dataset basketball_51 \
  --train-dataset borgo \
  --val-dataset e_bard_detection
```

Use `--masks/--no-masks` and `--detections/--no-detections` to choose which annotations to
export. At least one train or val subdataset must be selected, but both splits are optional:

```bash
uv run python scripts/export_dataset.py /path/to/basketball-imgs /path/to/output \
  --train-dataset basketball_player_detection_2 \
  --no-masks \
  --detections
```

## Segmentation

The current basketball segmentation model predicts six court masks from broadcast frames. Mask names come from
`sportanalytics.NbaCourt.areas()` and are ordered as court, three-point area, painted area, with left before right.

Train the basketball segmentation model:

```bash
uv run python scripts/segmentation/train_basket.py /path/to/output/train /path/to/output/val /path/to/checkpoints
```

Predict segmentation masks and fit homographies:

```bash
uv run python scripts/segmentation/predict_and_fit_homography.py /path/to/basketball-imgs /path/to/checkpoint.pt /path/to/report
```

## Detection

Export the basketball detection train/eval split:

```bash
uv run python scripts/export_dataset.py /path/to/basketball-imgs /path/to/detection-dataset \
  --train-dataset basketball_player_detection_2 \
  --val-dataset e_bard_detection \
  --no-masks \
  --detections
```

Fine-tune RF-DETR Large on the exported split. Training consumes the flat layout directly,
saves the checkpoint with the best validation mAP to `best.pt`, and defaults to the
square `640x640` setup:

```bash
uv run python scripts/detection/train_rfdetr.py /path/to/detection-dataset/train /path/to/runs/rfdetr \
  --val-root /path/to/detection-dataset/val
```

Evaluate a checkpoint, optionally with horizontal-flip TTA:

```bash
uv run python scripts/detection/evaluate_rfdetr.py /path/to/runs/rfdetr/best.pt /path/to/detection-dataset/val /path/to/runs/rfdetr/metrics.json --hflip
```

## API

The API is a small FastAPI app for low-volume frontend inference. Both models are loaded once at startup from
environment variables and then reused for every request.

```bash
export COURT_SEGMENTATION_CHECKPOINT=/path/to/segmentation/best.pt
export COURT_DETECTION_CHECKPOINT=/path/to/rfdetr-large/checkpoint_best_total.pth
uv run uvicorn court_training.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Predict segmentation and detections for one image:

```bash
curl -X POST http://localhost:8000/predict \
  -F image=@frame.jpg \
  -F segmentation=true \
  -F detection=true
```

The response matches the labeltool annotation schemas: mask polygons and keypoints (`{position, visible}`) in
normalized coordinates, the fitted homography (`null` when fewer than 4 keypoints are visible), and detection boxes
as pixel `bbox_xyxy` with `category_id` plus a `categories` list. Set `segmentation=false` or `detection=false` to
call only one model.

## Homography fitting

Fit a centered-initialization homography to a labeled basketball raster mask:

```bash
uv run python scripts/dataset/fit_homography.py /path/to/mask.webp --court fiba
```

The script reads raster bitfield masks directly and prints the fitted homography plus initial/final IoU.
