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

## Homography fitting

Fit a centered-initialization homography to a labeled basketball raster mask:

```bash
uv run python scripts/dataset/fit_homography.py /path/to/mask.webp --court fiba
```

The script reads raster bitfield masks directly and prints the fitted homography plus initial/final IoU.
