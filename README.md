# Court Training

Training and inference code for sport court segmentation models.

The current basketball model predicts six court masks from broadcast frames. Mask names come from
`sportanalytics.NbaCourt.areas()` and are ordered as court, three-point area, painted area, with left before right.

Input datasets are expected to be flat train/val folders:

```text
dataset-root/
  train/
    images/*.jpg
    masks/*.webp
  val/
    images/*.jpg
    masks/*.webp
```

Mask files are grayscale WebP bitfields. Bit `0..5` maps to the mask order above.

To export original labelled subdatasets into that layout:

```bash
uv run python scripts/export_dataset.py /path/to/output \
  --train-source /path/to/basketball_51 \
  --train-source /path/to/borgo \
  --val-source /path/to/e_bard_detection
```

## Homography fitting

Fit a centered-initialization homography to a labeled basketball raster mask:

```bash
uv run python scripts/fit_homography.py /path/to/exported-dataset borgo 003/u19_a_1a_college_crocetta_02438.jpg
```

The script reads raster bitfield masks directly and prints the fitted homography plus initial/final IoU.
