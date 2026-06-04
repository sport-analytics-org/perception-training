# Court Training

Training and inference code for sport court segmentation models.

The current basketball model predicts six court masks from broadcast frames:

- left court
- right court
- left three-point area
- right three-point area
- left painted area
- right painted area

Input datasets are expected to be exported label datasets with this structure:

```text
dataset-root/
  basket/
    basketball_51/
      images/000/*.jpg
      masks/000/*.webp
    borgo/
      images/000/*.jpg
      masks/000/*.webp
    e_bard_detection/
      images/000/*.jpg
      masks/000/*.webp
```

Mask files are grayscale WebP bitfields. Bit `0..5` maps to the mask order above.
