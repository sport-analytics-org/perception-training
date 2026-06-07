# Court Training

Training and inference code for sport court segmentation models.

The current basketball model predicts six court masks from broadcast frames. Mask names come from
`sportanalytics.NbaCourt.areas()` and are ordered as court, three-point area, painted area, with left before right.

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

## Train

```bash
uv run train-basket-court /path/to/exported-dataset /path/to/checkpoints
```

Current recommended training recipe:

- backbone: `vit_large_patch16_dinov3`
- size: `480x360`
- optimizer: AdamW, `lr=3e-5`, `weight_decay=1e-4`
- augmentation: Albumentations appearance jitter, JPEG compression, blur/noise, and left/right-aware horizontal flip
- eval: fixed 12-image `e_bard_detection` split, with multiscale + flip TTA

Observed validation mIoU improved from `0.9536` for the previous comparable baseline to `0.9707` with appearance augmentation (`+0.0171`). Appearance augmentation was the most reliable setting. Blur/noise was close (`0.9696` best), no extra augmentation overfit late, and full affine was too strong. Soft affine looked promising but did not beat the appearance recipe in the completed sweep.

## Inference

```python
from court_training.inference import CourtSegmenter

segmenter = CourtSegmenter.from_checkpoint("checkpoints/best.pt")
probabilities = segmenter.predict_proba("frame.jpg")
masks = segmenter.predict_masks("frame.jpg")
bitfield = segmenter.predict_bitfield("frame.jpg")
```
