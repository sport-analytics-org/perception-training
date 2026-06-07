import timm
import torch
from torch import nn
from torch.nn import functional as F

from court_training.constants import BACKBONE
from court_training.masks import LEFT_RIGHT_PAIRS, MASK_NAMES


class DinoSegmenter(nn.Module):
    def __init__(self, backbone: str = BACKBONE, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone.embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, len(MASK_NAMES), kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        patch_height, patch_width = self.backbone.patch_embed.patch_size
        pad_height = (-height) % patch_height
        pad_width = (-width) % patch_width
        padded_images = F.pad(images, (0, pad_width, 0, pad_height))

        tokens = self.backbone.forward_features(padded_images)
        patch_tokens = tokens[:, self.backbone.num_prefix_tokens :]
        features = patch_tokens.transpose(1, 2).reshape(
            padded_images.shape[0],
            self.backbone.embed_dim,
            padded_images.shape[-2] // patch_height,
            padded_images.shape[-1] // patch_width,
        )

        logits = self.decoder(features)
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)


def predict_multiscale(model: nn.Module, images: torch.Tensor, scales: tuple[float, ...]) -> torch.Tensor:
    output_size = images.shape[-2:]
    logits_by_scale = []
    for scale in scales:
        scaled_images = resize_images(images, scale)
        logits = model(scaled_images)
        logits = (logits + predict_flipped(model, scaled_images)) / 2
        logits_by_scale.append(F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False))
    return torch.stack(logits_by_scale).mean(dim=0)


def resize_images(images: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return images
    return F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)


def predict_flipped(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    logits = model(torch.flip(images, dims=(-1,)))
    logits = torch.flip(logits, dims=(-1,))
    return swap_left_right_channels(logits)


def swap_left_right_channels(tensor: torch.Tensor) -> torch.Tensor:
    swapped = tensor.clone()
    for left, right in LEFT_RIGHT_PAIRS:
        swapped[:, left] = tensor[:, right]
        swapped[:, right] = tensor[:, left]
    return swapped
