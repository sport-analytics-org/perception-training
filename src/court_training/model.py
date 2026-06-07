import timm
import torch
from jaxtyping import Float
from PIL import Image
from torch import nn
from torch.nn import functional as F

from court_training.constants import LEFT_RIGHT_PAIRS, MASK_NAMES
from court_training.dataset import image_to_tensor


class DinoSegmenter(nn.Module):
    def __init__(self, backbone: str = "vit_large_patch16_dinov3", pretrained: bool = True) -> None:
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

    def forward(
        self,
        images: Float[torch.Tensor, "batch 3 height width"],
    ) -> Float[torch.Tensor, "batch n_masks height width"]:
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

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        scales: tuple[float, ...],
    ) -> Float[torch.Tensor, "n_masks height width"]:
        was_training = self.training
        self.eval()
        try:
            images = image_to_tensor(image.convert("RGB")).unsqueeze(0).to(self.device)
            output_size = images.shape[-2:]
            logits_by_scale = []
            for scale in scales:
                scaled_images = resize_images(images, scale)
                logits = self(scaled_images)
                logits = (logits + self.predict_flipped(scaled_images)) / 2
                logits_by_scale.append(F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False))
            return torch.stack(logits_by_scale).mean(dim=0).squeeze(0)
        finally:
            self.train(was_training)

    def predict_flipped(
        self,
        images: Float[torch.Tensor, "batch 3 height width"],
    ) -> Float[torch.Tensor, "batch n_masks height width"]:
        logits = self(torch.flip(images, dims=(-1,)))
        logits = torch.flip(logits, dims=(-1,))
        return swap_left_right_channels(logits)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def resize_images(images: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return images
    return F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)


def swap_left_right_channels(tensor: torch.Tensor) -> torch.Tensor:
    swapped = tensor.clone()
    for left, right in LEFT_RIGHT_PAIRS:
        swapped[:, left] = tensor[:, right]
        swapped[:, right] = tensor[:, left]
    return swapped
