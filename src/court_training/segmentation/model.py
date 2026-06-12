import json
from pathlib import Path

import timm
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from court_training.segmentation import inference


class CourtSegmenter(nn.Module):
    def __init__(
        self,
        num_masks: int,
        num_keypoints: int | None = None,
        mask_names: tuple[str, ...] = (),
        keypoint_names: tuple[str, ...] = (),
        backbone: str = "vit_large_patch16_dinov3",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.mask_names = mask_names
        self.keypoint_names = keypoint_names
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.decoder = ProgressiveMaskHead(self.backbone.embed_dim, num_masks)
        self.keypoint_heatmaps = None
        self.keypoint_objectness = None
        if num_keypoints is not None:
            self.keypoint_heatmaps = nn.Sequential(
                nn.Conv2d(self.backbone.embed_dim, 256, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(256, num_keypoints, kernel_size=1),
            )
            self.keypoint_objectness = nn.Sequential(
                nn.Conv2d(self.backbone.embed_dim, 256, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(256, num_keypoints, kernel_size=1),
            )

    def forward(self, images: Float[Tensor, "B 3 H W"]):
        features = self.encode(images)
        masks = self.decode_masks(features, images.shape[-2:])
        keypoint_heatmaps = self.keypoint_heatmaps
        keypoint_objectness = self.keypoint_objectness
        if keypoint_heatmaps is None or keypoint_objectness is None:
            return {"masks": masks}

        keypoints, visibility, heatmaps = self.decode_keypoints(features, keypoint_heatmaps, keypoint_objectness)
        return {"masks": masks, "keypoints": keypoints, "visibility": visibility, "heatmaps": heatmaps}

    def decode_keypoints(
        self,
        features: Float[Tensor, "B C Hf Wf"],
        heatmap_head: nn.Module,
        objectness_head: nn.Module,
    ) -> tuple[Float[Tensor, "B K 2"], Float[Tensor, "B K"], Float[Tensor, "B K Hf Wf"]]:
        heatmaps = heatmap_head(features)
        visibility_logits = objectness_head(features).flatten(start_dim=2).amax(dim=2)
        return softargmax_2d(heatmaps), visibility_logits, heatmaps

    def decode_masks(
        self,
        features: Float[Tensor, "B C Hf Wf"],
        output_size: tuple[int, int],
    ) -> Float[Tensor, "B N H W"]:
        logits = self.decoder(features)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

    def encode(self, images: Float[Tensor, "B 3 H W"]) -> Float[Tensor, "B C Hf Wf"]:
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
        return features

    @classmethod
    def load(cls, checkpoint: Path, device: torch.device) -> "CourtSegmenter":
        """Build the model from the checkpoint's config.json sidecar and load its weights."""
        config = json.loads(checkpoint.with_name("config.json").read_text())
        model = cls(
            num_masks=len(config["mask_names"]),
            num_keypoints=len(config["keypoint_names"]),
            mask_names=tuple(config["mask_names"]),
            keypoint_names=tuple(config["keypoint_names"]),
            backbone=config["backbone"],
            pretrained=False,
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(device)
        model.eval()
        return model

    @torch.no_grad()
    def predict(
        self,
        images: Float[Tensor, "B 3 H W"],
        scales: tuple[float, ...] = (1.0,),
        fit_homography: bool = False,
        court_type: inference.CourtType = "nba",
    ) -> inference.Prediction:
        return inference.predict(self, images, self.mask_names, self.keypoint_names, scales, fit_homography, court_type)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def softargmax_2d(heatmaps: Float[Tensor, "B K H W"], temperature: float = 4.0) -> Float[Tensor, "B K 2"]:
    probabilities = (heatmaps * temperature).flatten(start_dim=2).softmax(dim=2)
    height, width = heatmaps.shape[-2:]
    x = torch.linspace(0, 1, width, device=heatmaps.device, dtype=heatmaps.dtype)
    y = torch.linspace(0, 1, height, device=heatmaps.device, dtype=heatmaps.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(height * width, 2)
    return torch.einsum("bkp,pd->bkd", probabilities, grid)


class ProgressiveMaskHead(nn.Module):
    def __init__(self, channels: int, num_masks: int) -> None:
        super().__init__()
        self.projection = ConvBlock(channels, 256)
        self.refine = nn.ModuleList([ConvBlock(256, 256) for _ in range(4)])
        self.output = nn.Conv2d(256, num_masks, kernel_size=1)

    def forward(self, features: Float[Tensor, "B C Hf Wf"]) -> Float[Tensor, "B N H W"]:
        features = self.projection(features)
        for block in self.refine:
            features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
            features = block(features)
        return self.output(features)


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(self, features: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        return self.net(features)
