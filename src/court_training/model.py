import timm
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from court_training.dataset import image_to_tensor


class CourtSegmenter(nn.Module):
    def __init__(
        self,
        num_masks: int,
        num_keypoints: int | None = None,
        left_right_pairs: tuple[tuple[int, int], ...] = (),
        backbone: str = "vit_large_patch16_dinov3",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.left_right_pairs = left_right_pairs
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone.embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, num_masks, kernel_size=1),
        )
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

    def forward(self, images: Float[Tensor, "B 3 H W"]) -> Float[Tensor, "B N H W"]:
        features = self.encode(images)
        return self.decode_masks(features, images.shape[-2:])

    def forward_with_keypoints(
        self,
        images: Float[Tensor, "B 3 H W"],
    ) -> tuple[
        Float[Tensor, "B N H W"],
        Float[Tensor, "B K 2"],
        Float[Tensor, "B K"],
        Float[Tensor, "B K Hf Wf"],
    ]:
        features = self.encode(images)
        mask_logits = self.decode_masks(features, images.shape[-2:])
        keypoints, visibility_logits, heatmaps = self.decode_keypoints(features)
        return mask_logits, keypoints, visibility_logits, heatmaps

    def predict_keypoints(
        self,
        images: Float[Tensor, "B 3 H W"],
    ) -> tuple[Float[Tensor, "B K 2"], Float[Tensor, "B K"]]:
        features = self.encode(images)
        keypoints, visibility_logits, _ = self.decode_keypoints(features)
        return keypoints, visibility_logits

    def decode_keypoints(
        self,
        features: Float[Tensor, "B C Hf Wf"],
    ) -> tuple[Float[Tensor, "B K 2"], Float[Tensor, "B K"], Float[Tensor, "B K Hf Wf"]]:
        heatmaps = self.keypoint_heatmaps(features)
        visibility_logits = self.keypoint_objectness(features).flatten(start_dim=2).amax(dim=2)
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

    @torch.no_grad()
    def predict(self, image: Image.Image, scales: tuple[float, ...]) -> Float[Tensor, "N H W"]:
        images = image_to_tensor(image.convert("RGB")).unsqueeze(0).to(self.device)
        output_size = images.shape[-2:]
        logits_by_scale = []
        for scale in scales:
            scaled_images = resize_images(images, scale)
            logits = self(scaled_images)
            logits = (logits + self.predict_flipped(scaled_images)) / 2
            logits_by_scale.append(F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False))
        return torch.stack(logits_by_scale).mean(dim=0).squeeze(0)

    def predict_flipped(self, images: Float[Tensor, "B 3 H W"]) -> Float[Tensor, "B N H W"]:
        logits = self(torch.flip(images, dims=(-1,)))
        logits = torch.flip(logits, dims=(-1,))
        return swap_left_right_channels(logits, self.left_right_pairs)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def resize_images(images: Tensor, scale: float) -> Tensor:
    return F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)


def swap_left_right_channels(tensor: Tensor, left_right_pairs: tuple[tuple[int, int], ...]) -> Tensor:
    swapped = tensor.clone()
    for left, right in left_right_pairs:
        swapped[:, left] = tensor[:, right]
        swapped[:, right] = tensor[:, left]
    return swapped


def softargmax_2d(heatmaps: Float[Tensor, "B K H W"], temperature: float = 4.0) -> Float[Tensor, "B K 2"]:
    probabilities = (heatmaps * temperature).flatten(start_dim=2).softmax(dim=2)
    height, width = heatmaps.shape[-2:]
    x = torch.linspace(0, 1, width, device=heatmaps.device, dtype=heatmaps.dtype)
    y = torch.linspace(0, 1, height, device=heatmaps.device, dtype=heatmaps.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(height * width, 2)
    return torch.einsum("bkp,pd->bkd", probabilities, grid)
