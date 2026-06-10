import timm
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


class DinoDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_queries: int = 64,
        backbone: str = "vit_large_patch16_dinov3",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.no_object_class = num_classes
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        channels = self.backbone.embed_dim
        self.input_projection = nn.Conv2d(channels, 256, kernel_size=1)
        self.queries = nn.Embedding(num_queries, 256)
        decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.class_head = nn.Linear(256, num_classes + 1)
        self.box_head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 4))

    def forward(self, images: Float[Tensor, "B 3 H W"]) -> dict[str, Tensor]:
        features = self.encode(images)
        memory = self.input_projection(features).flatten(start_dim=2).transpose(1, 2)
        queries = self.queries.weight.unsqueeze(0).expand(images.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        return {
            "logits": self.class_head(decoded),
            "boxes_xywh": self.box_head(decoded).sigmoid(),
        }

    def encode(self, images: Float[Tensor, "B 3 H W"]) -> Float[Tensor, "B C Hf Wf"]:
        height, width = images.shape[-2:]
        patch_height, patch_width = self.backbone.patch_embed.patch_size
        pad_height = (-height) % patch_height
        pad_width = (-width) % patch_width
        padded_images = F.pad(images, (0, pad_width, 0, pad_height))
        tokens = self.backbone.forward_features(padded_images)
        patch_tokens = tokens[:, self.backbone.num_prefix_tokens :]
        return patch_tokens.transpose(1, 2).reshape(
            padded_images.shape[0],
            self.backbone.embed_dim,
            padded_images.shape[-2] // patch_height,
            padded_images.shape[-1] // patch_width,
        )
