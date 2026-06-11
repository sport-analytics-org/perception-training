from types import SimpleNamespace
from typing import TypedDict

import torch
from jaxtyping import Float, Int64
from rfdetr.config import RFDETRLargeConfig, TrainConfig
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
from rfdetr.models.weights import load_pretrain_weights
from torch import Tensor, nn
from torchvision.ops import box_convert

from court_training import dataset

LR_VIT_LAYER_DECAY = 0.8
LR_COMPONENT_DECAY = 0.7


class Target(TypedDict):
    """Criterion contract: normalized cxcywh boxes, the one place that format is required."""

    boxes: Float[Tensor, "D 4"]
    labels: Int64[Tensor, "D"]


class CourtDetector(nn.Module):
    """RF-DETR Large detector using only the model, loss, and postprocessor from rfdetr."""

    def __init__(self, class_names: tuple[str, ...], resolution: int, pretrained: bool = True) -> None:
        super().__init__()
        config = RFDETRLargeConfig(num_classes=len(class_names))
        config.resolution = resolution
        config.positional_encoding_size = resolution // config.patch_size
        self.class_names = class_names
        self.resolution = resolution
        self.config = config
        self.model = build_model_from_config(config)
        if pretrained:
            load_pretrain_weights(self.model, config)
        train_config = TrainConfig(dataset_dir=".", output_dir=".")
        self.criterion, self.postprocess = build_criterion_from_config(config, train_config)

    def forward(self, images: Float[Tensor, "B 3 H W"]) -> dict:
        return self.model(images)

    def loss(self, outputs: dict, targets: list[Target]) -> Tensor:
        losses = self.criterion(outputs, targets)
        weights = self.criterion.weight_dict
        return sum(losses[name] * weights[name] for name in losses if name in weights)

    @torch.inference_mode()
    def predict(self, images: Float[Tensor, "B 3 H W"]) -> list[dict[str, Tensor]]:
        """Per-image scores, labels, and normalized xywh boxes."""
        outputs = self.model(images)
        unit_sizes = torch.ones((len(images), 2), device=images.device)
        results = self.postprocess(outputs, unit_sizes)
        predictions = []
        for result in results:
            keep = result["labels"] < len(self.class_names)
            prediction = {key: value[keep] for key, value in result.items()}
            prediction["boxes"] = box_convert(prediction["boxes"], "xyxy", "xywh")
            predictions.append(prediction)
        return predictions

    def param_groups(self, lr: float, lr_encoder: float, weight_decay: float) -> list[dict]:
        args = SimpleNamespace(
            lr=lr,
            lr_encoder=lr_encoder,
            weight_decay=weight_decay,
            lr_vit_layer_decay=LR_VIT_LAYER_DECAY,
            lr_component_decay=LR_COMPONENT_DECAY,
            out_feature_indexes=self.config.out_feature_indexes,
        )
        backbone_groups = self.model.backbone[0].get_named_param_lr_pairs(args, prefix="backbone.0")
        decoder_params = [
            param
            for name, param in self.model.named_parameters()
            if "transformer.decoder" in name and param.requires_grad
        ]
        other_params = [
            param
            for name, param in self.model.named_parameters()
            if name not in backbone_groups and "transformer.decoder" not in name and param.requires_grad
        ]
        return [
            {"params": other_params, "lr": lr},
            {"params": decoder_params, "lr": lr * LR_COMPONENT_DECAY},
            *backbone_groups.values(),
        ]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def collate(batch: list[dataset.TorchSample]) -> tuple[Float[Tensor, "B 3 H W"], list[Target]]:
    images = torch.stack([sample["image"] for sample in batch])
    targets: list[Target] = []
    for sample in batch:
        boxes_cxcywh = box_convert(sample["boxes_xywh"], "xywh", "cxcywh")
        targets.append({"boxes": boxes_cxcywh, "labels": sample["labels"]})
    return images, targets
