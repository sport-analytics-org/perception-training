import json
from pathlib import Path
from types import SimpleNamespace

import torch
from jaxtyping import Float
from rfdetr.config import RFDETRLargeConfig, TrainConfig
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
from rfdetr.models.weights import load_pretrain_weights
from torch import Tensor, nn
from torchvision.ops import box_convert

from court_training import dataset

LR_VIT_LAYER_DECAY = 0.8
LR_COMPONENT_DECAY = 0.7


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
        # build_criterion_from_config requires a TrainConfig but only reads loss weights, never the paths
        train_config = TrainConfig(dataset_dir=".", output_dir=".")
        self.criterion, self.postprocess = build_criterion_from_config(config, train_config)

    def forward(self, images: Float[Tensor, "B 3 H W"]) -> dict:
        return self.model(images)

    def loss(self, outputs: dict, targets: list[dataset.Target]) -> Tensor:
        criterion_targets = []
        for target in targets:
            boxes_cxcywh = box_convert(target["boxes_xywh"], "xywh", "cxcywh")
            criterion_targets.append({"boxes": boxes_cxcywh, "labels": target["labels"]})
        losses = self.criterion(outputs, criterion_targets)
        weights = self.criterion.weight_dict
        return sum(losses[name] * weights[name] for name in losses if name in weights)

    @classmethod
    def load(cls, checkpoint: Path, device: torch.device) -> "CourtDetector":
        """Build the model from the checkpoint's metadata.json sidecar and load its weights."""
        metadata = json.loads(checkpoint.with_name("metadata.json").read_text())
        model = cls(tuple(metadata["classes"]), metadata["resolution"], pretrained=False)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

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
        decoder_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad or name in backbone_groups:
                continue
            if "transformer.decoder" in name:
                decoder_params.append(param)
            else:
                other_params.append(param)
        return [
            {"params": other_params, "lr": lr},
            {"params": decoder_params, "lr": lr * LR_COMPONENT_DECAY},
            *backbone_groups.values(),
        ]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
