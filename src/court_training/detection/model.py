import json
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int
from PIL import Image
from rfdetr.config import RFDETRLargeConfig, TrainConfig
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
from rfdetr.models.weights import load_pretrain_weights
from torch import Tensor, nn
from torchvision.ops import box_convert

from court_training import dataset, flip
from court_training.image_io import image2tensor

LR_VIT_LAYER_DECAY = 0.8
LR_COMPONENT_DECAY = 0.7


class Prediction(TypedDict):
    boxes: Float[np.ndarray, "N 4"]
    scores: Float[np.ndarray, "N"]
    labels: Int[np.ndarray, "N"]


class CourtDetector(nn.Module):
    """RF-DETR Large detector using only the model, loss, and postprocessor from rfdetr."""

    def __init__(self, class_names: tuple[str, ...], image_size: tuple[int, int], pretrained: bool = True) -> None:
        super().__init__()
        if image_size[0] != image_size[1]:
            raise ValueError(f"RF-DETR requires square image_size, got {image_size}")
        resolution = image_size[0]
        config = RFDETRLargeConfig(num_classes=len(class_names))
        config.resolution = resolution
        config.positional_encoding_size = resolution // config.patch_size
        self.class_names = class_names
        self.image_size = image_size
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
        """Build the model from the checkpoint's args.json sidecar and load its weights."""
        metadata_path = checkpoint.with_name("args.json")
        metadata = json.loads(metadata_path.read_text())
        image_size_config = metadata["image_size"]
        image_size = (image_size_config["height"], image_size_config["width"])
        model = cls(tuple(metadata["classes"]), image_size, pretrained=False)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    @torch.inference_mode()
    def predict(
        self,
        images: list[Image.Image],
        scales: tuple[float, ...] = (1.0,),
        hflip: bool = False,
    ) -> list[Prediction]:
        """Per-image detections with normalized xyxy boxes as numpy arrays."""
        predictions_by_image = [[] for _ in images]
        variant_tensors = []
        variant_images = []
        variant_flipped = []
        for image_index, image in enumerate(images):
            for scale in scales:
                height, width = scaled_image_size(self.image_size, scale)
                resized = image.resize((width, height), Image.Resampling.BILINEAR)
                variant_tensors.append(image2tensor(resized))
                variant_images.append(image_index)
                variant_flipped.append(False)
                if hflip:
                    flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    resized = flipped.resize((width, height), Image.Resampling.BILINEAR)
                    variant_tensors.append(image2tensor(resized))
                    variant_images.append(image_index)
                    variant_flipped.append(True)

        for tensor, image_index, flipped in zip(variant_tensors, variant_images, variant_flipped, strict=True):
            prediction = _predict_tensors(self, tensor[None].to(self.device))[0]
            if flipped:
                prediction["boxes"] = flip.flip_torch(boxes_xywh=prediction["boxes"])["boxes_xywh"]
            predictions_by_image[image_index].append(prediction)

        outputs = []
        for predictions in predictions_by_image:
            prediction = merge_predictions(predictions)
            boxes = box_convert(prediction["boxes"], "xywh", "xyxy").cpu().numpy().astype(np.float32)
            scores = prediction["scores"].cpu().numpy().astype(np.float32)
            labels = prediction["labels"].cpu().numpy()
            outputs.append({"boxes": boxes, "scores": scores, "labels": labels})
        return outputs

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
    def resolution(self) -> int:
        return self.image_size[0]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def scaled_image_size(image_size: tuple[int, int], scale: float) -> tuple[int, int]:
    height, width = image_size
    return max(1, round(height * scale)), max(1, round(width * scale))


def _predict_tensors(model: CourtDetector, images: Float[Tensor, "B 3 H W"]) -> list[dict[str, Tensor]]:
    """Run RF-DETR on normalized tensors and return normalized xywh detections."""
    outputs = model.model(images)
    unit_sizes = torch.ones((len(images), 2), device=images.device)
    results = model.postprocess(outputs, unit_sizes)
    predictions = []
    for result in results:
        keep = result["labels"] < len(model.class_names)
        prediction = {key: value[keep] for key, value in result.items()}
        prediction["boxes"] = box_convert(prediction["boxes"], "xyxy", "xywh")
        predictions.append(prediction)
    return predictions


def merge_predictions(predictions: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    if not predictions:
        empty_float = torch.empty((0,), dtype=torch.float32)
        empty_long = torch.empty((0,), dtype=torch.long)
        return {"boxes": empty_float.reshape(0, 4), "scores": empty_float, "labels": empty_long}
    return {
        "boxes": torch.cat([prediction["boxes"] for prediction in predictions]),
        "scores": torch.cat([prediction["scores"] for prediction in predictions]),
        "labels": torch.cat([prediction["labels"] for prediction in predictions]),
    }

