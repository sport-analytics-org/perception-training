import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F  # noqa: N812
from jaxtyping import Float
from PIL import Image
from rfdetr.config import RFDETRLargeConfig, TrainConfig
from torch import Tensor, nn
from torchvision.ops import box_convert

import perception_training as pt
import perception_training.detection as detection
from perception_training.image_io import image2tensor
from perception_training.vendor.rfdetr_models.lwdetr import build_criterion_from_config, build_model_from_config
from perception_training.vendor.rfdetr_models.weights import load_pretrain_weights

LR_VIT_LAYER_DECAY = 0.8
LR_COMPONENT_DECAY = 0.7
ATTRIBUTE_LOSS_WEIGHT = 0.5


class CourtDetector(nn.Module):
    """RF-DETR Large detector using only the model, loss, and postprocessor from rfdetr."""

    def __init__(
        self,
        class_names: tuple[str, ...],
        image_size: tuple[int, int],
        pretrained: bool = True,
        attribute_names: tuple[str, ...] = pt.dataset.BASKETBALL_DETECTION_ATTRIBUTES,
        attribute_loss_weight: float = ATTRIBUTE_LOSS_WEIGHT,
        attribute_pos_weight: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        if image_size[0] != image_size[1]:
            raise ValueError(f"RF-DETR requires square image_size, got {image_size}")
        resolution = image_size[0]
        config = RFDETRLargeConfig(num_classes=len(class_names))
        config.resolution = resolution
        config.positional_encoding_size = resolution // config.patch_size
        self.class_names = class_names
        self.attribute_names = attribute_names
        self.attribute_loss_weight = attribute_loss_weight
        self.image_size = image_size
        self.config = config
        if attribute_pos_weight is None:
            attribute_pos_weight = (1.0,) * len(attribute_names)
        if len(attribute_pos_weight) != len(attribute_names):
            raise ValueError(f"Expected {len(attribute_names)} attribute weights, got {len(attribute_pos_weight)}")
        self.register_buffer("attribute_pos_weight", torch.tensor(attribute_pos_weight, dtype=torch.float32))
        self.register_buffer(
            "attribute_class_index",
            torch.tensor(
                [class_names.index(pt.dataset.BASKETBALL_ATTRIBUTE_BASE_CLASSES[name]) for name in attribute_names],
                dtype=torch.long,
            ),
        )
        self.model = build_model_from_config(config)
        if pretrained:
            load_pretrain_weights(self.model, config)
        self.model.reinitialize_attribute_head(len(attribute_names))
        # build_criterion_from_config requires a TrainConfig but only reads loss weights, never the paths
        train_config = TrainConfig(dataset_dir=".", output_dir=".")
        self.criterion, self.postprocess = build_criterion_from_config(config, train_config)

    def forward(self, images: Float[Tensor, "B 3 H W"]) -> dict:
        return self.model(images)

    def loss(self, outputs: dict, targets: list[pt.dataset.Target]) -> Tensor:
        criterion_targets = []
        for target in targets:
            boxes_cxcywh = box_convert(target["boxes_xywh"], "xywh", "cxcywh")
            criterion_targets.append(
                {
                    "boxes": boxes_cxcywh,
                    "labels": target["labels"],
                    "attributes": target["attributes"],
                }
            )
        losses = self.criterion(outputs, criterion_targets)
        weights = self.criterion.weight_dict
        detection_loss = sum(losses[name] * weights[name] for name in losses if name in weights)
        return detection_loss + self.attribute_loss_weight * self.attribute_loss(outputs, criterion_targets)

    def attribute_loss(self, outputs: dict, targets: list[dict[str, Tensor]]) -> Tensor:
        if not self.attribute_names or "pred_attributes" not in outputs:
            return outputs["pred_logits"].new_zeros(())

        num_boxes = self.criterion.num_boxes_for_targets(outputs, targets)
        loss = self.attribute_loss_for_outputs(outputs, targets, num_boxes)
        for aux_outputs in outputs.get("aux_outputs", []):
            loss = loss + self.attribute_loss_for_outputs(aux_outputs, targets, num_boxes)
        if "enc_outputs" in outputs:
            loss = loss + self.attribute_loss_for_outputs(outputs["enc_outputs"], targets, num_boxes)
        return loss

    def attribute_loss_for_outputs(
        self,
        outputs: dict,
        targets: list[dict[str, Tensor]],
        num_boxes: Tensor,
    ) -> Tensor:
        pred_attributes = outputs.get("pred_attributes")
        if pred_attributes is None:
            return outputs["pred_logits"].new_zeros(())

        group_detr = self.criterion.group_detr if self.criterion.training else 1
        indices = self.criterion.matcher(outputs, targets, group_detr=group_detr)
        if not any(len(src) for src, _target in indices):
            return pred_attributes.sum() * 0.0

        src_idx = self.criterion._get_src_permutation_idx(indices)
        matched_predictions = pred_attributes[src_idx]
        matched_targets = torch.cat(
            [target["attributes"][target_idx] for target, (_src_idx, target_idx) in zip(targets, indices, strict=True)]
        ).to(device=matched_predictions.device, dtype=matched_predictions.dtype)
        matched_labels = torch.cat(
            [target["labels"][target_idx] for target, (_src_idx, target_idx) in zip(targets, indices, strict=True)]
        ).to(device=matched_predictions.device)
        eligible = matched_labels[:, None] == self.attribute_class_index.to(matched_predictions.device)
        losses = F.binary_cross_entropy_with_logits(
            matched_predictions,
            matched_targets,
            pos_weight=self.attribute_pos_weight.to(matched_predictions.device),
            reduction="none",
        )
        losses = losses * eligible.to(losses.dtype)
        return losses.sum() / num_boxes

    @classmethod
    def load(cls, checkpoint: Path, device: torch.device) -> "CourtDetector":
        """Build the model from the checkpoint's args.json sidecar and load its weights."""
        metadata_path = checkpoint.with_name("args.json")
        metadata = json.loads(metadata_path.read_text())
        image_size_config = metadata["image_size"]
        image_size = (image_size_config["height"], image_size_config["width"])
        attribute_names = tuple(metadata.get("attributes", ()))
        model = cls(
            tuple(metadata["classes"]),
            image_size,
            pretrained=False,
            attribute_names=attribute_names,
            attribute_loss_weight=metadata.get("attribute_loss_weight", ATTRIBUTE_LOSS_WEIGHT),
            attribute_pos_weight=metadata.get("attribute_pos_weight"),
        )
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict.setdefault("attribute_pos_weight", model.attribute_pos_weight)
        state_dict.setdefault("attribute_class_index", model.attribute_class_index)
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
        threshold: float = 0.0,
        nms_iou: float | None = None,
        max_detections: int | None = None,
    ) -> list[detection.inference.Prediction]:
        """Per-image detections with normalized xyxy boxes as numpy arrays."""
        variant_tensors = []
        image_indexes = []
        flipped_flags = []
        for image_index, image in enumerate(images):
            for scale in scales:
                height, width = scaled_image_size(self.image_size, scale)
                resized = image.resize((width, height), Image.Resampling.BILINEAR)
                variant_tensors.append(image2tensor(resized))
                image_indexes.append(image_index)
                flipped_flags.append(False)
                if hflip:
                    flipped_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    resized = flipped_image.resize((width, height), Image.Resampling.BILINEAR)
                    variant_tensors.append(image2tensor(resized))
                    image_indexes.append(image_index)
                    flipped_flags.append(True)

        return detection.inference.predict(
            self,
            torch.stack(variant_tensors).to(self.device),
            image_indexes,
            flipped_flags,
            len(images),
            threshold=threshold,
            nms_iou=nms_iou,
            max_detections=max_detections,
        )

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
