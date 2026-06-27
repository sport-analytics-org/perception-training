# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Backward-compatibility shim — rfdetr.models.segmentation_head is deprecated; use rfdetr.models.heads.segmentation."""

from rfdetr.utilities.decorators import _warn_deprecated_module

_warn_deprecated_module(
    "rfdetr.models.segmentation_head", "rfdetr.models.heads.segmentation", deprecated_in="1.6.0", remove_in="1.9.0"
)

from perception_training.vendor.rfdetr_models.heads.segmentation import DepthwiseConvBlock, MLPBlock, SegmentationHead  # noqa: F401, E402
