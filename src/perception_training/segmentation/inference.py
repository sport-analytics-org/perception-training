from typing import Literal, NotRequired, TypedDict

import numpy as np
import sportkit as sk
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

import perception_training as pt
from perception_training.flip import flip_torch

CourtType = Literal["nba", "fiba"]
COURTS: dict[CourtType, sk.courts.BasketCourt] = {
    "nba": sk.courts.NbaCourt,
    "fiba": sk.courts.FibaCourt,
}


class Prediction(TypedDict):
    masks: Float[np.ndarray, "B N H W"]
    keypoints: Float[np.ndarray, "B K 2"]
    visibility: Float[np.ndarray, "B K"]
    homography: NotRequired[Float[np.ndarray, "B 3 3"]]


def predict(
    model: nn.Module,
    images: Float[Tensor, "B 3 H W"],
    mask_names: tuple[str, ...],
    keypoint_names: tuple[str, ...],
    scales: tuple[float, ...] = (1.0,),
    hflip: bool = True,
    fit_homography: bool = False,
    court_type: CourtType = "nba",
    output_size: tuple[int, int] | None = None,
) -> Prediction:
    inference_size = images.shape[-2:]
    masks_by_scale = []
    keypoints_by_scale = []
    visibility_by_scale = []
    for scale in scales:
        scaled_images = F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)
        prediction = model(scaled_images)
        assert len(mask_names) == prediction["masks"].shape[1]
        if "keypoints" in prediction:
            assert len(keypoint_names) == prediction["keypoints"].shape[1]
        else:
            assert not keypoint_names

        masks = prediction["masks"]
        if hflip:
            flipped_prediction = model(flip_torch(image=scaled_images)["image"])
            flipped = flip_torch(
                masks=flipped_prediction["masks"],
                keypoints=flipped_prediction.get("keypoints"),
                visibility=flipped_prediction.get("visibility"),
                mask_names=mask_names,
                keypoint_names=keypoint_names,
            )
            masks = (masks + flipped["masks"]) / 2
        masks_by_scale.append(F.interpolate(masks, size=inference_size, mode="bilinear", align_corners=False))
        if "keypoints" in prediction:
            keypoints = prediction["keypoints"]
            visibility = prediction["visibility"]
            if hflip:
                keypoints = (keypoints + flipped["keypoints"]) / 2
                visibility = (visibility + flipped["visibility"]) / 2
            keypoints_by_scale.append(keypoints)
            visibility_by_scale.append(visibility)

    output = {"masks": torch.stack(masks_by_scale).mean(dim=0)}
    if keypoints_by_scale:
        output["keypoints"] = torch.stack(keypoints_by_scale).mean(dim=0)
        output["visibility"] = torch.stack(visibility_by_scale).mean(dim=0)
    if fit_homography:
        output = fit_homography_to_masks(output, mask_names, keypoint_names, COURTS[court_type])
    if output_size is not None:
        output["masks"] = F.interpolate(output["masks"], size=output_size, mode="bilinear", align_corners=False)

    prediction: Prediction = {
        "masks": output["masks"].sigmoid().cpu().numpy().astype(np.float32),
        "keypoints": output["keypoints"].cpu().numpy().astype(np.float32),
        "visibility": output["visibility"].sigmoid().cpu().numpy().astype(np.float32),
    }
    if "homography" in output:
        prediction["homography"] = output["homography"].cpu().numpy().astype(np.float32)
    return prediction


def fit_homography_to_masks(
    prediction: dict[str, Tensor],
    mask_names: tuple[str, ...],
    keypoint_names: tuple[str, ...],
    court: sk.courts.BasketCourt,
) -> dict[str, Tensor]:
    mask_names = tuple(court.planar_areas())
    mask_count = len(mask_names)
    target_masks = prediction["masks"][:, :mask_count].sigmoid()
    source_keypoints = pt.homography.normalized_keypoints(court, keypoint_names)
    source_keypoints_tensor = torch.as_tensor(source_keypoints, dtype=target_masks.dtype, device=target_masks.device)
    width = target_masks.shape[-1]
    masks = []
    for label in mask_names:
        image = court.get_mask_image(label, width).convert("L")
        masks.append(torch.as_tensor(np.asarray(image) / 255, dtype=target_masks.dtype, device=target_masks.device))
    source_masks = torch.stack(masks)

    predicted_keypoints = prediction["keypoints"]
    initial_homographies = [
        pt.homography.find_keypoints_homography(source_keypoints, keypoints.detach().cpu().numpy())
        for keypoints in predicted_keypoints
    ]
    initial_homographies = np.stack(initial_homographies)
    initial_homographies = torch.as_tensor(initial_homographies, dtype=source_masks.dtype, device=source_masks.device)
    source_masks = source_masks.expand(*target_masks.shape[:-3], -1, -1, -1)
    homographies = pt.homography.fit_homography(source_masks, target_masks, initial_homographies)
    probabilities = pt.warp.warp(source_masks, homographies, target_masks.shape[-2:]).clamp(1e-4, 1 - 1e-4)
    ones = torch.ones(len(source_keypoints_tensor), 1, dtype=target_masks.dtype, device=target_masks.device)
    homogeneous = torch.cat((source_keypoints_tensor, ones), dim=1)
    projected = torch.einsum("kd,bhd->bkh", homogeneous, homographies)
    keypoints = projected[:, :, :2] / projected[:, :, 2:]
    visibility = ((keypoints >= 0) & (keypoints <= 1)).all(dim=2)
    return {
        "masks": torch.logit(probabilities),
        "keypoints": keypoints,
        "visibility": visibility.to(dtype=target_masks.dtype),
        "homography": homographies,
    }
