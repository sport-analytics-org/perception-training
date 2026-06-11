import cv2
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from court_training.loss import dice_loss
from court_training.warp import warp

CENTERED_SOURCE = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float64)
CENTERED_TARGET = np.array([(0.20, 0.40), (0.80, 0.40), (1.05, 0.90), (-0.05, 0.90)], dtype=np.float64)


def fit_homography(
    source_masks: Float[Tensor, "... N H W"],
    target_masks: Float[Tensor, "... N H W"],
    initial_homography: Float[Tensor, "... 3 3"],
    multipliers: Float[Tensor, "*N"] | None = None,
) -> Float[Tensor, "... 3 3"]:
    if source_masks.shape[-3] != target_masks.shape[-3]:
        raise ValueError(f"Got {source_masks.shape[-3]} source masks and {target_masks.shape[-3]} target masks")

    device = target_masks.device
    weights = area_weights(target_masks, multipliers)
    initial = initial_homography.to(device=device, dtype=target_masks.dtype)
    params = torch.zeros(*target_masks.shape[:-3], 8, dtype=target_masks.dtype, device=device)
    params[..., 0] = 1
    params[..., 4] = 1
    params.requires_grad_()
    optimizer = torch.optim.LBFGS([params], lr=0.6, max_iter=420, history_size=20, line_search_fn="strong_wolfe")

    def closure() -> Float[Tensor, ""]:
        optimizer.zero_grad(set_to_none=True)
        homography = multiply_hom(initial, params)
        predicted = warp(source_masks, homography, target_masks.shape[-2:])
        loss = dice_loss(predicted, target_masks, weights)
        loss.backward()
        return loss

    optimizer.step(closure)
    return multiply_hom(initial, params.detach())


def find_keypoints_homography(
    source: Float[np.ndarray, "P 2"],
    target: Float[np.ndarray, "P 2"],
    ransac_threshold: float = 0.02,
) -> Float[np.ndarray, "3 3"]:
    if len(source) < 4:
        raise ValueError(f"Need at least 4 point pairs, got {len(source)}")

    method = cv2.RANSAC if len(source) > 4 else 0
    homography, _ = cv2.findHomography(
        source.astype(np.float64),
        target.astype(np.float64),
        method=method,
        ransacReprojThreshold=ransac_threshold,
    )
    if homography is None:
        raise ValueError("OpenCV could not estimate a homography from the provided points")
    return homography / homography[2, 2]


def centered_homography() -> Float[np.ndarray, "3 3"]:
    return find_keypoints_homography(CENTERED_SOURCE, CENTERED_TARGET)


def area_weights(
    target_masks: Float[Tensor, "... N H W"],
    multipliers: Float[Tensor, "*N"] | None = None,
) -> Float[Tensor, "... N"]:
    areas = target_masks.sum(dim=(-2, -1)).sqrt()
    if multipliers is None:
        multipliers = torch.ones_like(areas)
    multipliers = multipliers.to(device=target_masks.device, dtype=target_masks.dtype)
    weights = areas * multipliers
    return weights / weights.sum(dim=-1, keepdim=True)


def multiply_hom(hom: Float[Tensor, "... 3 3"], delta: Float[Tensor, "... 8"]) -> Float[Tensor, "... 3 3"]:
    update = torch.stack(
        [
            torch.stack([delta[..., 0], delta[..., 1], delta[..., 2]], dim=-1),
            torch.stack([delta[..., 3], delta[..., 4], delta[..., 5]], dim=-1),
            torch.stack([delta[..., 6], delta[..., 7], torch.ones_like(delta[..., 0])], dim=-1),
        ],
        dim=-2,
    )
    homography = hom @ update
    return homography / homography[..., 2:3, 2:3]
