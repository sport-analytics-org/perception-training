import cv2
import numpy as np
import torch
from cnf.court.basket import BasketCourt
from jaxtyping import Bool, Float
from torch import Tensor

from perception_training.segmentation.loss import dice_loss
from perception_training.warp import warp

CENTERED_SOURCE = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float64)
CENTERED_TARGET = np.array([(0.20, 0.40), (0.80, 0.40), (1.05, 0.90), (-0.05, 0.90)], dtype=np.float64)
DEFAULT_MAX_ITERATIONS = 120


def fit_court(
    court: BasketCourt,
    mask_names: tuple[str, ...],
    keypoint_names: tuple[str, ...],
    probabilities: Float[Tensor, "N H W"],
    keypoints: Float[np.ndarray, "K 2"],
    visible: Bool[np.ndarray, "K"],
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[Float[Tensor, "3 3"], Float[Tensor, "N H W"], float]:
    """Fit a court homography to predicted masks, seeded from the visible keypoints.

    Returns the homography, the warped template masks, and their soft IoU against the predictions.
    """
    source_masks = template_masks(court, mask_names, probabilities.shape[-1], probabilities.device)
    source_keypoints = normalized_keypoints(court, keypoint_names)
    initial = torch.tensor(
        find_keypoints_homography(source_keypoints[visible], keypoints[visible]),
        dtype=probabilities.dtype,
    )
    multipliers = torch.tensor(
        [1.5 if "3pt_area" in name or "painted_area" in name else 1.0 for name in mask_names],
        device=probabilities.device,
    )
    matrix = fit_homography(source_masks, probabilities, initial, multipliers, max_iterations)
    fitted = warp(source_masks, matrix, probabilities.shape[-2:])
    return matrix, fitted, soft_iou(fitted, probabilities)


def fit_homography(
    source_masks: Float[Tensor, "... N H W"],
    target_masks: Float[Tensor, "... N H W"],
    initial_homography: Float[Tensor, "... 3 3"],
    multipliers: Float[Tensor, "*N"] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
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
    optimizer = torch.optim.LBFGS(
        [params],
        lr=0.6,
        max_iter=max_iterations,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

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


def normalized_keypoints(court: BasketCourt, labels: tuple[str, ...]) -> Float[np.ndarray, "K 2"]:
    points_by_name = court.keypoints()
    points = np.array([points_by_name[name] for name in labels], dtype=np.float64)
    x = (points[:, 0] + court.half_length) / court.length
    y = (points[:, 1] + court.half_width) / court.width
    return np.stack([x, y], axis=1)


def template_masks(
    court: BasketCourt,
    labels: tuple[str, ...],
    width: int,
    device: torch.device,
) -> Float[Tensor, "N H W"]:
    masks = []
    for label in labels:
        image = court.get_mask_image(label, width).convert("L")
        masks.append(torch.tensor(np.asarray(image, dtype=np.float32) / 255, device=device))
    return torch.stack(masks)


def soft_iou(predicted: Float[Tensor, "N H W"], target: Float[Tensor, "N H W"]) -> float:
    intersection = torch.minimum(predicted, target).sum(dim=(1, 2))
    union = torch.maximum(predicted, target).sum(dim=(1, 2)).clamp_min(1e-6)
    return (intersection / union).mean().item()


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
