import numpy as np
import torch
from jaxtyping import Float
from sportanalytics.court.basket import BasketCourt
from torch import Tensor
from torch.nn import functional as F

from court_training.loss import dice_loss
from court_training.warp import warp

CENTERED_SOURCE = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float64)
CENTERED_TARGET = np.array([(0.20, 0.40), (0.80, 0.40), (1.05, 0.90), (-0.05, 0.90)], dtype=np.float64)


def fit_homography(
    target_masks: Float[Tensor, "N H W"],
    court_template: BasketCourt,
    mask_names: tuple[str, ...],
    initial_homography: Float[np.ndarray, "3 3"],
    size: int = 384,
    boosted_masks: tuple[str, ...] = (),
    boost: float = 1.0,
) -> Float[np.ndarray, "3 3"]:
    if target_masks.shape[0] != len(mask_names):
        raise ValueError(f"Got {target_masks.shape[0]} target masks for {len(mask_names)} mask names")

    device = target_masks.device
    source_masks = _template_masks(court_template, mask_names, size, device)
    target_masks = _resize_masks(target_masks.float(), size)
    weights = area_weights(mask_names, target_masks, boosted_masks, boost)
    initial = torch.tensor(initial_homography, dtype=torch.float32, device=device)
    params = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([params], lr=0.6, max_iter=420, history_size=20, line_search_fn="strong_wolfe")

    def closure() -> Float[Tensor, ""]:
        optimizer.zero_grad(set_to_none=True)
        homography = _compose_homography(initial, params)
        predicted = warp(source_masks, homography, target_masks.shape[-2:])
        loss = dice_loss(predicted, target_masks, weights)
        loss.backward()
        return loss

    optimizer.step(closure)
    homography = _compose_homography(initial, params.detach()).cpu().numpy()
    return homography / homography[2, 2]


def render_masks(
    homography: Float[np.ndarray, "3 3"],
    court_template: BasketCourt,
    mask_names: tuple[str, ...],
    output_shape: tuple[int, int],
    size: int = 384,
) -> Float[Tensor, "N H W"]:
    source = _template_masks(court_template, mask_names, size, torch.device("cpu"))
    return warp(source, torch.tensor(homography, dtype=torch.float32), output_shape)


def keypoint_homography(
    court_template: BasketCourt,
    keypoint_names: tuple[str, ...],
    keypoints: Float[np.ndarray, "K 2"],
    visibility: Float[np.ndarray, "*K"],
    threshold: float = 0.5,
) -> Float[np.ndarray, "3 3"]:
    source = normalized_template_keypoints(court_template, keypoint_names)
    visible = visibility >= threshold
    if visible.sum() < 4:
        raise ValueError(f"Only {visible.sum()} keypoints are visible; need at least 4")
    return homography_from_points(source[visible], keypoints[visible], visibility[visible])


def centered_homography() -> Float[np.ndarray, "3 3"]:
    return homography_from_points(CENTERED_SOURCE, CENTERED_TARGET)


def homography_from_points(
    source: Float[np.ndarray, "P 2"],
    target: Float[np.ndarray, "P 2"],
    weights: Float[np.ndarray, "*P"] | None = None,
) -> Float[np.ndarray, "3 3"]:
    if len(source) < 4:
        raise ValueError(f"Need at least 4 point pairs, got {len(source)}")
    if weights is None:
        weights = np.ones(len(source), dtype=np.float64)

    rows = []
    values = []
    for (x, y), (u, v), weight in zip(source, target, weights, strict=True):
        scale = float(np.sqrt(weight))
        rows.append([scale * x, scale * y, scale, 0, 0, 0, -scale * u * x, -scale * u * y])
        rows.append([0, 0, 0, scale * x, scale * y, scale, -scale * v * x, -scale * v * y])
        values.extend((scale * u, scale * v))

    params, *_ = np.linalg.lstsq(np.array(rows), np.array(values), rcond=None)
    homography = np.array(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0],
        ],
        dtype=np.float64,
    )
    return homography / homography[2, 2]


def normalized_template_keypoints(
    court_template: BasketCourt,
    keypoint_names: tuple[str, ...],
) -> Float[np.ndarray, "K 2"]:
    points_by_name = court_template.keypoints()
    points = np.array([points_by_name[name] for name in keypoint_names], dtype=np.float64)
    x = (points[:, 0] + court_template.half_length) / court_template.length
    y = (points[:, 1] + court_template.half_width) / court_template.width
    return np.stack([x, y], axis=1)


def area_weights(
    mask_names: tuple[str, ...],
    target_masks: Float[Tensor, "N H W"],
    boosted_masks: tuple[str, ...] = (),
    boost: float = 1.0,
) -> Float[Tensor, "*N"]:
    areas = target_masks.sum(dim=(1, 2)).sqrt()
    multipliers = torch.tensor(
        [boost if any(name in mask_name for name in boosted_masks) else 1.0 for mask_name in mask_names],
        dtype=target_masks.dtype,
        device=target_masks.device,
    )
    weights = areas * multipliers
    return weights / weights.sum()


def _template_masks(
    court_template: BasketCourt,
    mask_names: tuple[str, ...],
    width: int,
    device: torch.device,
) -> Float[Tensor, "N H W"]:
    masks = []
    for name in mask_names:
        image = court_template.get_mask_image(name, width).convert("L")
        masks.append(torch.tensor(np.asarray(image, dtype=np.float32) / 255, device=device))
    return torch.stack(masks)


def _resize_masks(masks: Float[Tensor, "N H W"], size: int) -> Float[Tensor, "N size size"]:
    return F.interpolate(masks[None], size=(size, size), mode="area")[0]


def _compose_homography(initial: Float[Tensor, "3 3"], params: Float[Tensor, "8"]) -> Float[Tensor, "3 3"]:
    update = torch.stack(
        [
            torch.stack([params[0], params[1], params[2]]),
            torch.stack([params[3], params[4], params[5]]),
            torch.stack([params[6], params[7], torch.ones_like(params[0])]),
        ]
    )
    homography = initial @ update
    return homography / homography[2, 2]
