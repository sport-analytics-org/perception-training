from jaxtyping import Float
from torch import Tensor


def dice_loss(
    predicted: Float[Tensor, "... masks H W"],
    target: Float[Tensor, "... masks H W"],
    weights: Float[Tensor, "... masks"],
) -> Float[Tensor, ""]:
    intersection = (predicted * target).sum(dim=(-2, -1))
    denominator = predicted.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2 * intersection + 1) / (denominator + 1)
    return (1 - (dice * weights).sum(dim=-1)).mean()
