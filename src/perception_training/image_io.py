import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from perception_training.constants import IMAGE_MEAN, IMAGE_STD


def image2tensor(image: Image.Image) -> Float[Tensor, "3 H W"]:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def tensor2image(image: Float[Tensor, "3 H W"]) -> Image.Image:
    image_array = ((image.detach().cpu() * IMAGE_STD) + IMAGE_MEAN).clamp(0, 1)
    image_array = (image_array.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_array)
