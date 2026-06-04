from pathlib import Path

import numpy as np
import torch
from PIL import Image

from court_training.constants import TTA_SCALES
from court_training.data import image_to_tensor
from court_training.model import DinoSegmenter, predict_multiscale


class CourtSegmenter:
    def __init__(
        self,
        model: DinoSegmenter,
        device: str | torch.device | None = None,
        tta_scales: tuple[float, ...] = TTA_SCALES,
    ) -> None:
        self.device = torch.device(device) if device is not None else default_device()
        self.model = model.to(self.device).eval()
        self.tta_scales = tta_scales

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device | None = None) -> "CourtSegmenter":
        checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
        model = DinoSegmenter(pretrained=False)
        model.load_state_dict(state_dict_from_checkpoint(checkpoint))
        return cls(model, device=device)

    @torch.inference_mode()
    def predict_proba(self, image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
        image_tensor = image_to_tensor(load_image(image)).unsqueeze(0).to(self.device)
        logits = predict_multiscale(self.model, image_tensor, self.tta_scales)
        probabilities = logits.sigmoid().squeeze(0).permute(1, 2, 0)
        return probabilities.cpu().numpy()

    def predict_masks(self, image: str | Path | Image.Image | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba(image) >= threshold

    def predict_bitfield(self, image: str | Path | Image.Image | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        masks = self.predict_masks(image, threshold)
        bitfield = np.zeros(masks.shape[:2], dtype=np.uint8)
        for bit in range(masks.shape[-1]):
            bitfield[masks[..., bit]] |= np.uint8(1 << bit)
        return bitfield


def load_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.open(Path(image).expanduser()).convert("RGB")


def state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary, got {type(checkpoint).__name__}")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state_dict.items()):
        raise TypeError("Checkpoint does not contain a model state dict")
    return state_dict


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
