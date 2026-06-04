from torch import tensor

MASK_NAMES = (
    "left_court",
    "right_court",
    "left_3pt",
    "right_3pt",
    "left_paint",
    "right_paint",
)
LEFT_RIGHT_PAIRS = ((0, 1), (2, 3), (4, 5))
IMAGE_MEAN = tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
