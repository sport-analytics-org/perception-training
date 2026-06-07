from torch import tensor

IMAGE_MEAN = tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
