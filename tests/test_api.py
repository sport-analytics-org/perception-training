import numpy as np
import torch
from PIL import Image

from perception_training import api


class FakeSegmenter:
    mask_names = ("painted_area", "free_throw_circle", "left_backboard")
    keypoint_names = ("a", "b", "c", "d")

    def predict(self, images):
        masks = np.zeros((1, 3, 8, 8), dtype=np.float32)
        masks[0, 0, 1:4, 1:4] = 1
        masks[0, 1, 4:7, 1:4] = 1
        masks[0, 2, 2:6, 5:7] = 1
        return {
            "masks": masks,
            "keypoints": np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32),
            "visibility": np.ones((1, 4), dtype=np.float32),
        }


def test_predict_segmentation_fits_only_non_backboard_masks(monkeypatch):
    captured = {}

    def fit_homography(probabilities, keypoints, visibility, mask_names, keypoint_names, court_type, max_iterations):
        captured["probabilities"] = probabilities
        captured["mask_names"] = mask_names
        return api.Homography(court=court_type, matrix=torch.eye(3).tolist(), soft_iou=1), probabilities

    monkeypatch.setattr(api, "fit_homography", fit_homography)

    prediction = api.predict_segmentation(FakeSegmenter(), Image.new("RGB", (8, 8)), "nba", 0.5, 10)

    assert captured["probabilities"].shape == (2, 8, 8)
    assert captured["mask_names"] == ("painted_area", "free_throw_circle")
    assert {polygon.label for polygon in prediction.polygons} == {
        "painted_area",
        "free_throw_circle",
        "left_backboard",
    }
