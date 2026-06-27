import cv2
import numpy as np
from jaxtyping import Bool
from PIL import Image, ImageDraw

CONTOUR_EPSILON_RATIOS = (0.0015, 0.003, 0.006, 0.01, 0.02, 0.04, 0.08, 0.12)
HIGH_IOU_THRESHOLD = 0.99
MAX_POLYGON_POINTS = 20
APPROXIMATION_SEARCH_STEPS = 24
SIMPLIFICATION_IOU_TOLERANCE = 0.002

Point = tuple[float, float]


def mask_polygon(mask: Bool[np.ndarray, "H W"]) -> list[Point]:
    """Recover the largest mask component as normalized ``(x, y)`` polygon points."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, closed=True)
    candidates = []
    for ratio in CONTOUR_EPSILON_RATIOS:
        epsilon = ratio * perimeter
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(polygon) > MAX_POLYGON_POINTS:
            low = epsilon
            high = perimeter
            for _ in range(APPROXIMATION_SEARCH_STEPS):
                mid = (low + high) / 2
                candidate = cv2.approxPolyDP(contour, mid, closed=True)
                if len(candidate) > MAX_POLYGON_POINTS:
                    low = mid
                else:
                    high = mid
                    polygon = candidate
        if len(polygon) < 3:
            continue

        points = [(int(point[0][0]), int(point[0][1])) for point in polygon]
        image = Image.new("L", (mask.shape[1], mask.shape[0]), 0)
        ImageDraw.Draw(image).polygon(points, fill=1)
        rasterized = np.array(image, dtype=bool)
        intersection = np.logical_and(mask, rasterized).sum()
        union = np.logical_or(mask, rasterized).sum()
        candidates.append((float(intersection / union), len(points), points))

    best_iou = max(iou for iou, _, _ in candidates)
    min_iou = best_iou
    if best_iou >= HIGH_IOU_THRESHOLD:
        min_iou = max(HIGH_IOU_THRESHOLD, best_iou - SIMPLIFICATION_IOU_TOLERANCE)
    accurate = [candidate for candidate in candidates if candidate[0] >= min_iou]
    _, _, points = min(accurate, key=lambda x: x[1])

    height, width = mask.shape
    x_scale = width - 1
    y_scale = height - 1
    normalized = []
    for x, y in points:
        nx = 0.0 if x <= 0 else 1.0 if x >= x_scale else (x + 0.5) / x_scale
        ny = 0.0 if y <= 0 else 1.0 if y >= y_scale else (y + 0.5) / y_scale
        normalized.append((nx, ny))
    return normalized


def projected_rectangle_mask_polygon(mask: Bool[np.ndarray, "H W"]) -> list[Point]:
    """Recover a clipped projected rectangle as normalized ``(x, y)`` polygon points."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, closed=True)
    touches_top = bool(mask[0, :].any())
    touches_bottom = bool(mask[-1, :].any())
    touches_left = bool(mask[:, 0].any())
    touches_right = bool(mask[:, -1].any())
    border_count = sum((touches_top, touches_bottom, touches_left, touches_right))
    max_points = 4 + border_count
    height, width = mask.shape
    diagonal = (width**2 + height**2) ** 0.5
    candidates = []
    for ratio in CONTOUR_EPSILON_RATIOS:
        epsilon = ratio * perimeter
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(polygon) > max_points:
            low = epsilon
            high = perimeter
            for _ in range(APPROXIMATION_SEARCH_STEPS):
                mid = (low + high) / 2
                candidate = cv2.approxPolyDP(contour, mid, closed=True)
                if len(candidate) > max_points:
                    low = mid
                else:
                    high = mid
                    polygon = candidate
        if not 3 <= len(polygon) <= max_points:
            continue
        polygon_contour = polygon[:, 0, :].astype(np.float32)
        distances = []
        for point in contour:
            pixel = (int(point[0][0]), int(point[0][1]))
            distance = abs(cv2.pointPolygonTest(polygon_contour, pixel, True))
            distances.append(distance / diagonal)
        error = max(distances)
        points = [(int(point[0][0]), int(point[0][1])) for point in polygon]
        candidates.append((error, points))

    _, points = min(candidates, key=lambda x: x[0])
    x_scale = width - 1
    y_scale = height - 1
    normalized = []
    for x, y in points:
        nx = 0.0 if x <= 0 else 1.0 if x >= x_scale else (x + 0.5) / x_scale
        ny = 0.0 if y <= 0 else 1.0 if y >= y_scale else (y + 0.5) / y_scale
        normalized.append((nx, ny))
    return normalized
