from torchmetrics.detection import MeanAveragePrecision


def summarize(metric: MeanAveragePrecision, class_names: tuple[str, ...]) -> dict[str, float | dict[str, float]]:
    values = metric.compute()
    per_class_map = {
        class_names[int(class_id)]: round(float(value), 4)
        for class_id, value in zip(values["classes"], values["map_per_class"], strict=True)
    }
    return {
        "map50_95": float(values["map"]),
        "map50": float(values["map_50"]),
        "map75": float(values["map_75"]),
        "per_class_map": per_class_map,
    }
