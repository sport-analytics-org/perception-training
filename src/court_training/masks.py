from sportanalytics import BasketCourt, NbaCourt

BASKETBALL_AREA_ORDER = ("court", "3pt_area", "painted_area")


def court_mask_names(court: BasketCourt, area_order: tuple[str, ...]) -> tuple[str, ...]:
    area_names = set(court.areas())
    names = []
    for area in area_order:
        for side in ("left", "right"):
            name = f"{side}_{area}"
            if name in area_names:
                names.append(name)
    return tuple(names)


def left_right_pairs(names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    by_name = {name: index for index, name in enumerate(names)}
    pairs = []
    for left_name, left_index in by_name.items():
        if not left_name.startswith("left_"):
            continue
        right_name = f"right_{left_name.removeprefix('left_')}"
        if right_name in by_name:
            pairs.append((left_index, by_name[right_name]))
    return tuple(pairs)


MASK_NAMES = court_mask_names(NbaCourt, BASKETBALL_AREA_ORDER)
LEFT_RIGHT_PAIRS = left_right_pairs(MASK_NAMES)
