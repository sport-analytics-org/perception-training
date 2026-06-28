import json
from pathlib import Path

import numpy as np
import sportkit as sk
import typer
from jaxtyping import Float

app = typer.Typer(help="Generate dataset keypoint shards from homography shards.")

HOMOGRAPHY_ARGUMENT = typer.Argument(help="Dataset homography shard JSON.")
OUTPUT_OPTION = typer.Option(None, help="Output keypoint shard JSON. Defaults to the matching keypoints shard.")
COURTS = {
    "fiba": sk.FibaCourt,
    "nba": sk.NbaCourt,
}
KEYPOINT_NAMES = tuple(sk.NbaCourt.keypoints())


@app.command()
def main(
    homography: Path = HOMOGRAPHY_ARGUMENT,
    output: Path | None = OUTPUT_OPTION,
) -> None:
    homography_path = homography.expanduser().resolve()
    output_path = output.expanduser().resolve() if output else keypoint_path_for(homography_path)

    data = json.loads(homography_path.read_text())
    keypoints = {}
    for image_key, entry in data["homographies"].items():
        court = COURTS[entry["court"]]
        matrix = np.asarray(entry["matrix"], dtype=np.float64)
        points, visibility = project_keypoints(court, matrix)
        keypoints[image_key] = {
            "court": entry["court"],
            "points": [
                {"position": point.tolist(), "visible": bool(visible)}
                for point, visible in zip(points, visibility, strict=True)
            ],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"keypoints": keypoints}, indent=2) + "\n")


def keypoint_path_for(homography_path: Path) -> Path:
    parts = list(homography_path.parts)
    index = parts.index("homography")
    parts[index] = "keypoints"
    return Path(*parts)


def project_keypoints(
    court: sk.BasketCourt,
    homography: Float[np.ndarray, "3 3"],
) -> tuple[Float[np.ndarray, "K 2"], np.ndarray]:
    points = normalized_keypoints(court)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    projected = homogeneous @ homography.T
    keypoints = projected[:, :2] / projected[:, 2:]
    visibility = np.logical_and.reduce(
        [
            keypoints[:, 0] >= 0,
            keypoints[:, 0] <= 1,
            keypoints[:, 1] >= 0,
            keypoints[:, 1] <= 1,
        ]
    )
    return keypoints, visibility


def normalized_keypoints(court: sk.BasketCourt) -> Float[np.ndarray, "K 2"]:
    points_by_name = court.keypoints()
    points = np.array([points_by_name[name] for name in KEYPOINT_NAMES], dtype=np.float64)
    x = (points[:, 0] + court.half_length) / court.length
    y = (points[:, 1] + court.half_width) / court.width
    return np.stack([x, y], axis=1)


if __name__ == "__main__":
    app()
