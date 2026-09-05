from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


@dataclass(frozen=True)
class ARKitFrame:
    """One synchronized video frame, calibrated camera, and ARKit pose."""

    frame_index: int
    record_slot: int
    sensor_sec: float
    utc_sec: float
    tx_m: float
    ty_m: float
    tz_m: float
    qw: float
    qx: float
    qy: float
    qz: float
    exposure_sec: float
    tracking_state: str
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width_px: int
    height_px: int
    image_name: str = ""

    @property
    def translation(self) -> np.ndarray:
        return np.asarray([self.tx_m, self.ty_m, self.tz_m], dtype=np.float64)

    @property
    def quaternion_wxyz(self) -> np.ndarray:
        q = np.asarray([self.qw, self.qx, self.qy, self.qz], dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            raise ValueError(f"Frame {self.frame_index} has a zero quaternion.")
        return q / norm

    @property
    def world_from_camera(self) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = quaternion_wxyz_to_rotation(self.quaternion_wxyz)
        transform[:3, 3] = self.translation
        return transform

    @property
    def camera_from_world(self) -> np.ndarray:
        return invert_transform(self.world_from_camera)


_NUMERIC_TYPES = {
    "frame_index": int,
    "record_slot": int,
    "sensor_sec": float,
    "utc_sec": float,
    "tx_m": float,
    "ty_m": float,
    "tz_m": float,
    "qw": float,
    "qx": float,
    "qy": float,
    "qz": float,
    "exposure_sec": float,
    "fx_px": float,
    "fy_px": float,
    "cx_px": float,
    "cy_px": float,
    "width_px": int,
    "height_px": int,
}


def _pose_csv_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path / "arkit_pose.csv" if path.is_dir() else path


def read_arkit_poses(path: str | Path) -> list[ARKitFrame]:
    """Read Sensor Logger's commented ``arkit_pose.csv`` format."""

    csv_path = _pose_csv_path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"ARKit pose CSV not found: {csv_path}")
    with csv_path.open(newline="") as handle:
        rows = (line for line in handle if line.strip() and not line.startswith("#"))
        reader = csv.DictReader(rows)
        records = []
        for row in reader:
            converted = {
                key: converter(row[key]) for key, converter in _NUMERIC_TYPES.items()
            }
            converted["tracking_state"] = row["tracking_state"]
            converted["image_name"] = row.get("image_name", "")
            records.append(ARKitFrame(**converted))
    if not records:
        raise ValueError(f"No ARKit poses found in: {csv_path}")
    return records


def read_frame_manifest(path: str | Path) -> list[ARKitFrame]:
    """Read the manifest written by :func:`prepare_localization_frames`."""

    return read_arkit_poses(path)


def quaternion_wxyz_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-12:
        raise ValueError("Cannot convert a zero quaternion to a rotation.")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""

    r = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.asarray([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q = np.asarray([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q = np.asarray([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q = np.asarray([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def first_pose_alignment(
    reference_first_world_from_camera: np.ndarray,
    query_first_world_from_camera: np.ndarray,
) -> np.ndarray:
    """Return ``T_reference_world_query_world`` from a shared first camera pose."""

    return np.asarray(reference_first_world_from_camera) @ invert_transform(
        query_first_world_from_camera
    )


def rotation_error_degrees(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    relative = np.asarray(estimated) @ np.asarray(ground_truth).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _write_manifest(path: Path, records: list[ARKitFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _extract_sampled_frames(
    recording_dir: Path,
    image_dir: Path,
    selected: list[ARKitFrame],
    frame_stride: int,
    overwrite: bool,
) -> list[ARKitFrame]:
    video_path = recording_dir / "wide.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Wide camera video not found: {video_path}")
    image_dir.mkdir(parents=True, exist_ok=True)
    expected = [image_dir / f"frame_{record.frame_index:06d}.jpg" for record in selected]
    if all(path.is_file() for path in expected) and not overwrite:
        return [
            ARKitFrame(**{**asdict(record), "image_name": path.name})
            for record, path in zip(selected, expected)
        ]
    existing = list(image_dir.glob("frame_*.jpg"))
    if existing and not overwrite:
        raise RuntimeError(
            f"Partial extracted frame set in {image_dir}. Set overwrite=True to rebuild it."
        )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to extract frames from wide.mp4.")

    with TemporaryDirectory(prefix="indoor_vpr_frames_") as temporary:
        pattern = Path(temporary) / "selected_%06d.jpg"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select=not(mod(n\\,{frame_stride}))",
            "-fps_mode",
            "vfr",
            "-q:v",
            "2",
            str(pattern),
        ]
        subprocess.run(command, check=True)
        extracted = sorted(Path(temporary).glob("selected_*.jpg"))
        if len(extracted) != len(selected):
            raise RuntimeError(
                f"ffmpeg extracted {len(extracted)} frames but the ARKit CSV selected "
                f"{len(selected)}. Verify that video frame indices match arkit_pose.csv."
            )
        for source, destination in zip(extracted, expected):
            destination.unlink(missing_ok=True)
            shutil.move(str(source), destination)
    return [
        ARKitFrame(**{**asdict(record), "image_name": path.name})
        for record, path in zip(selected, expected)
    ]


def prepare_localization_frames(
    reference_recording: str | Path,
    query_recording: str | Path,
    image_root: str | Path,
    *,
    frame_stride: int = 5,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Extract synchronized reference/query JPEGs and write pose manifests.

    Returns ``(reference_manifest, query_manifest)``. Images are placed under
    ``image_root/reference`` and ``image_root/query`` so their relative paths are
    directly compatible with HLoc.
    """

    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")
    image_root = Path(image_root).expanduser()
    outputs = []
    for split, recording in (
        ("reference", Path(reference_recording).expanduser()),
        ("query", Path(query_recording).expanduser()),
    ):
        records = read_arkit_poses(recording)
        selected = records[::frame_stride]
        sampled = _extract_sampled_frames(
            recording, image_root / split, selected, frame_stride, overwrite
        )
        manifest = image_root / f"{split}_manifest.csv"
        _write_manifest(manifest, sampled)
        outputs.append(manifest)
    return outputs[0], outputs[1]
