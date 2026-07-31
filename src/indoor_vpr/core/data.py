from __future__ import annotations

"""Dataset configuration and image discovery shared by all algorithms."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class DatasetConfig:
    """Paths and loading options for one VPR experiment."""

    database_dir: str | Path
    query_dir: str | Path
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    max_database_images: int | None = None
    max_query_images: int | None = None


@dataclass(frozen=True)
class ImageDataset:
    database_paths: list[Path]
    query_paths: list[Path]

    @classmethod
    def from_config(cls, config: DatasetConfig) -> "ImageDataset":
        database_paths = _image_paths(
            config.database_dir, config.extensions, config.max_database_images
        )
        query_paths = _image_paths(
            config.query_dir, config.extensions, config.max_query_images
        )
        return cls(database_paths=database_paths, query_paths=query_paths)

    def summary(self) -> str:
        return (
            f"Database: {len(self.database_paths)} images | "
            f"Queries: {len(self.query_paths)} images"
        )


def _image_paths(
    directory: str | Path,
    extensions: Iterable[str],
    limit: int | None,
) -> list[Path]:
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")

    allowed = {extension.lower() for extension in extensions}
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed
    )
    if not paths:
        raise ValueError(f"No supported images found in: {directory}")
    return paths[:limit] if limit is not None else paths
