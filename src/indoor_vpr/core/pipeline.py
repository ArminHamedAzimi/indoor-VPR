from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .algorithm import VPRAlgorithm
from .data import ImageDataset


@dataclass(frozen=True)
class VPRResults:
    dataset: ImageDataset
    similarity: np.ndarray
    ranked_database_indices: np.ndarray

    @property
    def best_indices(self) -> np.ndarray:
        return self.ranked_database_indices[:, 0]

    @property
    def best_scores(self) -> np.ndarray:
        return self.similarity[np.arange(len(self.dataset.query_paths)), self.best_indices]

    def top_indices(self, query_index: int, top_k: int = 5) -> np.ndarray:
        return self.ranked_database_indices[query_index, :top_k]


class SimilarityCSVView:
    """Lazy row access for a wide similarity CSV written incrementally."""

    def __init__(self, csv_path: str | Path, query_count: int, database_count: int) -> None:
        self.csv_path = Path(csv_path)
        self.query_count = query_count
        self.database_count = database_count
        self._row_cache: dict[int, np.ndarray] = {}

    @property
    def shape(self) -> tuple[int, int]:
        return self.query_count, self.database_count

    def _load_row(self, query_index: int) -> np.ndarray:
        if not 0 <= query_index < self.query_count:
            raise IndexError(
                f"query_index must be between 0 and {self.query_count - 1}"
            )
        if query_index in self._row_cache:
            return self._row_cache[query_index]

        with self.csv_path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"Similarity CSV is empty: {self.csv_path}")
            for row_number, row in enumerate(reader):
                if row_number == query_index:
                    values = np.asarray(row[2:], dtype=np.float32)
                    self._row_cache[query_index] = values
                    return values
        raise IndexError(f"query_index out of range for CSV: {query_index}")

    def row(self, query_index: int) -> np.ndarray:
        """Return one full similarity row as a NumPy array."""

        return self._load_row(query_index)

    def __getitem__(self, key: tuple[int, int]) -> float:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("SimilarityCSVView expects [query_index, database_index].")
        query_index, database_index = key
        row = self._load_row(int(query_index))
        return float(row[int(database_index)])

    def top_indices(self, query_index: int, top_k: int = 5) -> np.ndarray:
        """Return the top-k database indices for one query row."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        row = self._load_row(query_index)
        if top_k >= len(row):
            return np.argsort(-row)
        candidate_indices = np.argpartition(row, -top_k)[-top_k:]
        return candidate_indices[np.argsort(-row[candidate_indices])]


@dataclass(frozen=True)
class StreamedVPRResults:
    dataset: ImageDataset
    similarity: SimilarityCSVView
    ranked_database_indices: np.ndarray

    @property
    def best_indices(self) -> np.ndarray:
        return self.ranked_database_indices[:, 0]

    @property
    def best_scores(self) -> np.ndarray:
        return np.asarray(
            [self.similarity[query_index, database_index] for query_index, database_index in enumerate(self.best_indices)],
            dtype=np.float32,
        )

    def top_indices(self, query_index: int, top_k: int = 5) -> np.ndarray:
        return self.ranked_database_indices[query_index, :top_k]


def _l2_normalize(descriptors: np.ndarray) -> np.ndarray:
    descriptors = np.asarray(descriptors, dtype=np.float32)
    if descriptors.ndim != 2:
        raise ValueError("Descriptors must have shape (number_of_images, descriptor_size).")
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return descriptors / np.maximum(norms, 1e-12)


def run_vpr(dataset: ImageDataset, algorithm: VPRAlgorithm) -> VPRResults:
    """Encode both sets and rank database images using cosine similarity."""

    database_descriptors = _l2_normalize(algorithm.encode_database(dataset.database_paths))
    query_descriptors = _l2_normalize(algorithm.encode(dataset.query_paths))
    if database_descriptors.shape[1] != query_descriptors.shape[1]:
        raise ValueError("Database and query descriptor sizes do not match.")
    similarity = np.asarray(
        algorithm.similarity(query_descriptors, database_descriptors),
        dtype=np.float32,
    )
    expected_shape = (len(dataset.query_paths), len(dataset.database_paths))
    if similarity.shape != expected_shape:
        raise ValueError(
            f"Algorithm similarity returned shape {similarity.shape}; expected {expected_shape}."
        )
    ranking = np.argsort(-similarity, axis=1)
    return VPRResults(dataset=dataset, similarity=similarity, ranked_database_indices=ranking)


def stream_vpr_similarity_csv(
    dataset: ImageDataset,
    algorithm: VPRAlgorithm,
    csv_path: str | Path,
    *,
    progress_every: int = 500,
    query_batch_size: int = 32,
    top_k: int = 5,
) -> StreamedVPRResults:
    """Write query-to-database similarities to CSV without keeping the matrix in memory."""

    if progress_every < 1:
        raise ValueError("progress_every must be at least 1.")
    if query_batch_size < 1:
        raise ValueError("query_batch_size must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    print("Encoding database descriptors...", flush=True)
    database_descriptors = _l2_normalize(algorithm.encode_database(dataset.database_paths))
    print(
        f"Encoded {len(dataset.database_paths)}/{len(dataset.database_paths)} database images.",
        flush=True,
    )

    csv_path = Path(csv_path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ranked_database_indices = np.empty(
        (len(dataset.query_paths), min(top_k, len(dataset.database_paths))), dtype=np.int32
    )

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["query_index", "query_path"]
            + [f"database_{index}" for index in range(len(dataset.database_paths))]
        )
        print("Encoding queries and streaming similarity rows...", flush=True)
        query_count = len(dataset.query_paths)
        query_descriptor_dim: int | None = None
        for start in range(0, query_count, query_batch_size):
            end = min(start + query_batch_size, query_count)
            query_batch = _l2_normalize(algorithm.encode(dataset.query_paths[start:end]))
            if query_descriptor_dim is None:
                query_descriptor_dim = int(query_batch.shape[1])
                if database_descriptors.shape[1] != query_descriptor_dim:
                    raise ValueError("Database and query descriptor sizes do not match.")
            elif query_batch.shape[1] != query_descriptor_dim:
                raise ValueError("Query descriptor sizes are inconsistent across batches.")
            batch_scores = np.asarray(
                algorithm.similarity(query_batch, database_descriptors),
                dtype=np.float32,
            )
            expected_shape = (end - start, len(dataset.database_paths))
            if batch_scores.shape != expected_shape:
                raise ValueError(
                    f"Algorithm similarity returned shape {batch_scores.shape}; "
                    f"expected {expected_shape}."
                )
            for offset, scores in enumerate(batch_scores):
                query_index = start + offset
                if len(scores) <= top_k:
                    top_indices = np.argsort(-scores)
                else:
                    candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
                    top_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
                ranked_database_indices[query_index, : len(top_indices)] = top_indices
                writer.writerow([query_index, dataset.query_paths[query_index].name, *scores.tolist()])
                if (query_index + 1) % progress_every == 0 or query_index + 1 == query_count:
                    print(
                        f"Wrote {query_index + 1}/{query_count} query rows to {csv_path}",
                        flush=True,
                    )

    return StreamedVPRResults(
        dataset=dataset,
        similarity=SimilarityCSVView(csv_path, len(dataset.query_paths), len(dataset.database_paths)),
        ranked_database_indices=ranked_database_indices,
    )
