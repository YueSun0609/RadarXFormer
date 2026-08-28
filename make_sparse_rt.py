#!/usr/bin/env python3
"""Precompute Range-MAD sparse REA tensors for faster data loading.

The generated directory mirrors ``SRC_ROOT``::

    kradar-mvd-all/train/1/frame/rea.npy
    kradar-mvd-sparse/train/1/frame/rea_sparse.npz

The NPZ contains only the arrays consumed by ``KRadarDataset``:

    coords          int32   [N, 3]  (range, elevation, azimuth)
    features        float32 [N, 3]  (mean, variance, peak Doppler velocity)
    spatial_shape   int64   [3]

Edit the configuration block below, then run ``python make_sparse_rt.py``.
Files are uncompressed by default because this script targets training and
inference throughput rather than minimum disk usage.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ========================== User configuration ===========================
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "kradar-mvd-all"
DST_ROOT = PROJECT_ROOT / "kradar-mvd-sparse"

SPLIT_LIST = ["train", "test"]
# None processes every sequence found under each split. Example: [1, 2, 3].
SEQ_LIST: Optional[Sequence[int]] = None

SOURCE_FILENAME = "rea.npy"
OUTPUT_FILENAME = "rea_sparse.npz"

# These values must match the model configuration.
MAD_THRESHOLD = 1.8
MIN_POINTS = 8000
MAX_POINTS = 64000
MAD_SCALE = 1.4826
MAD_EPSILON = 1e-6

# HDDs often prefer 2-4 workers; SSD/NVMe can usually use 4-8.
WORKERS = 4
LIMIT_PER_SEQUENCE: Optional[int] = None
OVERWRITE = False
VERIFY_OUTPUT = False
COMPRESSED = False
DRY_RUN = False
# ========================================================================

EXPECTED_SHAPE = (3, 256, 37, 107)  # C, R, E, A
SPATIAL_SHAPE = EXPECTED_SHAPE[1:]


@dataclass(frozen=True)
class JobResult:
    source: str
    destination: str
    status: str
    points: int
    threshold_points: int
    mode: str
    output_bytes: int
    elapsed_seconds: float


def validate_config() -> None:
    if not SRC_ROOT.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {SRC_ROOT}")
    if not SPLIT_LIST:
        raise ValueError("SPLIT_LIST cannot be empty")
    if MAD_THRESHOLD < 0:
        raise ValueError("MAD_THRESHOLD must be non-negative")
    if MAD_SCALE <= 0 or MAD_EPSILON <= 0:
        raise ValueError("MAD_SCALE and MAD_EPSILON must be positive")
    total_cells = int(np.prod(SPATIAL_SHAPE))
    if not 0 <= MIN_POINTS <= MAX_POINTS <= total_cells:
        raise ValueError(
            "Point counts must satisfy "
            f"0 <= MIN_POINTS <= MAX_POINTS <= {total_cells}"
        )
    if WORKERS < 1:
        raise ValueError("WORKERS must be at least 1")
    if LIMIT_PER_SEQUENCE is not None and LIMIT_PER_SEQUENCE < 1:
        raise ValueError("LIMIT_PER_SEQUENCE must be positive or None")


def normalized_sequences(split_root: Path) -> List[str]:
    available = sorted(
        (path.name for path in split_root.iterdir() if path.is_dir()),
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    if SEQ_LIST is None:
        return available

    requested = []
    seen = set()
    for value in SEQ_LIST:
        sequence = str(int(value))
        if sequence not in seen:
            requested.append(sequence)
            seen.add(sequence)
    missing = [value for value in requested if value not in available]
    if missing:
        raise FileNotFoundError(
            f"Sequences missing below {split_root}: {', '.join(missing)}"
        )
    return requested


def discover_jobs() -> Tuple[List[Tuple[Path, Path]], List[str]]:
    jobs: List[Tuple[Path, Path]] = []
    group_descriptions: List[str] = []

    for split in SPLIT_LIST:
        split_root = SRC_ROOT / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {split_root}")

        for sequence in normalized_sequences(split_root):
            sequence_root = split_root / sequence
            sample_dirs = sorted(
                path for path in sequence_root.iterdir() if path.is_dir()
            )
            if LIMIT_PER_SEQUENCE is not None:
                sample_dirs = sample_dirs[:LIMIT_PER_SEQUENCE]

            group_jobs = 0
            for sample_dir in sample_dirs:
                source = sample_dir / SOURCE_FILENAME
                if not source.is_file():
                    continue
                relative_sample = sample_dir.relative_to(SRC_ROOT)
                destination = DST_ROOT / relative_sample / OUTPUT_FILENAME
                jobs.append((source, destination))
                group_jobs += 1
            group_descriptions.append(f"{split}/{sequence}={group_jobs}")

    if not jobs:
        raise FileNotFoundError(
            f"No {SOURCE_FILENAME} files found under configured splits/sequences"
        )
    return jobs, group_descriptions


def largest_indices(values: np.ndarray, count: int) -> np.ndarray:
    """Return indices of the largest values without fully sorting them."""
    if count <= 0:
        return np.empty((0,), dtype=np.intp)
    if count >= values.size:
        return np.arange(values.size, dtype=np.intp)
    return np.argpartition(values, values.size - count)[-count:]


def sparsify_rea(
    rea: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """Apply exactly the same Range-MAD selection as KRadarDataset."""
    if rea.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected {EXPECTED_SHAPE}, got {rea.shape}")

    rea = np.asarray(rea, dtype=np.float32)
    power = rea[0]
    median = np.median(power, axis=(1, 2), keepdims=True)
    mad = np.median(np.abs(power - median), axis=(1, 2), keepdims=True)
    scale = np.maximum(MAD_SCALE * mad, MAD_EPSILON)
    scores = ((power - median) / scale).reshape(-1)

    selected = np.flatnonzero(scores > MAD_THRESHOLD)
    threshold_points = int(selected.size)
    mode = "threshold"
    if selected.size < MIN_POINTS:
        selected = largest_indices(scores, MIN_POINTS)
        mode = "minimum-floor"
    elif selected.size > MAX_POINTS:
        selected = selected[largest_indices(scores[selected], MAX_POINTS)]
        mode = "maximum-cap"
    selected = np.sort(selected)

    _, elevation_dim, azimuth_dim = SPATIAL_SHAPE
    elevation_azimuth_size = elevation_dim * azimuth_dim
    range_indices = selected // elevation_azimuth_size
    remainder = selected % elevation_azimuth_size
    elevation_indices = remainder // azimuth_dim
    azimuth_indices = remainder % azimuth_dim

    coordinates = np.stack(
        (range_indices, elevation_indices, azimuth_indices), axis=1
    ).astype(np.int32, copy=False)
    features = np.ascontiguousarray(
        rea[:, range_indices, elevation_indices, azimuth_indices].T,
        dtype=np.float32,
    )
    return coordinates, features, threshold_points, mode


def save_sparse(
    destination: Path,
    coordinates: np.ndarray,
    features: np.ndarray,
) -> None:
    """Atomically save arrays in the exact dtypes expected by the dataset."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    save_function = np.savez_compressed if COMPRESSED else np.savez
    try:
        with temporary.open("wb") as output_file:
            save_function(
                output_file,
                coords=coordinates,
                features=features,
                spatial_shape=np.asarray(SPATIAL_SHAPE, dtype=np.int64),
            )
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_sparse(path: Path) -> int:
    with np.load(path, allow_pickle=False) as sparse:
        if set(sparse.files) != {"coords", "features", "spatial_shape"}:
            raise ValueError(f"Unexpected NPZ keys in {path}: {sparse.files}")
        coordinates = sparse["coords"]
        features = sparse["features"]
        spatial_shape = sparse["spatial_shape"]

    if coordinates.dtype != np.int32 or coordinates.ndim != 2:
        raise ValueError(f"Invalid coordinates in {path}")
    if coordinates.shape[1] != 3:
        raise ValueError(f"Invalid coordinate shape in {path}: {coordinates.shape}")
    if features.dtype != np.float32 or features.shape != (len(coordinates), 3):
        raise ValueError(f"Invalid features in {path}: {features.shape}")
    if spatial_shape.dtype != np.int64 or tuple(spatial_shape) != SPATIAL_SHAPE:
        raise ValueError(f"Invalid spatial shape in {path}: {spatial_shape}")
    if not MIN_POINTS <= len(coordinates) <= MAX_POINTS:
        raise ValueError(f"Invalid point count in {path}: {len(coordinates)}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features in {path}")
    if np.any(coordinates < 0):
        raise ValueError(f"Negative coordinate in {path}")
    for axis, bound in enumerate(SPATIAL_SHAPE):
        if np.any(coordinates[:, axis] >= bound):
            raise ValueError(f"Coordinate axis {axis} out of bounds in {path}")
    return len(coordinates)


def process_one(source_string: str, destination_string: str) -> JobResult:
    source = Path(source_string)
    destination = Path(destination_string)
    start = time.perf_counter()

    if destination.exists() and not OVERWRITE:
        if VERIFY_OUTPUT:
            points = verify_sparse(destination)
        else:
            with np.load(destination, allow_pickle=False) as sparse:
                points = int(sparse["coords"].shape[0])
        return JobResult(
            str(source), str(destination), "skipped", points, -1, "existing",
            destination.stat().st_size, time.perf_counter() - start,
        )

    rea = np.load(source, mmap_mode="r", allow_pickle=False)
    coordinates, features, threshold_points, mode = sparsify_rea(rea)
    save_sparse(destination, coordinates, features)
    if VERIFY_OUTPUT:
        verify_sparse(destination)

    return JobResult(
        str(source), str(destination), "written", len(coordinates),
        threshold_points, mode, destination.stat().st_size,
        time.perf_counter() - start,
    )


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def sample_name(path_string: str) -> str:
    path = Path(path_string)
    return "/".join(path.parts[-4:-1])


def execute_jobs(jobs: Iterable[Tuple[Path, Path]]) -> Tuple[List[JobResult], list]:
    serialized = [(str(source), str(destination)) for source, destination in jobs]
    results: List[JobResult] = []
    failures = []

    def report(result: JobResult) -> None:
        completed = len(results) + len(failures)
        print(
            f"[{completed}/{len(serialized)}] {sample_name(result.source)} "
            f"{result.status}: points={result.points}, mode={result.mode}, "
            f"size={human_size(result.output_bytes)}, "
            f"time={result.elapsed_seconds:.3f}s",
            flush=True,
        )

    if WORKERS == 1:
        for source, destination in serialized:
            try:
                result = process_one(source, destination)
                results.append(result)
                report(result)
            except Exception as error:
                failures.append((source, repr(error)))
                print(f"FAILED {sample_name(source)}: {error}", file=sys.stderr)
        return results, failures

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(process_one, source, destination): source
            for source, destination in serialized
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                result = future.result()
                results.append(result)
                report(result)
            except Exception as error:
                failures.append((source, repr(error)))
                print(f"FAILED {sample_name(source)}: {error}", file=sys.stderr)
    return results, failures


def main() -> int:
    try:
        validate_config()
        jobs, groups = discover_jobs()
    except (FileNotFoundError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    print(f"Source root:       {SRC_ROOT}")
    print(f"Destination root:  {DST_ROOT}")
    print(f"Groups:            {', '.join(groups)}")
    print(f"Total frames:      {len(jobs)}")
    print(f"Range-MAD:         threshold={MAD_THRESHOLD}, scale={MAD_SCALE}")
    print(f"Point limits:      [{MIN_POINTS}, {MAX_POINTS}]")
    print(f"Workers:           {WORKERS}")
    print(f"Compressed:        {COMPRESSED}")
    if DRY_RUN:
        print("Dry run complete; nothing was written.")
        return 0

    start = time.perf_counter()
    results, failures = execute_jobs(jobs)
    elapsed = time.perf_counter() - start
    written = sum(result.status == "written" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    total_bytes = sum(result.output_bytes for result in results)

    print("\nSummary")
    print(f"Successful:        {len(results)}")
    print(f"Written/skipped:   {written}/{skipped}")
    print(f"Failed:            {len(failures)}")
    print(f"Output size:       {human_size(total_bytes)}")
    print(f"Elapsed:           {elapsed:.2f}s")
    if results:
        print(f"Average:           {elapsed / len(results):.3f}s/frame")
    if failures:
        print("Failed sources:", file=sys.stderr)
        for source, error in failures:
            print(f"  {source}: {error}", file=sys.stderr)
        return 1
    print("\nDataset configuration after generation:")
    print('  "radar_representation": "sparse",')
    print(f'  "radar_sparse_root": "{DST_ROOT}",')
    print(f'  "radar_sparse_filename": "{OUTPUT_FILENAME}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
