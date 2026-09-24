"""Read, write, validate, and clean the stable experiment artifact schema."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


SCHEMA_VERSION = 1
SCHEMA_NAME = "jepas-experiment-v1"
EMBEDDINGS_NAME = "embeddings.h5"
FIGURE_NAME = "visualization.png"
FRAME_GLOB = "visualization_frame_*.png"
METADATA_NAME = "metadata.json"


def _chunks_for(array: np.ndarray) -> tuple[int, ...]:
    """Choose bounded chunks while keeping the embedding dimension contiguous."""
    shape = tuple(int(value) for value in array.shape)
    if not shape:
        return ()
    chunks = list(shape)
    itemsize = max(int(array.dtype.itemsize), 1)
    target_items = max((1024 * 1024) // itemsize, 1)
    while int(np.prod(chunks)) > target_items:
        candidates = range(max(len(chunks) - 1, 1))
        axis = max(candidates, key=lambda index: chunks[index])
        chunks[axis] = max((chunks[axis] + 1) // 2, 1)
    return tuple(chunks)


def write_embeddings(
    path: Path,
    tokens: np.ndarray,
    *,
    metadata: dict[str, Any],
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Atomically store tokens losslessly and an FP32 mean-pooled view.

    `sources` is optional per-row provenance (e.g. the input file each row of
    a batched `(N, T, D)` array came from). Its length must equal `tokens`'
    leading dimension. Single-example callers (`N` implicitly 1) can omit it.
    """
    path = path.resolve()
    array = np.asarray(tokens)
    if array.ndim < 2:
        raise ValueError(f"Expected token embeddings, got {array.shape}.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Expected a numeric token dtype, got {array.dtype}.")
    if sources is not None and len(sources) != array.shape[0]:
        raise ValueError(
            f"sources has {len(sources)} entries but tokens has "
            f"{array.shape[0]} rows."
        )
    pooled = array.mean(axis=-2, dtype=np.float32)
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = SCHEMA_NAME
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
            handle.attrs["tokens_shape_json"] = json.dumps(list(array.shape))
            handle.attrs["tokens_dtype"] = array.dtype.str
            handle.attrs["pooled_shape_json"] = json.dumps(list(pooled.shape))
            handle.attrs["pooled_dtype"] = pooled.dtype.str
            handle.create_dataset(
                "tokens",
                data=array,
                chunks=_chunks_for(array),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            handle.create_dataset(
                "pooled",
                data=pooled,
                chunks=_chunks_for(pooled),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            if sources is not None:
                handle.create_dataset(
                    "sources",
                    data=np.asarray(sources, dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            handle.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "token_shape": list(array.shape),
        "token_dtype": array.dtype.str,
        "pooled_shape": list(pooled.shape),
        "pooled_dtype": pooled.dtype.str,
    }


def read_embedding_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        metadata.update(
            {
                "token_shape": list(handle["tokens"].shape),
                "token_dtype": handle["tokens"].dtype.str,
                "pooled_shape": list(handle["pooled"].shape),
                "pooled_dtype": handle["pooled"].dtype.str,
            }
        )
        if "sources" in handle:
            metadata["sources"] = [
                value.decode("utf-8") if isinstance(value, bytes) else value
                for value in handle["sources"][...]
            ]
        return metadata


def valid_embeddings(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema") != SCHEMA_NAME:
                return False
            tokens = handle["tokens"]
            pooled = handle["pooled"]
            if tokens.ndim < 2 or pooled.dtype != np.dtype(np.float32):
                return False
            if pooled.shape != tokens.shape[:-2] + tokens.shape[-1:]:
                return False
            if "sources" in handle and handle["sources"].shape[0] != tokens.shape[0]:
                return False
            json.loads(str(handle.attrs["metadata_json"]))
            return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def load_visualization_metadata(directory: Path) -> dict[str, Any] | None:
    path = directory / METADATA_NAME
    try:
        metadata = json.loads(path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def temporal_frame_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob(FRAME_GLOB))


def valid_visualization(
    directory: Path, *, require_temporal: bool | None = None
) -> bool:
    metadata = load_visualization_metadata(directory)
    if metadata is None or metadata.get("schema") != SCHEMA_NAME:
        return False
    extraction = metadata.get("extraction")
    if not isinstance(extraction, dict) or not all(
        key in extraction
        for key in ("token_shape", "token_dtype", "pooled_shape", "pooled_dtype")
    ):
        return False
    if not isinstance(metadata.get("grid_shape"), list):
        return False
    if not isinstance(metadata.get("pca_explained_variance"), list):
        return False
    if metadata.get("figure") != FIGURE_NAME:
        return False
    figure = directory / FIGURE_NAME
    try:
        with Image.open(figure) as image:
            image.verify()
    except (OSError, ValueError):
        return False
    expected = (
        bool(metadata.get("temporal_expected"))
        if require_temporal is None
        else require_temporal
    )
    frame_names = metadata.get("frames")
    if expected:
        if not isinstance(frame_names, list) or not frame_names:
            return False
        expected_names = [f"visualization_frame_{index:03d}.png" for index in range(len(frame_names))]
        if frame_names != expected_names:
            return False
        try:
            for name in frame_names:
                with Image.open(directory / name) as image:
                    image.verify()
        except (OSError, ValueError):
            return False
    elif frame_names not in (None, []):
        return False
    return True


def cleanup_embeddings(
    directory: Path, *, require_temporal: bool | None = None
) -> bool:
    """Delete embeddings only after all declared visualization outputs validate."""
    embeddings = directory / EMBEDDINGS_NAME
    if not embeddings.exists():
        return False
    if not valid_visualization(directory, require_temporal=require_temporal):
        raise ValueError(
            f"Refusing to delete {embeddings}: visualization artifacts or metadata are invalid."
        )
    embeddings.unlink()
    return True
