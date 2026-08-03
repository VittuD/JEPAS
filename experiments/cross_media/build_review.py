#!/usr/bin/env python3
"""Build an offline qualitative browser for a cross-media experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).with_name("review") / "static"
sys.path.insert(0, str(ROOT_DIR))

from experiments.common_resolution.render_comparison import load_maps  # noqa: E402
from experiments.shared.provenance import (  # noqa: E402
    portable_path,
    resolve_portable_path,
)


VIEW_NAMES = ("pca_rgb", "pc1", "kmeans")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Cross-media manifest.json.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Browser destination. Defaults to <manifest-directory>/review.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate review tiles even when their source artifacts are unchanged.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_url(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def load_summary(record: dict[str, Any]) -> dict[str, Any]:
    summary_path = resolve_portable_path(
        str(record["artifacts"]["visualization_summary"])
    )
    rows = json.loads(summary_path.read_text())
    base = rows[0] if rows else {}
    kmeans = next(
        (
            row
            for row in rows
            if row.get("method") == "kmeans" and row.get("params") == "k=4"
        ),
        {},
    )
    return {
        "adjacent_cosine_similarity": base.get("adjacent_cosine_similarity"),
        "pca_explained_variance": json.loads(base.get("pca_explained_variance", "[]")),
        "boundary_fraction": kmeans.get("boundary_fraction"),
        "silhouette": kmeans.get("silhouette"),
    }


def as_rgb_image(array: np.ndarray, view: str) -> Image.Image:
    from matplotlib import colormaps

    if view == "pca_rgb":
        rgb = np.clip(array, 0.0, 1.0)
    elif view == "pc1":
        rgb = colormaps["magma"](np.clip(array, 0.0, 1.0))[..., :3]
    elif view == "kmeans":
        rgb = colormaps["tab10"](array.astype(int) % 10)[..., :3]
    else:
        raise ValueError(f"Unsupported review view: {view}")

    image = Image.fromarray(np.asarray(rgb * 255, dtype=np.uint8), mode="RGB")
    return image.resize((384, 384), Image.Resampling.NEAREST)


def render_tiles(
    record: dict[str, Any],
    *,
    out_dir: Path,
    force: bool,
) -> dict[str, Path]:
    tile_dir = out_dir / "tiles" / str(record["input_name"]) / str(record["model"])
    tile_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: tile_dir / f"{name}.png" for name in VIEW_NAMES}

    embedding = resolve_portable_path(str(record["embedding"]))
    newest_source = max(
        embedding.stat().st_mtime_ns,
        resolve_portable_path(
            str(record["artifacts"]["visualization_summary"])
        ).stat().st_mtime_ns,
    )
    if not force and all(
        path.exists() and path.stat().st_mtime_ns >= newest_source for path in outputs.values()
    ):
        return outputs

    _preview, pca_rgb, pc1, clusters = load_maps(record)
    maps = {
        "pca_rgb": pca_rgb,
        "pc1": pc1,
        "kmeans": clusters,
    }
    for name, array in maps.items():
        as_rgb_image(array, name).save(outputs[name], optimize=True)
    return outputs


def artifact_links(record: dict[str, Any], out_dir: Path) -> dict[str, str]:
    result_dir = resolve_portable_path(str(record["figure"])).parents[1]
    artifacts = record["artifacts"]
    paths = {
        "full_figure": resolve_portable_path(str(record["figure"])),
        "animation": result_dir / "visualization" / "embedding_visualization.mp4",
        "result": result_dir / "result.json",
        "summary": resolve_portable_path(str(artifacts["visualization_summary"])),
        "visualization_manifest": resolve_portable_path(
            str(artifacts["visualization_manifest"])
        ),
        "embedding": resolve_portable_path(str(record["embedding"])),
        "extract_log": result_dir / "extract.log",
        "visualize_log": result_dir / "visualize.log",
    }
    return {
        name: relative_url(path, out_dir)
        for name, path in paths.items()
        if path.exists()
    }


def build_record(
    record: dict[str, Any],
    *,
    out_dir: Path,
    force: bool,
) -> dict[str, Any]:
    tiles = render_tiles(record, out_dir=out_dir, force=force)
    preview = resolve_portable_path(str(record["artifacts"]["preview"]["path"]))
    model_preview = resolve_portable_path(
        str(record["artifacts"]["model_preprocessed_preview"]["path"])
    )
    links = artifact_links(record, out_dir)
    return {
        "name": record["name"],
        "model": record["model"],
        "modelModality": record["model_modality"],
        "input": record["input_name"],
        "inputModality": record["input_modality"],
        "adaptation": record["adaptation"],
        "nativeResolution": record["native_resolution"],
        "inputResolution": record["input_resolution"],
        "patchSize": record["patch_size"],
        "spatialGrid": record["spatial_grid"],
        "temporalGrid": record["temporal_grid"],
        "embeddingShape": record["embedding_shape"],
        "positionalGeometry": record.get("positional_geometry"),
        "inputSource": record.get("input_source"),
        "preview": relative_url(preview, out_dir),
        "modelPreview": relative_url(model_preview, out_dir),
        "views": {
            name: relative_url(path, out_dir)
            for name, path in tiles.items()
        }
        | {"full": links["full_figure"]},
        "links": links,
        "metrics": load_summary(record),
        "previewSha256": record["artifacts"]["preview"]["sha256"],
        "figureSha256": record["artifacts"]["figure"]["sha256"],
    }


def copy_static_files(out_dir: Path) -> None:
    for name in ("index.html", "review.css", "review.js"):
        shutil.copy2(STATIC_DIR / name, out_dir / name)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    out_dir = (args.out_dir or manifest_path.parent / "review").resolve()
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("results", [])
    if not records:
        raise ValueError(f"{manifest_path} contains no result records.")

    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
    copy_static_files(out_dir)
    review_records = [
        build_record(record, out_dir=out_dir, force=bool(args.force))
        for record in records
    ]

    generated_at = datetime.now(timezone.utc).isoformat()
    native_resolution = manifest.get("resolution_mode") == "native"
    data = {
        "title": (
            "Cross-Media Native-Resolution Review"
            if native_resolution
            else f"Cross-Media {manifest['resolution']} Review"
        ),
        "resolution": manifest["resolution"],
        "resolutionMode": manifest.get("resolution_mode", "fixed"),
        "modelResolutions": manifest.get("model_resolutions", {}),
        "inputs": manifest["inputs"],
        "models": manifest["models"],
        "records": review_records,
        "manifestHash": sha256(manifest_path),
        "generatedAt": generated_at,
    }
    (out_dir / "review-data.js").write_text(
        "window.REVIEW_DATA = "
        + json.dumps(data, indent=2, sort_keys=False)
        + ";\n"
    )

    build = {
        "schema_version": 1,
        "generated_at": generated_at,
        "source_manifest": portable_path(manifest_path),
        "source_manifest_sha256": data["manifestHash"],
        "repository_revision": git_revision(),
        "generator": portable_path(Path(__file__)),
        "result_count": len(review_records),
        "models": manifest["models"],
        "inputs": manifest["inputs"],
        "view_assets": list(VIEW_NAMES),
    }
    (out_dir / "build.json").write_text(json.dumps(build, indent=2) + "\n")
    print(f"[cross_media] review browser: {out_dir / 'index.html'}")
    print(f"[cross_media] records: {len(review_records)}")


if __name__ == "__main__":
    main()
