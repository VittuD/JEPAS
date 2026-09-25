#!/usr/bin/env python3
"""Generate the synthetic probe set (see experiments/SYNTHETIC_PROBE.md).

Layout under --out-dir:

    image/<family>/<family>_<idx>.png
    video/<family>/<family>_<idx>.mkv          (ffv1, lossless, 32 frames @ 16 fps)
    masks/image/<family>/<family>_<idx>.png    (uint8 label maps: regions, shapes)
    manifest.jsonl                             (one record per input file)

Everything is deterministic in (seed, family, idx): each file draws from its own
`default_rng([seed, family_id, idx])`, so files do not depend on generation order or
on the number of workers. Patterns are parametrised in relative units (cells or
regions per image), not pixels, because every model resizes to its own resolution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

VIDEO_FRAMES = 32
VIDEO_FPS = 16

NOISE_BLOCK_SCALES = (2, 4, 8, 16, 32, 64)
STRIPE_SCALES = (2, 4, 8, 16, 32)
CHECKER_SCALES = (2, 4, 8, 16, 32)
REGION_COUNTS = (2, 3, 4, 5)
MOVING_SPEEDS = (0.1, 0.25, 0.5)  # image widths per second
VIDEO_NOISE_CELLS = 16

IMAGE_FAMILIES = (
    "uniform",
    "gradient",
    "noise_pixel",
    "noise_blocks",
    "stripes",
    "checker",
    "regions",
    "shapes",
)
VIDEO_FAMILIES = (
    "static_pattern",
    "moving_shape",
    "noise_temporal",
    "noise_static",
    "flicker",
    "cut",
)
# Fixed ids (not list positions), so adding a family never changes existing files.
FAMILY_ID = {
    "uniform": 1, "gradient": 2, "noise_pixel": 3, "noise_blocks": 4,
    "stripes": 5, "checker": 6, "regions": 7, "shapes": 8,
    "static_pattern": 21, "moving_shape": 22, "noise_temporal": 23,
    "noise_static": 24, "flicker": 25, "cut": 26,
}
FAMILIES_WITH_MASKS = ("regions", "shapes")


def _cycle(values: tuple, idx: int):
    return values[idx % len(values)]


def distinct_colors(rng: np.random.Generator, k: int, min_dist: float = 90.0) -> np.ndarray:
    """k RGB colours (uint8) that are pairwise at least `min_dist` apart (Euclidean)."""
    colors: list[np.ndarray] = []
    while len(colors) < k:
        candidate = rng.integers(0, 256, size=3).astype(np.float64)
        if all(np.linalg.norm(candidate - c) >= min_dist for c in colors):
            colors.append(candidate)
    return np.stack(colors).astype(np.uint8)


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    return (xs + 0.5) / size, (ys + 0.5) / size  # unit-square coordinates


def _cell_index(size: int, cells: int) -> np.ndarray:
    return (np.arange(size) * cells) // size


def block_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    blocks = rng.integers(0, 256, size=(cells, cells, 3), dtype=np.uint8)
    index = _cell_index(size, cells)
    return blocks[index][:, index]


def voronoi_regions(rng: np.random.Generator, size: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat-colour Voronoi image. Returns (image, label mask, colours)."""
    x, y = _grid(size)
    seeds = rng.random(size=(k, 2))
    distance = (x[..., None] - seeds[:, 0]) ** 2 + (y[..., None] - seeds[:, 1]) ** 2
    labels = distance.argmin(axis=-1)
    colors = distinct_colors(rng, k)
    return colors[labels], labels.astype(np.uint8), colors


def shapes_image(rng: np.random.Generator, size: int, count: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Flat circles/rectangles on a flat background. Mask: 0 = background, i+1 = shape i."""
    x, y = _grid(size)
    colors = distinct_colors(rng, count + 1)
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = colors[0]
    mask = np.zeros((size, size), dtype=np.uint8)
    described = []
    for i in range(count):
        kind = "circle" if rng.random() < 0.5 else "rect"
        cx, cy = rng.uniform(0.2, 0.8, size=2)
        half = rng.uniform(0.08, 0.2)
        inside = (
            (x - cx) ** 2 + (y - cy) ** 2 <= half**2
            if kind == "circle"
            else (np.abs(x - cx) <= half) & (np.abs(y - cy) <= half)
        )
        image[inside] = colors[i + 1]
        mask[inside] = i + 1
        described.append({"shape": kind, "cx": round(float(cx), 3), "cy": round(float(cy), 3), "half": round(float(half), 3)})
    return image, mask, described


def _reflect(value: np.ndarray, low: float, high: float) -> np.ndarray:
    """Triangle-wave reflection of `value` into [low, high] (a bouncing coordinate)."""
    span = high - low
    phase = np.mod(value - low, 2 * span)
    return low + np.where(phase > span, 2 * span - phase, phase)


# --- image families ---------------------------------------------------------------------


def make_image(family: str, rng: np.random.Generator, idx: int, size: int) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Return (HxWx3 uint8 image, optional label mask, parameters)."""
    if family == "uniform":
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        image = np.empty((size, size, 3), dtype=np.uint8)
        image[:] = color
        return image, None, {"color": color.tolist()}
    if family == "gradient":
        c0, c1 = distinct_colors(rng, 2).astype(np.float64)
        x, y = _grid(size)
        if idx % 2 == 0:
            angle = float(rng.uniform(0, np.pi))
            t = (x - 0.5) * np.cos(angle) + (y - 0.5) * np.sin(angle)
            t = (t - t.min()) / (t.max() - t.min())
            params = {"mode": "linear", "angle": round(angle, 3)}
        else:
            cx, cy = rng.uniform(0.3, 0.7, size=2)
            t = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            t = t / t.max()
            params = {"mode": "radial"}
        image = (c0 + t[..., None] * (c1 - c0)).round().astype(np.uint8)
        return image, None, params
    if family == "noise_pixel":
        return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8), None, {}
    if family == "noise_blocks":
        cells = int(_cycle(NOISE_BLOCK_SCALES, idx))
        return block_noise(rng, size, cells), None, {"cells": cells}
    if family == "stripes":
        n = int(_cycle(STRIPE_SCALES, idx))
        angle = float(rng.uniform(0, np.pi))
        c0, c1 = distinct_colors(rng, 2)
        x, y = _grid(size)
        u = (x - 0.5) * np.cos(angle) + (y - 0.5) * np.sin(angle)
        # n stripes across the image side: stripe width = 1/n of the side.
        stripe = np.floor((u + 1.0) * n).astype(np.int64) % 2
        image = np.where(stripe[..., None] == 0, c0, c1).astype(np.uint8)
        return image, None, {"stripes": n, "angle": round(angle, 3)}
    if family == "checker":
        n = int(_cycle(CHECKER_SCALES, idx))
        c0, c1 = distinct_colors(rng, 2)
        index = _cell_index(size, n)
        parity = (index[:, None] + index[None, :]) % 2
        return np.where(parity[..., None] == 0, c0, c1).astype(np.uint8), None, {"cells": n}
    if family == "regions":
        k = int(_cycle(REGION_COUNTS, idx))
        image, mask, colors = voronoi_regions(rng, size, k)
        return image, mask, {"k": k, "colors": colors.tolist()}
    if family == "shapes":
        count = 1 + idx % 4
        image, mask, described = shapes_image(rng, size, count)
        return image, mask, {"count": count, "shapes": described}
    raise ValueError(f"Unknown image family {family!r}.")


# --- video families ---------------------------------------------------------------------


def make_video(family: str, rng: np.random.Generator, idx: int, size: int, frames: int = VIDEO_FRAMES) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (T x H x W x 3 uint8 video, parameters)."""
    if family == "static_pattern":
        k = int(_cycle(REGION_COUNTS, idx))
        image, _, colors = voronoi_regions(rng, size, k)
        return np.repeat(image[None], frames, axis=0), {"k": k}
    if family == "moving_shape":
        speed = float(_cycle(MOVING_SPEEDS, idx))
        bg, fg = distinct_colors(rng, 2)
        kind = "circle" if idx % 2 == 0 else "rect"
        half = 0.15
        angle = float(rng.uniform(0, 2 * np.pi))
        start = rng.uniform(0.3, 0.7, size=2)
        x, y = _grid(size)
        video = np.empty((frames, size, size, 3), dtype=np.uint8)
        for f in range(frames):
            t = f / VIDEO_FPS
            cx = _reflect(np.asarray(start[0] + speed * t * np.cos(angle)), half, 1 - half)
            cy = _reflect(np.asarray(start[1] + speed * t * np.sin(angle)), half, 1 - half)
            inside = (
                (x - cx) ** 2 + (y - cy) ** 2 <= half**2
                if kind == "circle"
                else (np.abs(x - cx) <= half) & (np.abs(y - cy) <= half)
            )
            video[f] = bg
            video[f][inside] = fg
        return video, {"speed": speed, "shape": kind, "angle": round(angle, 3)}
    if family == "noise_temporal":
        video = np.stack([block_noise(rng, size, VIDEO_NOISE_CELLS) for _ in range(frames)])
        return video, {"cells": VIDEO_NOISE_CELLS}
    if family == "noise_static":
        return np.repeat(block_noise(rng, size, VIDEO_NOISE_CELLS)[None], frames, axis=0), {"cells": VIDEO_NOISE_CELLS}
    if family == "flicker":
        c0, c1 = distinct_colors(rng, 2).astype(np.float64)
        if idx % 2 == 0:
            weights = np.linspace(0.0, 1.0, frames)
            mode = "ramp"
            colors = c0 + weights[:, None] * (c1 - c0)
        else:
            mode = "random"
            colors = rng.integers(0, 256, size=(frames, 3)).astype(np.float64)
        video = np.empty((frames, size, size, 3), dtype=np.uint8)
        video[:] = colors.round().astype(np.uint8)[:, None, None, :]
        return video, {"mode": mode}
    if family == "cut":
        k = int(_cycle(REGION_COUNTS, idx))
        a, _, _ = voronoi_regions(rng, size, k)
        b, _, _ = voronoi_regions(rng, size, k)
        half = frames // 2
        return np.concatenate([np.repeat(a[None], half, axis=0), np.repeat(b[None], frames - half, axis=0)]), {"k": k, "cut_frame": half}
    raise ValueError(f"Unknown video family {family!r}.")


def write_video(path: Path, video: np.ndarray) -> None:
    """Lossless ffv1 in Matroska; bgr0 keeps RGB values exact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    _, height, width, _ = video.shape
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(VIDEO_FPS), "-i", "-",
            "-c:v", "ffv1", "-pix_fmt", "bgr0", str(tmp),
        ],
        input=np.ascontiguousarray(video).tobytes(),
        check=True,
    )
    tmp.replace(path)


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    Image.fromarray(array).save(tmp, format="PNG")
    tmp.replace(path)


def _rng(seed: int, family: str, idx: int) -> np.random.Generator:
    return np.random.default_rng([seed, FAMILY_ID[family], idx])


def generate_one(task: tuple[str, str, int, int, int, str]) -> dict[str, Any]:
    kind, family, idx, seed, size, out = task
    root = Path(out)
    rng = _rng(seed, family, idx)
    name = f"{family}_{idx:04d}"
    if kind == "image":
        image, mask, params = make_image(family, rng, idx, size)
        rel = f"image/{family}/{name}.png"
        _write_png(root / rel, image)
        if mask is not None:
            _write_png(root / "masks" / "image" / family / f"{name}.png", mask)
    else:
        video, params = make_video(family, rng, idx, size)
        rel = f"video/{family}/{name}.mkv"
        write_video(root / rel, video)
    return {"file": rel, "kind": kind, "family": family, "idx": idx, "size": size, **params}


def build_tasks(out_dir: Path, *, seed: int, per_family: int, size: int, kinds: tuple[str, ...]) -> list[tuple[str, str, int, int, int, str]]:
    plan = {"image": IMAGE_FAMILIES, "video": VIDEO_FAMILIES}
    return [
        (kind, family, idx, seed, size, str(out_dir))
        for kind in kinds
        for family in plan[kind]
        for idx in range(per_family)
    ]


def generate(out_dir: Path, *, seed: int, per_family: int, size: int, kinds: tuple[str, ...], workers: int) -> list[dict[str, Any]]:
    tasks = build_tasks(out_dir, seed=seed, per_family=per_family, size=size, kinds=kinds)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            records = list(pool.map(generate_one, tasks, chunksize=8))
    else:
        records = [generate_one(t) for t in tasks]
    records.sort(key=lambda r: r["file"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-family", type=int, default=128)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--kinds", nargs="+", choices=("image", "video"), default=["image", "video"])
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = generate(
        args.out_dir, seed=args.seed, per_family=args.per_family, size=args.size,
        kinds=tuple(args.kinds), workers=args.workers,
    )
    counts: dict[str, int] = {}
    for record in records:
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1
    print(f"wrote {len(records)} files to {args.out_dir}: {counts}")


if __name__ == "__main__":
    main()
