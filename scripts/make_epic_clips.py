#!/usr/bin/env python3
"""Cut fixed-length clips from EPIC-Kitchens recordings for the bulk experiment.

The bulk pipeline samples 16 frames uniformly over a whole video, which is
meaningless for EPIC's minutes-to-half-hour recordings. This picks `--n`
non-overlapping `--clip-seconds` windows uniformly (seeded) over all windows of
all readable recordings, and re-encodes each to a short-side-384 H.264 clip.
Unreadable recordings (e.g. truncated files) are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def duration(path: Path) -> float | None:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def cut(src: Path, start: float, seconds: float, dst: Path) -> bool:
    if dst.exists():
        return True
    tmp = dst.with_suffix(".tmp.mp4")
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
            "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{seconds:g}", "-an",
            "-vf", "scale=-2:384", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            str(tmp),
        ]
    )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="epic-kitchens-100 directory (P*/videos/*.MP4).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip-seconds", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    videos = sorted(args.root.glob("P*/videos/*.MP4"))
    with ThreadPoolExecutor(args.workers) as pool:
        durations = list(pool.map(duration, videos))
    usable = [(v, d) for v, d in zip(videos, durations) if d is not None and d >= args.clip_seconds]
    skipped = len(videos) - len(usable)
    slots = [(i, s) for i, (_, d) in enumerate(usable) for s in range(int(d // args.clip_seconds))]
    print(f"{len(usable)}/{len(videos)} usable recordings ({skipped} skipped), {len(slots)} windows")
    if len(slots) < args.n:
        raise ValueError(f"Only {len(slots)} windows available; need {args.n}.")

    rng = np.random.default_rng(args.seed)
    chosen = sorted(int(k) for k in rng.choice(len(slots), size=args.n, replace=False))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for k in chosen:
        i, s = slots[k]
        src = usable[i][0]
        jobs.append((src, s * args.clip_seconds, args.clip_seconds, args.out_dir / f"{src.stem}_s{s:04d}.mp4"))
    with ThreadPoolExecutor(args.workers) as pool:
        ok = list(pool.map(lambda j: cut(*j), jobs))
    failed = ok.count(False)
    print(f"wrote {ok.count(True)}/{len(jobs)} clips to {args.out_dir}")
    if failed:
        raise SystemExit(f"{failed} clips failed")


if __name__ == "__main__":
    main()
