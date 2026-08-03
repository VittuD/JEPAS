"""Artifact and command provenance helpers for generated experiments."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def portable_path(path: Path) -> str:
    """Return a repo-relative path when possible, otherwise an absolute path."""
    expanded = path.expanduser()
    resolved = Path(os.path.abspath(expanded))
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(resolved)


def resolve_portable_path(path: str | Path) -> Path:
    """Resolve a serialized path, anchoring relative values to the repo root."""
    candidate = Path(path).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (ROOT_DIR / candidate).resolve()
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(
    path: Path,
    *,
    include_hash: bool,
    previous: dict[str, object] | None = None,
) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": portable_path(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if (
        previous
        and previous.get("size_bytes") == stat.st_size
        and previous.get("mtime_ns") == stat.st_mtime_ns
        and previous.get("sha256")
    ):
        result["sha256"] = previous["sha256"]
    elif include_hash:
        result["sha256"] = sha256_file(path)
    return result


def model_code_provenance(path: Path) -> dict[str, object]:
    if path.is_file():
        return {
            **file_provenance(path, include_hash=True),
            "kind": "local Hugging Face model code",
        }
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "-C", str(path), "config", "--get", "remote.origin.url"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "path": portable_path(path),
        "kind": "git source repository",
        "revision": revision,
        "remote": remote or None,
    }


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path | None = None,
) -> None:
    print("+", shlex.join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=cwd, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
