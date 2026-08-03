#!/usr/bin/env python3
"""Serve the permanent JEPA experiment browser and new-schema outputs."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit


ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = Path(__file__).with_name("index.html")
DEFAULT_OUTPUTS = ROOT_DIR / "experiments" / "outputs"
DEFAULT_PORT = 43871
SCHEMA_NAME = "jepas-experiment-v1"


def discover_experiments(outputs: Path) -> list[dict[str, Any]]:
    """Build browser data strictly from new-schema directories on disk."""
    outputs = outputs.expanduser().resolve()
    experiments: list[dict[str, Any]] = []
    if not outputs.is_dir():
        return experiments
    for experiment_dir in sorted(path for path in outputs.iterdir() if path.is_dir()):
        try:
            manifest = json.loads((experiment_dir / "manifest.json").read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if manifest.get("schema") != SCHEMA_NAME:
            continue
        cells = []
        for input_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            for model_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
                image = model_dir / "visualization.png"
                metadata_path = model_dir / "metadata.json"
                try:
                    metadata = json.loads(metadata_path.read_text())
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if not image.is_file() or metadata.get("schema") != SCHEMA_NAME:
                    continue
                prefix = "/outputs/" + "/".join(
                    quote(part, safe="")
                    for part in (experiment_dir.name, input_dir.name, model_dir.name)
                )
                video = model_dir / "visualization.mp4"
                cells.append(
                    {
                        "input": input_dir.name,
                        "model": model_dir.name,
                        "image": f"{prefix}/visualization.png",
                        "video": f"{prefix}/visualization.mp4" if video.is_file() else None,
                        "metadata": metadata,
                    }
                )
        if not cells:
            continue
        experiments.append(
            {
                "id": experiment_dir.name,
                "name": manifest.get("experiment", experiment_dir.name),
                "rows": sorted({cell["input"] for cell in cells}),
                "columns": sorted({cell["model"] for cell in cells}),
                "cells": cells,
            }
        )
    return experiments


def create_handler(outputs: Path) -> type[BaseHTTPRequestHandler]:
    output_root = outputs.expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send_file(INDEX_PATH, "text/html; charset=utf-8")
                return
            if path == "/api/experiments":
                body = json.dumps(discover_experiments(output_root)).encode()
                self._send_bytes(body, "application/json; charset=utf-8")
                return
            if path.startswith("/outputs/"):
                relative = Path(unquote(path.removeprefix("/outputs/")))
                candidate = (output_root / relative).resolve()
                try:
                    candidate.relative_to(output_root)
                except ValueError:
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                if candidate.is_file():
                    media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                    self._send_file(candidate, media_type)
                    return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _send_file(self, path: Path, content_type: str) -> None:
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(body, content_type)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def create_server(outputs: Path, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), create_handler(outputs))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in 0..65535")
    return args


def main() -> None:
    args = parse_args()
    server = create_server(args.outputs, args.port)
    host, port = server.server_address
    print(f"Serving JEPA experiments from {args.outputs.resolve()}")
    print(f"Open http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
