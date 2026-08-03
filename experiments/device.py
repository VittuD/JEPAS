"""Shared inference-device and mixed-precision policy."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


DEVICE_CHOICES = ("auto", "cpu", "cuda")
PRECISION_CHOICES = ("auto", "fp32", "bf16", "fp16")


@dataclass(frozen=True)
class InferenceRuntime:
    device: torch.device
    precision: str
    autocast_dtype: torch.dtype | None


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Inference device. 'auto' selects CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--precision",
        choices=PRECISION_CHOICES,
        default="auto",
        help="Inference precision. CUDA auto uses BF16 when supported, otherwise FP16.",
    )


def resolve_runtime(device: str, precision: str) -> InferenceRuntime:
    if device not in DEVICE_CHOICES:
        raise ValueError(f"Unsupported device {device!r}.")
    if precision not in PRECISION_CHOICES:
        raise ValueError(f"Unsupported precision {precision!r}.")

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Use the CUDA environment blueprint on a host with a compatible NVIDIA driver."
        )

    if resolved_device == "cpu":
        if precision not in ("auto", "fp32"):
            raise ValueError("CPU inference currently supports --precision fp32 only.")
        return InferenceRuntime(torch.device("cpu"), "fp32", None)

    if precision == "auto":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    autocast_dtype = {
        "fp32": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[precision]
    torch.set_float32_matmul_precision("high")
    return InferenceRuntime(torch.device("cuda"), precision, autocast_dtype)


def seed_inference(seed: int, runtime: InferenceRuntime) -> None:
    torch.manual_seed(seed)
    if runtime.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@contextmanager
def inference_context(runtime: InferenceRuntime) -> Iterator[None]:
    with torch.inference_mode():
        if runtime.autocast_dtype is None:
            yield
        else:
            with torch.autocast(
                device_type=runtime.device.type,
                dtype=runtime.autocast_dtype,
            ):
                yield
