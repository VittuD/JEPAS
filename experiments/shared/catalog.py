"""Authoritative local model and media catalog for JEPA experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT_DIR / "experiments" / "inputs" / "web_examples"
IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST_PATH = (
    ROOT_DIR / "manifests" / "ijepa_lite_affinity_novelty.json"
)
IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST = json.loads(
    IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST_PATH.read_text()
)
_IJEPA_LITE_CHECKPOINT_OVERRIDE = os.environ.get(
    "IJEPA_LITE_AFFINITY_NOVELTY_CHECKPOINT"
)
_IJEPA_LITE_CHECKPOINT_PATH = Path(
    _IJEPA_LITE_CHECKPOINT_OVERRIDE
    or IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST["checkpoint"]["default_path"]
).expanduser()
IJEPA_LITE_AFFINITY_NOVELTY_CHECKPOINT = (
    _IJEPA_LITE_CHECKPOINT_PATH
    if _IJEPA_LITE_CHECKPOINT_PATH.is_absolute()
    else ROOT_DIR / _IJEPA_LITE_CHECKPOINT_PATH
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    modality: str
    native_resolution: int
    patch_size: int
    extractor: tuple[str, ...]
    checkpoint_path: Path
    checkpoint_source: str
    model_code_path: Path
    positional_geometry: str
    enabled_by_default: bool = True
    configuration_path: Path | None = None


@dataclass(frozen=True)
class MediaSpec:
    name: str
    modality: str
    path: Path
    source: str


def model_specs() -> dict[str, ModelSpec]:
    return {
        "ijepa": ModelSpec(
            name="ijepa",
            modality="image",
            native_resolution=224,
            patch_size=14,
            extractor=("ijepa_random_image_smoke.py",),
            checkpoint_path=ROOT_DIR / "weights" / "ijepa" / "IN1K-vit.h.14-300e.pth.tar",
            checkpoint_source="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar",
            model_code_path=ROOT_DIR / "repos" / "ijepa",
            positional_geometry="bicubic checkpoint position-grid interpolation",
        ),
        "radjepa": ModelSpec(
            name="radjepa",
            modality="image",
            native_resolution=224,
            patch_size=14,
            extractor=("radjepa_random_image_smoke.py",),
            checkpoint_path=ROOT_DIR / "weights" / "radjepa" / "hf-RadJEPA" / "jepa_encoder.pth.tar",
            checkpoint_source="https://huggingface.co/AIDELab-IITBombay/RadJEPA",
            model_code_path=ROOT_DIR / "weights" / "radjepa" / "hf-RadJEPA" / "modeling_radjepa.py",
            positional_geometry="bicubic checkpoint position-grid interpolation",
        ),
        "vjepa2": ModelSpec(
            name="vjepa2",
            modality="video",
            native_resolution=256,
            patch_size=16,
            extractor=("vjepa2_random_image_smoke.py", "--variant", "vjepa2-vitl"),
            checkpoint_path=ROOT_DIR / "weights" / "vjepa2" / "vitl.pt",
            checkpoint_source="https://dl.fbaipublicfiles.com/vjepa2/vitl.pt",
            model_code_path=ROOT_DIR / "repos" / "vjepa2",
            positional_geometry="source-model RoPE at the runtime grid",
        ),
        "vjepa2-1": ModelSpec(
            name="vjepa2-1",
            modality="video",
            native_resolution=384,
            patch_size=16,
            extractor=("vjepa2_random_image_smoke.py", "--variant", "vjepa2-1-vitb"),
            checkpoint_path=ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitb_dist_vitG_384.pt",
            checkpoint_source=(
                "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
            ),
            model_code_path=ROOT_DIR / "repos" / "vjepa2",
            positional_geometry="native 384 source-model RoPE",
        ),
        "echojepa": ModelSpec(
            name="echojepa",
            modality="video",
            native_resolution=224,
            patch_size=16,
            extractor=("echojepa_vjepa2_random_image_smoke.py",),
            checkpoint_path=ROOT_DIR / "weights" / "echojepa" / "vitl-vmix22m-pt220-c55.pt",
            checkpoint_source=(
                "https://drive.google.com/uc?id=1T_ubAMpDMEByH7V6TJu9iT5Yf9IxlDBB"
            ),
            model_code_path=ROOT_DIR / "repos" / "EchoJEPA",
            positional_geometry="source-model RoPE at the runtime grid",
        ),
        "ijepa-lite-affinity-novelty": ModelSpec(
            name="ijepa-lite-affinity-novelty",
            modality="image",
            native_resolution=int(
                IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST["model"]["image_size"]
            ),
            patch_size=int(
                IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST["model"]["patch_size"]
            ),
            extractor=(
                "ijepa_lite_affinity_novelty_smoke.py",
                "--encoder",
                str(IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST["checkpoint"]["encoder"]),
            ),
            checkpoint_path=IJEPA_LITE_AFFINITY_NOVELTY_CHECKPOINT,
            checkpoint_source=str(
                IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST["checkpoint"]["source_uri"]
            ),
            model_code_path=ROOT_DIR / "repos" / "ijepa_lite",
            positional_geometry=(
                "learned patch-only 16x16 grid; bicubic interpolation at runtime"
            ),
            enabled_by_default=False,
            configuration_path=IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST_PATH,
        ),
    }


def default_model_names() -> tuple[str, ...]:
    return tuple(
        name
        for name, spec in model_specs().items()
        if spec.enabled_by_default
    )


def media_specs() -> dict[str, MediaSpec]:
    return {
        "dog": MediaSpec(
            name="dog",
            modality="image",
            path=INPUT_DIR / "dog_image.jpg",
            source="https://commons.wikimedia.org/wiki/File:Dog_image.jpg",
        ),
        "chest_xray": MediaSpec(
            name="chest_xray",
            modality="image",
            path=INPUT_DIR / "chest_xray_pneumonia.jpg",
            source="https://commons.wikimedia.org/wiki/File:Pneumonia_x_ray.jpg",
        ),
        "diving48": MediaSpec(
            name="diving48",
            modality="video",
            path=INPUT_DIR / "diving48_sample.mp4",
            source=(
                "https://huggingface.co/datasets/bkprocovid19/diving48 "
                "(train sample rgb/zYHstCxnAPA_00361)"
            ),
        ),
        "echonet": MediaSpec(
            name="echonet",
            modality="video",
            path=INPUT_DIR / "echonet_dynamic_example.mp4",
            source=(
                "https://raw.githubusercontent.com/echonet/dynamic/master/docs/media/"
                "example_small.mp4"
            ),
        ),
    }
