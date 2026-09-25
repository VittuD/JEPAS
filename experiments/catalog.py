"""Model and example-media definitions used by every experiment command."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "experiments" / "inputs" / "web_examples"
MODEL_DIR = ROOT_DIR / "experiments" / "models"
CUSTOM_MANIFEST_PATH = ROOT_DIR / "manifests" / "ijepa_lite_affinity_novelty.json"
CUSTOM_MANIFEST = json.loads(CUSTOM_MANIFEST_PATH.read_text())


def custom_checkpoint_path() -> Path:
    configured = Path(
        os.environ.get(
            "IJEPA_LITE_AFFINITY_NOVELTY_CHECKPOINT",
            CUSTOM_MANIFEST["checkpoint"]["default_path"],
        )
    ).expanduser()
    return configured if configured.is_absolute() else ROOT_DIR / configured


@dataclass(frozen=True)
class ModelSpec:
    name: str
    modality: str
    input_kinds: tuple[str, ...]
    native_resolution: int
    patch_size: int
    adapter: Path
    weights_flag: str
    weights_path: Path
    checkpoint_path: Path
    model_code_path: Path
    checkpoint_source: str
    positional_geometry: str
    adapter_args: tuple[str, ...] = ()
    compare_by_default: bool = False
    configuration_path: Path | None = None


@dataclass(frozen=True)
class MediaSpec:
    name: str
    modality: str
    path: Path
    source: str


def model_specs() -> dict[str, ModelSpec]:
    vjepa_adapter = MODEL_DIR / "vjepa2.py"
    vjepa_code = ROOT_DIR / "repos" / "vjepa2"
    rad_dir = ROOT_DIR / "weights" / "radjepa" / "hf-RadJEPA"
    neuro_dir = ROOT_DIR / "weights" / "neurojepa" / "Neuro-JEPA"
    custom_checkpoint = custom_checkpoint_path()
    return {
        "ijepa": ModelSpec(
            name="ijepa",
            modality="image",
            input_kinds=("image",),
            native_resolution=224,
            patch_size=14,
            adapter=MODEL_DIR / "ijepa.py",
            weights_flag="--checkpoint",
            weights_path=ROOT_DIR / "weights" / "ijepa" / "IN1K-vit.h.14-300e.pth.tar",
            checkpoint_path=ROOT_DIR / "weights" / "ijepa" / "IN1K-vit.h.14-300e.pth.tar",
            model_code_path=ROOT_DIR / "repos" / "ijepa",
            checkpoint_source="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar",
            positional_geometry="bicubic checkpoint position-grid interpolation",
            compare_by_default=True,
        ),
        "radjepa": ModelSpec(
            name="radjepa",
            modality="image",
            input_kinds=("image",),
            native_resolution=224,
            patch_size=14,
            adapter=MODEL_DIR / "radjepa.py",
            weights_flag="--model-dir",
            weights_path=rad_dir,
            checkpoint_path=rad_dir / "jepa_encoder.pth.tar",
            model_code_path=rad_dir / "modeling_radjepa.py",
            checkpoint_source="https://huggingface.co/AIDELab-IITBombay/RadJEPA",
            positional_geometry="bicubic checkpoint position-grid interpolation",
            compare_by_default=True,
        ),
        "vjepa2-vitl": ModelSpec(
            name="vjepa2-vitl",
            modality="video",
            input_kinds=("image", "video"),
            native_resolution=256,
            patch_size=16,
            adapter=vjepa_adapter,
            weights_flag="--checkpoint",
            weights_path=ROOT_DIR / "weights" / "vjepa2" / "vitl.pt",
            checkpoint_path=ROOT_DIR / "weights" / "vjepa2" / "vitl.pt",
            model_code_path=vjepa_code,
            checkpoint_source="https://dl.fbaipublicfiles.com/vjepa2/vitl.pt",
            positional_geometry="source-model RoPE at the runtime grid",
            adapter_args=("--variant", "vjepa2-vitl"),
            compare_by_default=True,
        ),
        "vjepa2-vith": ModelSpec(
            name="vjepa2-vith",
            modality="video",
            input_kinds=("image", "video"),
            native_resolution=256,
            patch_size=16,
            adapter=vjepa_adapter,
            weights_flag="--checkpoint",
            weights_path=ROOT_DIR / "weights" / "vjepa2" / "vith.pt",
            checkpoint_path=ROOT_DIR / "weights" / "vjepa2" / "vith.pt",
            model_code_path=vjepa_code,
            checkpoint_source="https://dl.fbaipublicfiles.com/vjepa2/vith.pt",
            positional_geometry="source-model RoPE at the runtime grid",
            adapter_args=("--variant", "vjepa2-vith"),
        ),
        "vjepa2-vitg": ModelSpec(
            name="vjepa2-vitg",
            modality="video",
            input_kinds=("image", "video"),
            native_resolution=256,
            patch_size=16,
            adapter=vjepa_adapter,
            weights_flag="--checkpoint",
            weights_path=ROOT_DIR / "weights" / "vjepa2" / "vitg.pt",
            checkpoint_path=ROOT_DIR / "weights" / "vjepa2" / "vitg.pt",
            model_code_path=vjepa_code,
            checkpoint_source="https://dl.fbaipublicfiles.com/vjepa2/vitg.pt",
            positional_geometry="source-model RoPE at the runtime grid",
            adapter_args=("--variant", "vjepa2-vitg"),
        ),
        "vjepa2-vitg-384": ModelSpec(
            name="vjepa2-vitg-384",
            modality="video",
            input_kinds=("image", "video"),
            native_resolution=384,
            patch_size=16,
            adapter=vjepa_adapter,
            weights_flag="--checkpoint",
            weights_path=ROOT_DIR / "weights" / "vjepa2" / "vitg-384.pt",
            checkpoint_path=ROOT_DIR / "weights" / "vjepa2" / "vitg-384.pt",
            model_code_path=vjepa_code,
            checkpoint_source="https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt",
            positional_geometry="native 384 source-model RoPE",
            adapter_args=("--variant", "vjepa2-vitg-384"),
        ),
        "vjepa2-1-vitb": ModelSpec(
            name="vjepa2-1-vitb",
            modality="video",
            input_kinds=("image", "video"),
            native_resolution=384,
            patch_size=16,
            adapter=vjepa_adapter,
            weights_flag="--checkpoint",
            weights_path=(
                ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitb_dist_vitG_384.pt"
            ),
            checkpoint_path=(
                ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitb_dist_vitG_384.pt"
            ),
            model_code_path=vjepa_code,
            checkpoint_source=(
                "https://dl.fbaipublicfiles.com/vjepa2/"
                "vjepa2_1_vitb_dist_vitG_384.pt"
            ),
            positional_geometry="native 384 source-model RoPE",
            adapter_args=("--variant", "vjepa2-1-vitb"),
            compare_by_default=True,
        ),
        "vjepa2-1-vitl": ModelSpec(
            name="vjepa2-1-vitl",
            modality="video",
            input_kinds=("image", "video"),
            native_resolution=384,
            patch_size=16,
            adapter=vjepa_adapter,
            weights_flag="--checkpoint",
            weights_path=ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitl_dist_vitG_384.pt",
            checkpoint_path=ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitl_dist_vitG_384.pt",
            model_code_path=vjepa_code,
            checkpoint_source="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
            positional_geometry="native 384 source-model RoPE",
            adapter_args=("--variant", "vjepa2-1-vitl"),
        ),
        "vjepa2-1-vitg": ModelSpec(
            name="vjepa2-1-vitg",
            modality="video",
            input_kinds=("image", "video"),
            native_resolution=384,
            patch_size=16,
            adapter=vjepa_adapter,
            weights_flag="--checkpoint",
            weights_path=ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitg_384.pt",
            checkpoint_path=ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitg_384.pt",
            model_code_path=vjepa_code,
            checkpoint_source="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt",
            positional_geometry="native 384 source-model RoPE",
            adapter_args=("--variant", "vjepa2-1-vitg"),
        ),
        "vjepa2-1-vitG": ModelSpec(
            name="vjepa2-1-vitG",
            modality="video",
            input_kinds=("image", "video"),
            native_resolution=384,
            patch_size=16,
            adapter=vjepa_adapter,
            weights_flag="--checkpoint",
            weights_path=ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitG_384.pt",
            checkpoint_path=ROOT_DIR / "weights" / "vjepa2" / "vjepa2_1_vitG_384.pt",
            model_code_path=vjepa_code,
            checkpoint_source="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt",
            positional_geometry="native 384 source-model RoPE",
            adapter_args=("--variant", "vjepa2-1-vitG"),
        ),
        "echojepa": ModelSpec(
            name="echojepa",
            modality="video",
            input_kinds=("image", "video"),
            native_resolution=224,
            patch_size=16,
            adapter=MODEL_DIR / "echojepa.py",
            weights_flag="--checkpoint",
            weights_path=ROOT_DIR / "weights" / "echojepa" / "vitl-vmix22m-pt220-c55.pt",
            checkpoint_path=ROOT_DIR / "weights" / "echojepa" / "vitl-vmix22m-pt220-c55.pt",
            model_code_path=ROOT_DIR / "repos" / "EchoJEPA",
            checkpoint_source="https://drive.google.com/uc?id=1T_ubAMpDMEByH7V6TJu9iT5Yf9IxlDBB",
            positional_geometry="source-model RoPE at the runtime grid",
            compare_by_default=True,
        ),
        "neurojepa": ModelSpec(
            name="neurojepa",
            modality="volume",
            input_kinds=("volume",),
            native_resolution=96,
            patch_size=8,
            adapter=MODEL_DIR / "neurojepa.py",
            weights_flag="--model-dir",
            weights_path=neuro_dir,
            checkpoint_path=neuro_dir / "model.safetensors",
            model_code_path=ROOT_DIR / "repos" / "Neuro-JEPA",
            checkpoint_source="https://huggingface.co/rajpurkarlab/Neuro-JEPA",
            positional_geometry="native 3D learned position embedding",
        ),
        "ijepa-lite-affinity-novelty": ModelSpec(
            name="ijepa-lite-affinity-novelty",
            modality="image",
            input_kinds=("image",),
            native_resolution=int(CUSTOM_MANIFEST["model"]["image_size"]),
            patch_size=int(CUSTOM_MANIFEST["model"]["patch_size"]),
            adapter=MODEL_DIR / "ijepa_lite.py",
            weights_flag="--checkpoint",
            weights_path=custom_checkpoint,
            checkpoint_path=custom_checkpoint,
            model_code_path=ROOT_DIR / "repos" / "ijepa_lite",
            checkpoint_source=str(CUSTOM_MANIFEST["checkpoint"]["source_uri"]),
            positional_geometry="learned patch-only grid with bicubic interpolation",
            adapter_args=("--encoder", str(CUSTOM_MANIFEST["checkpoint"]["encoder"])),
            configuration_path=CUSTOM_MANIFEST_PATH,
        ),
    }


MODEL_ALIASES = {
    "vjepa2": "vjepa2-vitl",
    "vjepa2-1": "vjepa2-1-vitb",
    "echojepa-vjepa2": "echojepa",
}


def canonical_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def default_compare_models() -> tuple[str, ...]:
    return tuple(name for name, spec in model_specs().items() if spec.compare_by_default)


def media_specs() -> dict[str, MediaSpec]:
    return {
        "dog": MediaSpec(
            "dog",
            "image",
            INPUT_DIR / "dog_image.jpg",
            "https://commons.wikimedia.org/wiki/File:Dog_image.jpg",
        ),
        "chest_xray": MediaSpec(
            "chest_xray",
            "image",
            INPUT_DIR / "chest_xray_pneumonia.jpg",
            "https://commons.wikimedia.org/wiki/File:Pneumonia_x_ray.jpg",
        ),
        "diving48": MediaSpec(
            "diving48",
            "video",
            INPUT_DIR / "diving48_sample.mp4",
            "https://huggingface.co/datasets/bkprocovid19/diving48",
        ),
        "echonet": MediaSpec(
            "echonet",
            "video",
            INPUT_DIR / "echonet_dynamic_example.mp4",
            "https://github.com/echonet/dynamic",
        ),
    }


MATCHED_INPUT = {
    "ijepa": "dog",
    "radjepa": "chest_xray",
    "vjepa2-vitl": "diving48",
    "vjepa2-vith": "diving48",
    "vjepa2-1-vitb": "diving48",
    "vjepa2-1-vitg": "diving48",
    "vjepa2-vitg": "diving48",
    "vjepa2-vitg-384": "diving48",
    "vjepa2-1-vitl": "diving48",
    "vjepa2-1-vitG": "diving48",
    "echojepa": "echonet",
    "ijepa-lite-affinity-novelty": "chest_xray",
}
