from __future__ import annotations

import unittest
from pathlib import Path

import torch

from experiments.shared.catalog import (
    IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST,
    default_model_names,
)
from experiments.smoke.ijepa_lite_affinity_novelty_smoke import (
    build_encoder,
    extract_encoder_state_dict,
)


ROOT_DIR = Path(__file__).resolve().parents[2]


class AffinityNoveltyAdapterTests(unittest.TestCase):
    def test_custom_model_is_opt_in(self) -> None:
        self.assertNotIn("ijepa-lite-affinity-novelty", default_model_names())

    def test_default_checkpoint_path_is_repository_relative(self) -> None:
        default_path = Path(
            IJEPA_LITE_AFFINITY_NOVELTY_MANIFEST["checkpoint"]["default_path"]
        )

        self.assertFalse(default_path.is_absolute())
        self.assertEqual(
            default_path,
            Path("weights/ijepa_lite_affinity_novelty/last.pt"),
        )

    def test_extracts_requested_encoder_from_flat_ddp_state(self) -> None:
        payload = {
            "model": {
                "module.target_encoder.vit.weight": torch.ones(1),
                "module.context_encoder.vit.weight": torch.zeros(1),
            }
        }

        target = extract_encoder_state_dict(payload, encoder="target")
        context = extract_encoder_state_dict(payload, encoder="context")

        self.assertEqual(tuple(target), ("vit.weight",))
        self.assertEqual(tuple(context), ("vit.weight",))
        self.assertEqual(target["vit.weight"].item(), 1.0)
        self.assertEqual(context["vit.weight"].item(), 0.0)

    def test_builds_training_geometry_on_meta_device(self) -> None:
        model = build_encoder(ROOT_DIR / "repos" / "ijepa_lite", 224)

        self.assertEqual(tuple(model.vit.encoder.pos_embedding.shape), (1, 256, 1280))
        self.assertEqual(len(model.vit.encoder.layers), 32)
        self.assertEqual(model.vit.conv_proj.weight.device.type, "meta")
        self.assertFalse(model.use_cls_token)

    def test_supports_nondivisible_common_resolution(self) -> None:
        model = build_encoder(ROOT_DIR / "repos" / "ijepa_lite", 384)
        patches = model.vit._process_input(
            torch.empty(1, 3, 384, 384, device="meta")
        )

        self.assertEqual(model.vit.image_size, 384)
        self.assertEqual(tuple(model.vit.encoder.pos_embedding.shape), (1, 729, 1280))
        self.assertEqual(tuple(patches.shape), (1, 729, 1280))


if __name__ == "__main__":
    unittest.main()
