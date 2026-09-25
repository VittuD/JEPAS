from __future__ import annotations

import unittest

from experiments.synthetic_scale_curves import attach_params, curve_rows, to_markdown


def _row(model: str, name: str, boundary: float, params: str = "k=3") -> dict:
    return {"model": model, "method": "kmeans", "params": params,
            "source": f"/x/datasets_derived/synthetic_v1_seed42/image/{name}",
            "boundary_fraction": boundary, "total_connected_components": 10.0,
            "adjacent_cosine_similarity": 0.5}


MANIFEST = {
    "image/checker/checker_0000.png": {"family": "checker", "cells": 2},
    "image/checker/checker_0001.png": {"family": "checker", "cells": 4},
    "image/uniform/uniform_0000.png": {"family": "uniform", "color": [1, 2, 3]},
}


class AttachParamsTest(unittest.TestCase):
    def test_param_comes_from_the_family_field_and_unknown_files_are_dropped(self) -> None:
        rows = [_row("m", "checker/checker_0000.png", 0.1), _row("m", "uniform/uniform_0000.png", 0.2),
                _row("m", "checker/missing.png", 0.3)]
        out = attach_params(rows, MANIFEST)
        self.assertEqual([(r["family"], r["param"]) for r in out], [("checker", "2"), ("uniform", "-")])


class CurveTest(unittest.TestCase):
    def test_groups_by_family_param_and_model_and_orders_numerically(self) -> None:
        rows = attach_params(
            [_row("m", "checker/checker_0001.png", 0.4), _row("m", "checker/checker_0000.png", 0.2),
             _row("m", "checker/checker_0000.png", 0.4), _row("m", "checker/checker_0000.png", 0.9, "k=2")],
            MANIFEST,
        )
        null = [{**_row("m", "n", 0.5), "source": "n"}]
        aggregates = curve_rows(rows, null, method="kmeans", params="k=3")
        self.assertEqual([r["param"] for r in aggregates], ["2", "4"])
        self.assertAlmostEqual(aggregates[0]["boundary_fraction_mean"], 0.3)
        self.assertAlmostEqual(aggregates[0]["boundary_fraction_ratio"], 0.6)
        self.assertIn("### checker: boundary_fraction (ratio)", to_markdown(aggregates))


if __name__ == "__main__":
    unittest.main()
