import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.linear_model import LinearRegression

from vv4ml.config import VerificationThresholds, load_config
from vv4ml.validation import run_validation_test
from vv4ml.verification import build_evaluation_dataset, run_smoothness_test, run_verification

CFG = Path(__file__).resolve().parents[2] / "test_config_file.yaml"


class TestSuite(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CFG)
        self.cfg.test_configurations.verification.existence.samples = 200
        rng = np.random.default_rng(0)
        x = rng.uniform(-4, 4, size=(600, 3))
        y = 1.5 + 0.05 * x[:, 0] + 0.02 * x[:, 1] + rng.normal(0, 1e-3, 600)
        self.x = pd.DataFrame(x, columns=["feature_1", "feature_2", "feature_3"])
        self.y = pd.Series(y, name="target")
        self.model = LinearRegression().fit(self.x, self.y)

    def test_run_verification_all_tests(self):
        rep = run_verification(
            self.cfg, "target", self.model, "LinearRegressor", {}, self.x, self.y
        )
        self.assertEqual(
            set(rep.results), {"existence", "convergence", "numerical_error", "smoothness"}
        )
        self.assertTrue(rep.results["existence"].passed)
        self.assertTrue(rep.results["convergence"].passed)
        d = rep.to_dict()
        self.assertIn("outcome", d["results"]["smoothness"])

    def test_evaluation_dataset_respects_ranges(self):
        d = build_evaluation_dataset(self.cfg)
        self.assertEqual(len(d), 200)
        self.assertTrue(((d >= -4) & (d <= 4)).all().all())

    def test_gate_metric_must_match_test(self):
        with self.assertRaises(ValidationError):
            VerificationThresholds(
                smoothness=[{"field": "t", "metric": "RMSE", "value": 1.0}]
            )

    def test_validation_test(self):
        r = run_validation_test(self.model, self.x, self.y, "RMSE", 0.01)
        self.assertTrue(r.passed)
        self.assertLess(r.value, 0.01)

    def test_range_normalisation_defined_at_zero(self):
        x0 = pd.DataFrame({"a": [0.0, 1.0]})

        class Lin:
            def predict(self, x):
                return 2.0 + np.asarray(x["a"])

        r_val = run_smoothness_test(Lin(), x0, [(-1.0, 1.0)], 1.0, 1e-3, 10.0)
        r_rng = run_smoothness_test(
            Lin(), x0, [(-1.0, 1.0)], 1.0, 1e-3, 10.0, normalization="range"
        )
        # value-normalised ratio vanishes at x=0, range-normalised one does not
        self.assertAlmostEqual(r_val.max_ratio_per_input["a"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(r_rng.max_error_ratio, 1.0, places=6)


class TestConfigValidation(unittest.TestCase):
    def _raw(self):
        import yaml

        return yaml.safe_load(CFG.read_text())

    def _check(self, raw):
        from vv4ml.config import Config

        with self.assertRaises(ValidationError):
            Config.model_validate(raw)

    def test_rejects_unknown_output_in_gate(self):
        raw = self._raw()
        raw["gate_thresholds"]["verification"]["convergence"][0]["field"] = "nope"
        self._check(raw)

    def test_rejects_inverted_range(self):
        raw = self._raw()
        raw["dataset"]["inputs"][0]["range"] = [4.0, -4.0]
        self._check(raw)

    def test_rejects_single_repetition(self):
        raw = self._raw()
        raw["test_configurations"]["verification"]["numerical_error"]["sets"] = 1
        self._check(raw)

    def test_rejects_non_comparison_validation_metric(self):
        raw = self._raw()
        raw["gate_thresholds"]["validation"][0]["metric"] = "OutOfRangeCount"
        self._check(raw)


class TestConvergenceSeed(unittest.TestCase):
    def test_seed_changes_subsets(self):
        from vv4ml.verification import run_convergence_test

        rng = np.random.default_rng(1)
        x = pd.DataFrame(rng.normal(size=(400, 3)), columns=list("abc"))
        y = pd.Series(2.0 + x["a"] + 0.1 * rng.normal(size=400))
        ref = LinearRegression().fit(x, y).predict(x)
        kw = dict(x_train=x, y_train=y, x_eval=x, ref_predictions=ref, model_type="LinearRegressor",
                  best_params={}, n_rows=[20, 50], threshold=0.05, metric="MeanRelativeError")
        a = run_convergence_test(**kw, random_state=1).errors
        b = run_convergence_test(**kw, random_state=1).errors
        c = run_convergence_test(**kw, random_state=2).errors
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
