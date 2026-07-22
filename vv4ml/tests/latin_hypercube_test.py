import unittest
from dataclasses import dataclass
import numpy as np
import pandas as pd
from vv4ml.utils.latin_hypercube import (
    _lh_basic,
    _lh_maximin,
    lh_gen,
    generate_lhs_samples,
)


@dataclass
class MockInputField:
    """Minimal mock to mimic the behaviour of vv4ml.config.InputField without its dependencies."""

    name: str
    range: tuple[float, float]


class TestLatinHypercubeSampling(unittest.TestCase):
    """test class for latin hypercube module"""

    def test_lh_basic_stratification_property(self):
        """Case 1: Verify that _lh_basic satisfies the core Latin Hypercube stratification property."""
        sample_size = 10
        par_num = 3
        rand_gen = np.random.default_rng(seed=42)

        samples = _lh_basic(par_num, sample_size, rand_gen)

        # Check structural properties
        self.assertEqual(samples.shape, (sample_size, par_num))
        self.assertTrue(np.all(samples >= 0.0) and np.all(samples <= 1.0))

        # Check stratification: map values to their respective bin indexes [0, N-1]
        for col_idx in range(par_num):
            bins = np.floor(samples[:, col_idx] * sample_size).astype(int)
            # Sorting the bin indices must yield exactly [0, 1, 2, ..., N-1] (no duplicates)
            unique_bins = np.unique(bins)
            self.assertEqual(len(unique_bins), sample_size)
            self.assertEqual(list(unique_bins), list(range(sample_size)))

    def test_lh_maximin_optimization_validity(self):
        """Case 2: Verify that _lh_maximin runs successfully and returns a valid space-filling matrix."""
        sample_size = 5
        par_num = 2
        rand_gen = np.random.default_rng(seed=42)
        iter_num = 3

        optim_samples = _lh_maximin(par_num, sample_size, rand_gen, iter_num)

        # Output dimensions must remain structurally intact
        self.assertEqual(optim_samples.shape, (sample_size, par_num))
        self.assertTrue(np.all(optim_samples >= 0.0) and np.all(optim_samples <= 1.0))

    def test_lh_gen_invalid_optimization_raises_value_error(self):
        """Case 3: Exception boundary condition. Passing an invalid optimization token must raise ValueError."""
        with self.assertRaises(ValueError) as context:
            # Type ignoring intentionally to test defensive runtime checks
            lh_gen(par_num=2, sample_size=5, optimization="invalid_method")  # type: ignore

        self.assertIn("Unsupported optimization method", str(context.exception))

    def test_generate_lhs_samples_scaling(self):
        """Case 4: Verify that [0, 1] samples are linearly scaled to custom InputField bounding ranges."""
        # Setup mock fields representing asymmetric real-world bounds
        mock_inputs = [
            MockInputField(name="feature_alpha", range=(-10.0, 10.0)),
            MockInputField(name="feature_beta", range=(100.0, 200.0)),
        ]
        sample_size = 20

        # We can pass mock_inputs since the function only accesses '.range' and '.name' properties
        df_result = generate_lhs_samples(mock_inputs, sample_size, random_state=42)  # type: ignore

        # Structural assertions on DataFrame
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertEqual(df_result.shape, (sample_size, 2))
        self.assertEqual(list(df_result.columns), ["feature_alpha", "feature_beta"])

        # Boundary assertions ensuring constraints are respected
        self.assertTrue((df_result["feature_alpha"] >= -10.0).all())
        self.assertTrue((df_result["feature_alpha"] <= 10.0).all())
        self.assertTrue((df_result["feature_beta"] >= 100.0).all())
        self.assertTrue((df_result["feature_beta"] <= 200.0).all())


if __name__ == "__main__":
    unittest.main()
