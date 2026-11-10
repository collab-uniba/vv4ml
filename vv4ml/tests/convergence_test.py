import unittest
from unittest.mock import patch

from vv4ml.verification.convergence import _min_k, run_convergence_test


class TestConvergence(unittest.TestCase):
    """Test class for convergence module"""

    def test_min_k_all_errors_below_threshold(self):
        """Case 1: All resulting errors are already below the threshold"""
        errors = [0.05, 0.04, 0.02, 0.01]
        n_rows = [100, 200, 500, 1000]
        threshold = 0.1

        min_k, passed = _min_k(errors, n_rows, threshold)

        self.assertTrue(passed)
        self.assertEqual(min_k, 100)

    def test_min_k_never_converges(self):
        """Case 2: the last error exceeds the threshold"""
        errors = [0.5, 0.4, 0.3, 0.15]
        n_rows = [100, 200, 500, 1000]
        threshold = 0.1

        min_k, passed = _min_k(errors, n_rows, threshold)

        self.assertFalse(passed)
        self.assertIsNone(min_k)

    def test_min_k_converges_midway(self):
        """Case 3: Initial errors exceed the threshold, then go and keep below it."""
        # the last error exceeding the threshold (0.1) is 0.12 at index 1 (n_rows = 200)
        errors = [0.25, 0.12, 0.08, 0.05]
        n_rows = [100, 200, 500, 1000]
        threshold = 0.1

        min_k, converged = _min_k(errors, n_rows, threshold)

        self.assertTrue(converged)
        self.assertEqual(min_k, 500)

    def test_min_k_fluctuating_but_stabilizes(self):
        """Case 4: The error is not stable but at the end."""
        # Exceeding at index 0 (0.2) and 2 (0.15). Last exceeding is at index 2 (n_rows = 300)
        errors = [0.2, 0.05, 0.15, 0.04, 0.02]
        n_rows = [100, 200, 300, 400, 500]
        threshold = 0.1

        min_k, converged = _min_k(errors, n_rows, threshold)

        self.assertTrue(converged)
        self.assertEqual(
            min_k, 400
        )  # It must be n_rows[3], where the error goes and keeps below the threshold

    def test_min_k_exact_threshold_edge_case(self):
        """Case 5: Bound value. The error corresponds to the threshold"""
        errors = [0.1, 0.05]
        n_rows = [100, 200]
        threshold = 0.1

        min_k, converged = _min_k(errors, n_rows, threshold)

        self.assertTrue(converged)
        self.assertEqual(min_k, 100)


if __name__ == "__main__":
    unittest.main()


class TestRunConvergenceRandomState(unittest.TestCase):
    """Test that run_convergence_test forwards random_state to the sampling routine"""

    def _run(self, **kwargs):
        with patch(
            "vv4ml.verification.convergence._error_handle", return_value=[0.01, 0.01]
        ) as mock_error_handle:
            run_convergence_test(
                x_train=None,  # type: ignore
                y_train=None,  # type: ignore
                x_eval=None,  # type: ignore
                ref_predictions=None,  # type: ignore
                model_type="any",
                best_params={},
                n_rows=[100, 200],
                threshold=0.1,
                metric="MeanRelativeError",
                **kwargs,
            )
        return mock_error_handle.call_args.kwargs["random_state"]

    def test_default_random_state(self):
        """The default seed is 42, preserving the previous behaviour"""
        self.assertEqual(self._run(), 42)

    def test_custom_random_state_is_forwarded(self):
        """A custom seed is forwarded to the sampling routine"""
        self.assertEqual(self._run(random_state=7), 7)
