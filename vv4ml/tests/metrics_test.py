import unittest
import numpy as np
from vv4ml.utils.metrics import (
    METRICS_REGISTRY,
    max_absolute_error,
    max_relative_error,
    max_std_dev,
    mean_absolute_error,
    mean_relative_error,
    rmse,
    rmspe,
    std_dev,
)


class TestMetrics(unittest.TestCase):
    """Test class for metrics module"""

    def setUp(self) -> None:
        """Set up standard arrays for testing metrics."""
        # reference: [10.0, 20.0, 50.0]
        # prediction: [11.0, 19.0, 45.0]
        # Absolute differences: [1.0, 1.0, 5.0]
        # Relative differences: [1/10=0.1, 1/20=0.05, 5/50=0.1]
        self.y_ref = [10.0, 20.0, 50.0]
        self.y_pred = [11.0, 19.0, 45.0]

    def test_mean_relative_error(self):
        """Verify Mean Relative Error calculation.

        Expected: (0.1 + 0.05 + 0.1) / 3 = 0.25 / 3 = 0.08333...
        """
        result = mean_relative_error(self.y_pred, self.y_ref)
        self.assertAlmostEqual(result, 0.08333333, places=5)

    def test_max_relative_error(self):
        """Verify Maximum Relative Error calculation.

        Expected: max(0.1, 0.05, 0.1) = 0.1
        """
        result = max_relative_error(self.y_pred, self.y_ref)
        self.assertAlmostEqual(result, 0.1, places=5)

    def test_mean_absolute_error(self):
        """Verify Mean Absolute Error calculation.

        Expected: (1.0 + 1.0 + 5.0) / 3 = 7.0 / 3 = 2.33333...
        """
        result = mean_absolute_error(self.y_pred, self.y_ref)
        self.assertAlmostEqual(result, 2.3333333, places=5)

    def test_max_absolute_error(self):
        """Verify Maximum Absolute Error calculation.

        Expected: max(1.0, 1.0, 5.0) = 5.0
        """
        result = max_absolute_error(self.y_pred, self.y_ref)
        self.assertAlmostEqual(result, 5.0, places=5)

    def test_rmse(self):
        """Verify Root Mean Squared Error calculation.

        Expected: sqrt((1.0^2 + 1.0^2 + 5.0^2) / 3) = 3.0
        """
        result = rmse(self.y_pred, self.y_ref)
        self.assertAlmostEqual(result, 3.0, places=5)

    def test_rmspe(self):
        """Verify Root Mean Squared Percentage Error calculation.

        Expected: sqrt(((1/10)^2 + (-1/20)^2 + (-5/50)^2) / 3) = 0.08660...
        """
        result = rmspe(self.y_pred, self.y_ref)
        self.assertAlmostEqual(result, 0.08660254, places=5)

    def test_std_dev(self):
        """Verify sample standard deviation of prediction-reference differences."""
        result = std_dev(self.y_pred, self.y_ref)
        self.assertAlmostEqual(result, 3.05505046, places=5)

    def test_max_std_dev(self):
        """Verify maximum standard deviation of prediction-reference differences."""
        result = max_std_dev(self.y_pred, self.y_ref)
        self.assertAlmostEqual(result, 3.05505046, places=5)

    def test_metrics_with_perfect_predictions(self):
        """Edge case: Verify that all metrics evaluate to exactly 0.0 when predictions match reference."""
        identical_ref = np.array([1.5, 2.5, 3.5])
        identical_pred = np.array([1.5, 2.5, 3.5])

        self.assertEqual(mean_relative_error(identical_pred, identical_ref), 0.0)
        self.assertEqual(max_relative_error(identical_pred, identical_ref), 0.0)
        self.assertEqual(mean_absolute_error(identical_pred, identical_ref), 0.0)
        self.assertEqual(max_absolute_error(identical_pred, identical_ref), 0.0)
        self.assertEqual(rmse(identical_pred, identical_ref), 0.0)
        self.assertEqual(rmspe(identical_pred, identical_ref), 0.0)
        self.assertEqual(std_dev(identical_pred, identical_ref), 0.0)
        self.assertEqual(max_std_dev(identical_pred, identical_ref), 0.0)

    def test_metrics_registry_mapping(self):
        """Verify that the METRICS_REGISTRY dictionary maps keys to the exact function references."""
        self.assertIs(METRICS_REGISTRY["MeanRelativeError"], mean_relative_error)
        self.assertIs(METRICS_REGISTRY["MaximumRelativeError"], max_relative_error)
        self.assertIs(METRICS_REGISTRY["MeanAbsoluteError"], mean_absolute_error)
        self.assertIs(METRICS_REGISTRY["MaximumAbsoluteError"], max_absolute_error)
        self.assertIs(METRICS_REGISTRY["RMSE"], rmse)
        self.assertIs(METRICS_REGISTRY["PercentageRMSE"], rmspe)
        self.assertIs(METRICS_REGISTRY["StandardDeviation"], std_dev)
        self.assertIs(METRICS_REGISTRY["MaxStandardDeviation"], max_std_dev)


if __name__ == "__main__":
    unittest.main()
