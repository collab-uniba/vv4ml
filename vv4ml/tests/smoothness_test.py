import unittest
import numpy as np
from vv4ml.verification.smoothness import _calculate_relative_error_ratio


class TestSmoothness(unittest.TestCase):
    """Test class for smoothness module."""

    def test_calculate_relative_error_ratio_standard_case(self):
        """Case 1: Standard mathematical calculation of the normalized partial derivative"""
        y_plus = np.array([11.0])
        y_minus = np.array([9.0])
        x_col_values = np.array([5.0])
        delta = 0.5
        y_ref = np.array([10.0])

        result = _calculate_relative_error_ratio(y_plus, y_minus, x_col_values, delta, y_ref)

        self.assertAlmostEqual(result[0], 1.0, places=5)

    def test_calculate_relative_error_ratio_zero_output_delta(self):
        """Case 2: Insensitive model dimension where perturbations produce identical outputs."""
        y_plus = np.array([10.0, 15.0])
        y_minus = np.array([10.0, 15.0])
        x_col_values = np.array([2.0, 4.0])
        delta = 0.1
        y_ref = np.array([10.0, 15.0])

        result = _calculate_relative_error_ratio(y_plus, y_minus, x_col_values, delta, y_ref)

        np.testing.assert_array_almost_equal(result, np.array([0.0, 0.0]), decimal=5)


if __name__ == "__main__":
    unittest.main()
