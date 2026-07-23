import unittest
import numpy as np
from vv4ml.verification.numerical_error import _max_std_dev_along_axis


class TestNumericalError(unittest.TestCase):
    """Test class for numerical error module"""

    def test_max_std_var_homogeneous_spread(self):
        """Case 1: Standard 2D array where all rows have an identical standard deviation."""
        array_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = _max_std_dev_along_axis(array_data)

        # Using assertAlmostEqual to safely handle floating-point precision
        self.assertAlmostEqual(result, 2.12132034, places=5)

    def test_max_std_var_heterogeneous_spread(self):
        """Case 2: 2D array where different rows exhibit different prediction variations."""
        array_data = np.array([[1.0, 2.0, 3.0], [1.0, 5.0, 9.0]])

        result = _max_std_dev_along_axis(array_data)

        self.assertAlmostEqual(result, 4.24264069, places=5)

    def test_max_std_var_constant_offsets(self):
        """Case 3: Rows are constant, but predictions vary across runs for each column."""
        array_data = np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]])

        result = _max_std_dev_along_axis(array_data)

        self.assertAlmostEqual(result, 2.12132034, places=5)

    def test_max_std_var_empty_array_exception(self):
        """Case 4: Boundary condition where an empty array is passed."""
        empty_array = np.array([])

        with self.assertRaises(Exception) as context:
            _max_std_dev_along_axis(empty_array)

        self.assertEqual(
            str(context.exception), "Can't compute max variance across empty array"
        )


if __name__ == "__main__":
    unittest.main()
