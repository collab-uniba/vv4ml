import unittest

import numpy as np
import pandas as pd

from vv4ml.verification.existence import run_existence_test


class _Const:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, x):
        return self.values[: len(x)]


class TestExistence(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})

    def test_all_in_range_passes(self):
        r = run_existence_test(_Const([0.1, 0.2, 0.3, 0.4]), self.x, (0.0, 1.0))
        self.assertTrue(r.passed)
        self.assertEqual(r.n_out_of_range, 0)
        self.assertAlmostEqual(r.max_prediction, 0.4)

    def test_out_of_range_and_non_finite_fail(self):
        r = run_existence_test(_Const([-0.1, 0.2, np.nan, 2.0]), self.x, (0.0, 1.0))
        self.assertFalse(r.passed)
        self.assertEqual(r.n_out_of_range, 2)
        self.assertEqual(r.n_non_finite, 1)

    def test_tolerated_violations(self):
        r = run_existence_test(
            _Const([-0.1, 0.2, 0.3, 0.4]), self.x, (0.0, 1.0), max_violations=1
        )
        self.assertTrue(r.passed)


if __name__ == "__main__":
    unittest.main()
