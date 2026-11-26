"""Existence test: model outputs must lie in the admissible output space."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vv4ml.verification.base_result import Result


@dataclass
class ExistenceResult(Result):
    """Result of an existence test for a single output/model combination."""

    n_evaluated: int
    n_out_of_range: int
    n_non_finite: int
    output_range: tuple[float, float]
    min_prediction: float
    max_prediction: float


def run_existence_test(
    model,
    x_eval: pd.DataFrame,
    output_range: tuple[float, float],
    max_violations: int = 0,
) -> ExistenceResult:
    """Run the existence test for a single output/model combination.

    The model is evaluated at every point of ``x_eval``. A prediction violates the
    test when it is not finite (NaN or infinite) or falls outside the closed
    admissible interval ``output_range``. The test passes when the number of
    violations does not exceed ``max_violations`` (zero by default).

    Args:
        model: Trained model (sklearn-compatible).
        x_eval: Evaluation features.
        output_range: Admissible (min, max) interval of the output.
        max_violations: Maximum number of tolerated violations.

    Returns:
        ExistenceResult with violation counts and the observed prediction range.
    """
    lo, hi = output_range
    y = np.asarray(model.predict(x_eval), dtype=float).ravel()

    finite = np.isfinite(y)
    n_non_finite = int(np.count_nonzero(~finite))
    n_out = int(np.count_nonzero(finite & ((y < lo) | (y > hi))))
    n_viol = n_out + n_non_finite

    return ExistenceResult(
        passed=n_viol <= max_violations,
        n_evaluated=int(y.size),
        n_out_of_range=n_out,
        n_non_finite=n_non_finite,
        output_range=(float(lo), float(hi)),
        min_prediction=float(np.min(y[finite])) if finite.any() else float("nan"),
        max_prediction=float(np.max(y[finite])) if finite.any() else float("nan"),
    )
