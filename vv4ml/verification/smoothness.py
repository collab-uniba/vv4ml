"""Smoothness test for verifying model output regularity under input perturbations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vv4ml.verification.base_result import Result


@dataclass
class SmoothnessResult(Result):
    """Result of a smoothness test for a single output/model combination."""

    max_error_ratio: float
    threshold: float

def _calculate_relative_error_ratio(
    y_plus: np.ndarray,
    y_minus: np.ndarray,
    x_col_values: np.ndarray,
    delta: float,
    y_ref: np.ndarray,
) -> np.ndarray:
    """Compute the MaximumRelativeErrorRatio (normalized partial derivative) for a single dimension."""
    return np.abs((y_plus - y_minus) * x_col_values / (2 * delta * y_ref))

def run_smoothness_test(
    model,
    x_eval: pd.DataFrame,
    input_ranges: list[tuple[float, float]],
    sample_fraction: float,
    perturbation_scaled_magnitude: float,
    threshold: float,
    random_state: int = 42,
) -> SmoothnessResult:
    """Run the smoothness test for a single output/model combination.

    Samples a subset of the evaluation data, then for each sample point and
    each input dimension, creates positive and negative perturbations and
    computes the MaximumRelativeErrorRatio (normalized partial derivative).
    The test passes if the maximum ratio across all points and dimensions
    is below the threshold.

    Args:
        model: Trained model (sklearn-compatible, already loaded).
        x_eval: Evaluation features.
        input_ranges: List of (min, max) tuples per input dimension.
        sample_fraction: Fraction of X_eval to sample.
        perturbation_scaled_magnitude: Perturbation size as fraction of input range.
        threshold: Maximum acceptable error ratio.
        random_state: Random seed for reproducibility.

    Returns:
        SmoothnessResult with the max error ratio and pass/fail status.
    """
    rng = np.random.default_rng(random_state)
    n_eval = len(x_eval)
    n_subset = int(np.ceil(n_eval * sample_fraction))

    # Sample subset of evaluation points
    indices = rng.choice(n_eval, size=n_subset, replace=False)
    x_subset = x_eval.iloc[indices]

    # Reference (unperturbed) predictions
    y_ref = model.predict(x_subset)

    n_dims = x_subset.shape[1]
    results_arr = np.zeros((n_subset, n_dims))

    for j in range(n_dims):
        lo, hi = input_ranges[j]
        delta = (hi - lo) * perturbation_scaled_magnitude

        # Create perturbed copies along dimension j
        col = x_subset.columns[j]
        x_plus = x_subset.copy()
        x_minus = x_subset.copy()
        x_plus[col] = x_subset[col] + delta
        x_minus[col] = x_subset[col] - delta

        y_plus = model.predict(x_plus)
        y_minus = model.predict(x_minus)

        # MaximumRelativeErrorRatio: |(dy * x_j) / (2 * delta * y_ref)|
        results_arr[:, j] = _calculate_relative_error_ratio(
            y_plus, y_minus, x_subset[col].values, delta, y_ref
        )

    max_ratio = float(np.max(results_arr))

    return SmoothnessResult(
        max_error_ratio=max_ratio,
        threshold=threshold,
        passed=max_ratio <= threshold,
    )
