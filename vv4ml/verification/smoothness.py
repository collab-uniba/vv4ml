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
    n_points: int = 0
    normalization: str = "value"
    max_ratio_per_input: dict[str, float] | None = None
    """Maximum ratio observed for each input variable."""
    argmax_input: str | None = None
    """Input variable at which the overall maximum ratio was observed."""
    ratio_quantiles: dict[str, float] | None = None
    """Distribution of all point/input ratios (keys: ``p50``, ``p95``, ``p99``, ``max``)."""
    n_perturbations_outside_domain: int = 0
    """Number of perturbed configurations lying outside the admissible input domain."""


def _rel_err_ratio(
    y_plus: np.ndarray,
    y_minus: np.ndarray,
    x_col_values: np.ndarray,
    delta: float,
    y_ref: np.ndarray,
) -> np.ndarray:
    """Compute the MaximumRelativeErrorRatio (normalized partial derivative) for a single dimension."""
    return np.abs((y_plus - y_minus) * x_col_values / (2 * delta * y_ref))


def _range_rel_ratio(
    y_plus: np.ndarray,
    y_minus: np.ndarray,
    delta: float,
    input_range: float,
    y_ref: np.ndarray,
) -> np.ndarray:
    """Relative output change per relative input change, with the input change
    normalised by the admissible range of the variable instead of its value."""
    return np.abs((y_plus - y_minus) * input_range / (2 * delta * y_ref))


def run_smoothness_test(
    model,
    x_eval: pd.DataFrame,
    input_ranges: list[tuple[float, float]],
    sample_fraction: float,
    perturbation_scaled_magnitude: float,
    threshold: float,
    random_state: int = 42,
    normalization: str = "value",
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
        normalization: ``"value"`` (default) normalises the input change by the
            unperturbed input value, ``|(dy/y) / (2 delta / x_j)|``; ``"range"``
            normalises it by the admissible range of the input,
            ``|(dy/y) / (2 delta / (max_j - min_j))|``, which remains defined for
            inputs whose admissible interval contains zero.

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
    y_ref = np.asarray(model.predict(x_subset)).ravel()

    if normalization not in ("value", "range"):
        raise ValueError(f"Unsupported normalization: {normalization!r}")

    n_dims = x_subset.shape[1]
    results_arr = np.zeros((n_subset, n_dims))
    n_outside = 0

    for j in range(n_dims):
        lo, hi = input_ranges[j]
        delta = (hi - lo) * perturbation_scaled_magnitude

        # Create perturbed copies along dimension j
        col = x_subset.columns[j]
        x_plus = x_subset.copy()
        x_minus = x_subset.copy()
        x_plus[col] = x_subset[col] + delta
        x_minus[col] = x_subset[col] - delta

        n_outside += int(np.count_nonzero(x_plus[col].values > hi))
        n_outside += int(np.count_nonzero(x_minus[col].values < lo))

        y_plus = np.asarray(model.predict(x_plus)).ravel()
        y_minus = np.asarray(model.predict(x_minus)).ravel()

        if normalization == "value":
            # MaximumRelativeErrorRatio: |(dy * x_j) / (2 * delta * y_ref)|
            results_arr[:, j] = _rel_err_ratio(
                y_plus, y_minus, x_subset[col].values, delta, y_ref
            )
        else:
            results_arr[:, j] = _range_rel_ratio(
                y_plus, y_minus, delta, hi - lo, y_ref
            )

    max_ratio = float(np.max(results_arr))
    per_input = results_arr.max(axis=0)
    cols = list(x_subset.columns)

    return SmoothnessResult(
        max_error_ratio=max_ratio,
        threshold=threshold,
        passed=max_ratio <= threshold,
        n_points=n_subset,
        normalization=normalization,
        max_ratio_per_input={c: float(v) for c, v in zip(cols, per_input)},
        argmax_input=cols[int(np.argmax(per_input))],
        ratio_quantiles={
            "p50": float(np.quantile(results_arr, 0.50)),
            "p95": float(np.quantile(results_arr, 0.95)),
            "p99": float(np.quantile(results_arr, 0.99)),
            "max": max_ratio,
        },
        n_perturbations_outside_domain=n_outside,
    )
