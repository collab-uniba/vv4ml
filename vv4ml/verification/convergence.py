"""Convergence test for verifying model stability across training set sizes."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from vv4ml.utils.metrics import METRICS_REGISTRY, MetricName
from vv4ml.utils.model_factory import recreate_model
from vv4ml.verification.base_result import Result


@dataclass
class ConvergenceResult(Result):
    """Result of a convergence test for a single output/model combination."""

    n_rows_tested: list[int]
    errors: list[float]
    threshold: float
    min_k: int | None


def _error_handle(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    ref_predictions: np.ndarray,
    model_type: str,
    best_params: dict,
    n_rows: list[int],
    metric: Callable[[ArrayLike, ArrayLike], float],
    random_state: int = 42,
) -> list[float]:
    """Executes the convergence test sampling and error computation.

    For each subset size k in ``n_rows``, randomly samples k rows from the
    training data, fits a fresh model with the same hyperparameters as the
    reference model, and compares its predictions on the evaluation set
    against the reference model's predictions using the chosen metric.
    """

    rng = np.random.default_rng(random_state)
    errors: list[float] = []

    for k in n_rows:
        # Randomly sample k rows from the training set
        indices = rng.choice(len(x_train), size=min(k, len(x_train)), replace=False)
        x_subset = x_train.iloc[indices]
        y_subset = y_train.iloc[indices]

        # Instantiate fresh model with same hyperparameters
        fresh_model = recreate_model(model_type, best_params, random_state)
        fresh_model.fit(x_subset, y_subset)

        # Predict on evaluation set and compute MRE
        subset_predictions = fresh_model.predict(x_eval)
        errors.append(metric(subset_predictions, ref_predictions))

    return errors


def _min_k(
    errors: list[float], n_rows: list[int], threshold: float
) -> tuple[int | None, bool]:
    """Find minimum K: the first subset size at which error is below threshold and remains below
    for all subsequent sizes.
    Find the last index where error > threshold, then min_k is the next index's n_rows value.
    If no error exceeds threshold, min_k is the first subset size.

    Args:
        errors: list of gathered errors.
        n_rows: Sorted list of subset sizes to test.
        threshold: error threshold.

    Returns:
        min_k: the first subset size k at which error is below threshold.

    """
    over_threshold = [i for i, e in enumerate(errors) if e > threshold]
    if not over_threshold:
        # All errors are below threshold
        min_k = n_rows[0]
        converged = True
    elif over_threshold[-1] >= len(n_rows) - 1:
        # The last (largest) subset still exceeds threshold — no convergence
        min_k = None
        converged = False
    else:
        # min_k is the n_rows value right after the last exceedance
        min_k = n_rows[over_threshold[-1] + 1]
        converged = True

    return min_k, converged


def run_convergence_test(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    ref_predictions: np.ndarray,
    model_type: str,
    best_params: dict,
    n_rows: list[int],
    threshold: float,
    metric: MetricName,
) -> ConvergenceResult:
    """Run the convergence test for a single output/model combination.

    Convergence is achieved when the error drops below ``threshold`` and stays
    below for all subsequent (larger) subset sizes. The minimum K is the
    first subset size at which this sustained convergence begins.

    Args:
        x_train: Training features.
        y_train: Training target values (single output).
        x_eval: Evaluation features (typically the test set).
        ref_predictions: Reference model predictions on X_eval.
        model_type: Model type key in MODEL_REGISTRY.
        best_params: Best hyperparameters from the reference model.
        n_rows: Sorted list of subset sizes to test.
        threshold: Error threshold for convergence.
        metric: Error metric to be used.

    Returns:
        ConvergenceResult with errors per subset size and convergence info.
    """

    error_metric = METRICS_REGISTRY.get(metric)
    if error_metric is None:
        raise ValueError(f"Invalid metric: {metric}")
    errors = _error_handle(
        x_train,
        y_train,
        x_eval,
        ref_predictions,
        model_type,
        best_params,
        n_rows,
        metric=error_metric,
    )
    min_k, converged = _min_k(errors, n_rows, threshold)

    return ConvergenceResult(
        n_rows_tested=n_rows,
        errors=errors,
        threshold=threshold,
        min_k=min_k,
        passed=converged,
    )
