"""Validation test: agreement between model predictions and reference outputs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vv4ml.utils.metrics import METRICS_REGISTRY
from vv4ml.verification.base_result import Result


@dataclass
class ValidationResult(Result):
    """Result of a validation test for a single output/model combination."""

    metric: str
    value: float
    threshold: float
    n_samples: int


def run_validation_test(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    metric: str,
    threshold: float,
) -> ValidationResult:
    """Compare model predictions with reference outputs not used for training.

    Args:
        model: Trained model (sklearn-compatible).
        x_test: Features of an independent dataset.
        y_test: Reference outputs (e.g. from the computational model being surrogated).
        metric: Name of a metric in ``METRICS_REGISTRY``.
        threshold: Acceptance threshold; the test passes when metric <= threshold.

    Returns:
        ValidationResult with the metric value and pass/fail status.
    """
    fn = METRICS_REGISTRY.get(metric)
    if fn is None:
        raise ValueError(f"Invalid validation metric: {metric}")
    y_pred = np.asarray(model.predict(x_test), dtype=float).ravel()
    value = float(fn(y_pred, np.asarray(y_test, dtype=float).ravel()))
    return ValidationResult(
        passed=value <= threshold,
        metric=metric,
        value=value,
        threshold=float(threshold),
        n_samples=int(len(y_pred)),
    )
