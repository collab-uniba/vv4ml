"""Custom scorers and evaluation metrics for model training."""

import numpy as np
from numpy.typing import ArrayLike


def mean_relative_error(y_pred: ArrayLike, y_ref: ArrayLike) -> float:
    """Calculate Mean Relative Error between predictions and reference values.

    Used in convergence tests to compare subset-trained model predictions
    against a reference (fully-trained) model's predictions.

    Args:
        y_pred: Predicted values from the subset-trained model.
        y_ref: Reference predictions from the fully-trained model.

    Returns:
        Mean relative error as a fraction (e.g. 0.05 = 5%).
    """
    y_pred_arr = np.asarray(y_pred)
    y_ref_arr = np.asarray(y_ref)
    return float(np.mean(np.abs((y_pred_arr - y_ref_arr) / y_ref_arr)))
