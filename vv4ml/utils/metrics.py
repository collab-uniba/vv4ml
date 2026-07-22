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


def max_relative_error(y_pred: ArrayLike, y_ref: ArrayLike) -> float:
    """Calculate Max Relative Error between predictions and reference values.

    Args:
        y_pred: Predicted values from the subset-trained model.
        y_ref: Reference predictions from the fully trained model.

    Returns:
        Max Relative Error as a fraction (e.g. 0.05 = 5%).
    """
    y_pred_arr = np.asarray(y_pred)
    y_ref_arr = np.asarray(y_ref)

    return float(np.max(np.abs((y_pred_arr - y_ref_arr) / y_ref_arr)))


def mean_absolute_error(y_pred: ArrayLike, y_ref: ArrayLike) -> float:
    """Calculate Mean Absolute Error between predictions and reference values.

    Args:
        y_pred: Predicted values from the subset-trained model.
        y_ref: Reference predictions from the fully trained model.

    Returns:
        Mean Absolute Error as a fraction (e.g. 0.05 = 5%).
    """
    y_pred_arr = np.asarray(y_pred)
    y_ref_arr = np.asarray(y_ref)

    return float(np.mean(np.abs(y_pred_arr - y_ref_arr)))


def max_absolute_error(y_pred: ArrayLike, y_ref: ArrayLike) -> float:
    """Calculate Max Absolute Error between predictions and reference values.

    Args:
        y_pred: Predicted values from the subset-trained model.
        y_ref: Reference predictions from the fully trained model.

    Returns:
        Max Absolute Error as a fraction (e.g. 0.05 = 5%).
    """
    y_pred_arr = np.asarray(y_pred)
    y_ref_arr = np.asarray(y_ref)

    return float(np.max(np.abs(y_pred_arr - y_ref_arr)))


def rmse(y_pred: ArrayLike, y_ref: ArrayLike) -> float:
    """Calculate Root Mean Squared Error between predictions and reference values.

    Args:
        y_pred: Predicted values from the subset-trained model.
        y_ref: Reference predictions from the fully-trained model.

    Returns:
        Root Mean Squared Error as a fraction (e.g. 0.05 = 5%).
    """
    y_pred_arr = np.asarray(y_pred)
    y_ref_arr = np.asarray(y_ref)

    return float(np.sqrt(np.mean(np.square(y_pred_arr - y_ref_arr))))


def rmspe(y_pred: ArrayLike, y_ref: ArrayLike) -> float:
    """Calculate Root Mean Squared Percentage Error between predictions and reference values.

    Args:
        y_pred: Predicted values from the subset-trained model.
        y_ref: Reference predictions from the fully-trained model.

    Returns:
        Root Mean Squared Percentage Error between predictions and reference values.
    """
    y_pred_arr = np.asarray(y_pred)
    y_ref_arr = np.asarray(y_ref)

    return float(np.sqrt(np.mean(np.square((y_pred_arr - y_ref_arr) / y_ref_arr))))


def std_dev(y_pred: ArrayLike, y_ref: ArrayLike) -> float:
    """Calculate Standard Deviation between predictions and reference values.

    Args:
        y_pred: Predicted values from the subset-trained model.
        y_ref: Reference predictions from the fully-trained model.
    Returns:
        Standard Deviation between predictions and reference values.
    """

    y_pred_arr = np.asarray(y_pred)
    y_ref_arr = np.asarray(y_ref)

    return float(np.std(y_pred_arr - y_ref_arr, ddof=1))


def max_std_dev(y_pred: ArrayLike, y_ref: ArrayLike) -> float:
    """Calculate Maximum Standard Deviation between predictions and reference values.

    Args:
        y_pred: Predicted values from the subset-trained model.
        y_ref: Reference predictions from the fully-trained model.
    Returns:
        Maximum Standard Deviation between predictions and reference values.
    """

    y_pred_arr = np.asarray(y_pred)
    y_ref_arr = np.asarray(y_ref)

    return float(np.max(np.std(y_pred_arr - y_ref_arr, ddof=1)))


# =============================================================================
# Metrics Registry
# =============================================================================

# This dictionary maps the available metrics strings directly
# to their respective mathematical functions.
METRICS_REGISTRY = {
    "MeanRelativeError": mean_relative_error,
    "MaximumRelativeError": max_relative_error,
    "MeanAbsoluteError": mean_absolute_error,
    "MaximumAbsoluteError": max_absolute_error,
    "RMSE": rmse,
    "PercentageRMSE": rmspe,
    "StandardDeviation": std_dev,
    "MaxStandardDeviation": max_std_dev,
}
