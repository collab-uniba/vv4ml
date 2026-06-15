"""Numerical error test"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vv4ml.utils.model_factory import recreate_model
from vv4ml.verification.base_result import Result


@dataclass
class NumericalErrorResult(Result):
    """Result of a numerical error test"""

    max_variance: float
    repetitions: int
    threshold: float


def max_std_var(a: np.ndarray) -> float:
    """Compute the maximum standard deviation (variance) over an n-dimensional array"""
    if len(a) > 0:
        return np.max(np.std(a, ddof=1, axis=1))
    else:
        raise Exception("Can't compute max variance across empty array")


def run_numerical_error_test(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_eval: pd.DataFrame,
    model_type: str,
    best_params: dict,
    n_repetitions: int,
    convergence_rows: int,
    threshold: float,
    output_name: str,
    random_state: int = 42,
) -> NumericalErrorResult:
    """Run the numerical error test.

    For ``n_repetitions`` times randomly chooses ``convergence_rows`` rows from the
    training data, fits a fresh model with the same hyperparameters as the reference
    model on subsets of input sets and computes max variance for each of prediction.

    It finally checks for the max variance exceeding threshold. Test succeeds if max
    variance doesn't exceed threshold.

    Args:
        x_train: Training features
        y_train: Training target labels
        x_eval: Validation features
        model_type: Model type key in MODEL_REGISTRY
        best_params: Best hyperparameters for the reference model
        n_repetitions: Number of times when training is performed
        convergence_rows: Number of rows to randomly choose (that is the minimum number
        of rows needed to observe convergence)
        threshold: Maximum tolerance threshold for numeric instability
        output_name: Name of output being tested
        random_state: Random seed for reproducibility

    Returns:
         NumericalErrorResult with maximum variance registered in n_repetitions and
         threshold.
    """
    rng = np.random.default_rng(random_state)
    predictions: list[np.ndarray] = []
    for _ in range(n_repetitions):
        # Randomly choose k independent rows (current_train_rows) without replacement
        current_train_rows = rng.choice(
            len(x_train), size=convergence_rows, replace=False
        )
        x_subset = x_train.iloc[current_train_rows]
        y_subset = y_train.iloc[current_train_rows]
        fresh_model = recreate_model(model_type, best_params, random_state)
        fresh_model.fit(x_subset, y_subset)
        subset_predictions = fresh_model.predict(x_eval)
        predictions.append(subset_predictions)

    max_variance = max_std_var(np.array(predictions))
    passed = max_variance <= threshold

    return NumericalErrorResult(
        output_name=output_name,
        model_type=model_type,
        passed=passed,
        threshold=threshold,
        max_variance=max_variance,
        repetitions=n_repetitions,
    )
