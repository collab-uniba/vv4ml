"""Numerical error test"""

from array import ArrayType
from dataclasses import dataclass

import numpy as np
import pandas as pd
from base_result import Result
from utils.model_factory import recreate_model


@dataclass
class NumericalErrorResult(Result):
    """Result of a numerical error test"""

    max_variances: list[float]
    repetitions: int
    threshold: float


def max_std_var(a: ArrayType) -> float:
    if len(a) > 0:
        return np.max(np.std(a, ddof=1, axis=1))
    else:
        raise Exception("Can't compute max variance across empty array")


def numerical_error(
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
    """Numerical error test"""
    rng = np.random.default_rng(random_state)
    repetition_variances: list[float] = []
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
        repetition_variances.append(max_std_var(subset_predictions))

    over_threshold = [i for i, e in enumerate(repetition_variances) if e > threshold]
    if not over_threshold:
        passed = True
    else:
        passed = False

    return NumericalErrorResult(
        output_name=output_name,
        model_type=model_type,
        passed=passed,
        threshold=threshold,
        max_variances=repetition_variances,
        repetitions=n_repetitions,
    )
