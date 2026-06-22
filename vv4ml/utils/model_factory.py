"""Model factory for instantiating sklearn/xgboost models from configuration."""

import inspect
from typing import Any

from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


# Dictionary with supported models
MODEL_REGISTRY: dict[str, type[BaseEstimator]] = {
    "RandomForestRegressor": RandomForestRegressor,
    "XGBoostRegressor": XGBRegressor,
    "SupportVectorRegressor": SVR,
    "LinearRegressor": LinearRegression,
    "MLPRegressor": MLPRegressor,
    "PLSRegression": PLSRegression,
    "CatBoostRegressor": CatBoostRegressor,
    "GaussianProcessRegressor": GaussianProcessRegressor,
}

# ---------------------------------------------------------------------------
# GPR kernel registry — maps YAML-safe string names to kernel objects
# ---------------------------------------------------------------------------
KERNEL_REGISTRY: dict[str, Any] = {
    "rbf_short": RBF(length_scale=0.1, length_scale_bounds=(1e-2, 10.0)),
    "rbf_unit": RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)),
    "rbf_dot_white": (
        RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))
        + DotProduct()
        + WhiteKernel(noise_level=1e-3)
    ),
}


def recreate_model(
    model_type: str, best_params: dict[str, Any], random_state: int = 42
) -> BaseEstimator:
    """Reconstruct a model from its type and best_params (e.g. from MLflow).

    Handles GPR (resolves kernel names, builds Pipeline) and all other models
    (coerces string types, instantiates directly).

    Args:
        model_type: Model type key (e.g. ``"RandomForestRegressor"``).
        best_params: Best hyperparameters — values may be strings from MLflow.
        random_state: Random seed for reproducibility.

    Returns:
        Fresh model instance ready for fitting.

    Raises:
        ValueError: If model type is unknown.
    """
    if model_type == "GaussianProcessRegressor":
        return _recreate_gpr_pipeline(best_params, random_state)

    model_class = MODEL_REGISTRY.get(model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}")

    coerced = _coerce_param_types(model_class, best_params)

    if _supports_param(model_class, "random_seed"):
        coerced["random_seed"] = random_state
    elif _supports_param(model_class, "random_state"):
        coerced["random_state"] = random_state

    return model_class(**coerced)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _recreate_gpr_pipeline(best_params: dict[str, Any], random_state: int) -> Pipeline:
    """Reconstruct a GPR Pipeline from normalized best_params."""
    gpr_params: dict[str, Any] = {"random_state": random_state}

    for key, value in best_params.items():
        if key == "kernel":
            kernel = KERNEL_REGISTRY.get(str(value))
            if kernel is None:
                raise ValueError(
                    f"Unknown kernel name: {value!r}. "
                    f"Available: {list(KERNEL_REGISTRY)}"
                )
            gpr_params["kernel"] = kernel
        else:
            # Coerce numeric strings from MLflow
            gpr_params[key] = _coerce_single_value(value)

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gpr", GaussianProcessRegressor(**gpr_params)),
        ]
    )


def _coerce_single_value(value: Any) -> Any:
    """Try to coerce a single string value to int, float, or bool."""
    if not isinstance(value, str):
        return value
    # Bool-like
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    # Int
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _coerce_param_types(model_class: type, params: dict[str, Any]) -> dict[str, Any]:
    """Coerce string parameter values from MLflow to appropriate Python types.

    MLflow stores all parameters as strings. This inspects the model class
    constructor signature to infer the correct types and converts accordingly.
    """
    sig = inspect.signature(model_class.__init__)
    coerced: dict[str, Any] = {}
    for k, v in params.items():
        param = sig.parameters.get(k)
        if param is not None and param.annotation != inspect.Parameter.empty:
            ann = param.annotation
            try:
                if ann is bool or (
                    hasattr(ann, "__origin__") and bool in getattr(ann, "__args__", ())
                ):
                    coerced[k] = (
                        v.lower() in ("true", "1", "yes") if isinstance(v, str) else v
                    )
                elif ann is int:
                    coerced[k] = int(v)
                elif ann is float:
                    coerced[k] = float(v)
                else:
                    coerced[k] = v
                continue
            except (ValueError, TypeError):
                pass

        # Fallback: coerce heuristically
        coerced[k] = _coerce_single_value(v)
    return coerced


def _supports_param(model_class: type, param_name: str) -> bool:
    """Check if model class constructor accepts a given parameter."""
    sig = inspect.signature(model_class.__init__)
    return param_name in sig.parameters
