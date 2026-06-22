# Configuration Guide

VV4ML uses a declarative, configuration-driven approach. Instead of hardcoding paths, thresholds, and model parameters
in your scripts, you can define your entire Machine Learning V&V pipeline in a single structured **YAML file**.

Before execution, VV4ML validates this file using **Pydantic** to ensure type safety and prevent runtime failures deep
into your pipeline.

---

## Complete YAML Template

Here is a comprehensive blueprint of a vv4ml compliant `config.yaml` file containing all available blocks.

```yaml
# =============================================================================
# 1. Dataset Configuration
# =============================================================================
dataset:
  name: "housing_model_v1"
  path: "data/raw/housing.csv"
  source: "internal_database"
  provenance: "data_engineering_team"
  version: "1.0.0"
  file_format: "csv"
  duplicate_rows: false
  format_metadata:
    delimiter: ";"
    decimal_separator: "."
    header_lines: 1
    encoding: "UTF-8"
  inputs:
    - name: "feat_A"
      data_type: "float" # Options: float, int
      range: [-3.5, 3.5] # Automatically converted to tuple
      units: "scaled_score"
      missing_values_allowed: false
  outputs:
    - name: "target"
      data_type: "float"
      range: [0.0, 500.0]
      units: "k_euros"

# =============================================================================
# 2. Gate Thresholds
# =============================================================================
gate_thresholds:
  validation:
    - field: "target"
      metric: "RMSE" # See supported metrics below
      value: 15.0
  verification:
    convergence:
      - field: "loss"
        metric: "StandardDeviation"
        value: 0.01
    smoothness:
      - field: "target"
        metric: "MaximumRelativeErrorRatio"
        value: 0.5
    numerical_error:
      - field: "target"
        metric: "MaxStandardDeviation"
        value: 1e-6

# =============================================================================
# 3. Test Configurations
# =============================================================================
test_configurations:
  verification:
    convergence:
      n_rows: [500, 1000, 2500]
    existence:
      samples: 50
    smoothness:
      sample_fraction: 0.5
      perturbation_scaled_magnitude: 0.01
    numerical_error:
      sets: 5

# =============================================================================
# 4. Model Development
# =============================================================================
model_development:
  train_test_split:
    train_fraction: 0.8
    method: "random" # Options: random, grouped
  preprocessing:
    normalization: false
    standardization: true
  optimization:
    optimization_method: "GridSearch" # Options: GridSearch, RandomizedSearchCV
    n_jobs: 1
    cv: 5
    scoring: None
  model_list:
    - type: "LinearRegressor" # Supported types listed below
      label: "baseline_linear" # Optional, defaults to type if omitted
      hyperparameters: {}
    - type: "RandomForestRegressor"
      optimization:
        feature_selection: null
        optimization_method: RandomizedSearchCV
        n_jobs: 1
        cv: 5
      hyperparameters:
        n_estimators: [50, 100, 150]
        max_depth: [5, 10, 150]
        min_samples_split: [2, 10]
        min_samples_leaf: [1, 4]
```

## Supported Reference Fields

### Metric Types

When configuring `gate_thresholds`, the `metric` field strictly accepts one of the following exact strings:

- `MaximumAbsoluteError`, `MeanRelativeError`, `MaximumRelativeErrorRatio`
- `StandardDeviation`, `MaxStandardDeviation`
- `RMSE` (Root Mean Squared Error), `PercentageRMSE`

### Model types

The `model_list.type` field supports the following algorithms out of the box:

- `RandomForestRegressor`
- `XGBoostRegressor`
- `SupportVectorRegressor`
- `LinearRegressor`
- `MLPRegressor`
- `PLSRegression`
- `CatBoostRegressor`
- `GaussianProcessRegressor`

## Python Usage Examples

### Loading a Valid Configuration

To load your YAML configuration file in your Python workflow, use the built-in `load_config` utility. It handles path
expansion (including `~`) and triggers Pydantic validation.

```python
from vv4ml.config import load_config

# Load and validate
config = load_config("vv4ml_config.yaml")

# Data is fully parsed
print(f"Loaded pipeline: {config.dataset.name}")
print(f"Train fraction: {config.model_development.train_test_split.train_fraction}")

# Accessing the first configured smoothness threshold
smoothness_gate = config.gate_thresholds.verification.smoothness[0]
print(f"Gate metric: {smoothness_gate.metric} (Threshold: {smoothness_gate.value})")
```

### Validation Errors

If you accidentally enter an invalid type (e.g. passing a string into a numeric threshold or an unsupported model type),
Pydantic will raise a `ValidationError`. You can catch it with:

```python
from pydantic import ValidationError
from vv4ml.config import load_config

try:
    config = load_config("corrupted_config.yaml")
except ValidationError as err:
    print("Configuration schema validation failed!")
    print(err.json(indent=2))
```