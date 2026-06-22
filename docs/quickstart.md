# Quickstart

This guide walks you through a basic workflow:

* Preparing a model and dataset
* Running a Verification and Validation (V&V) checks
* Inspecting validation results

## Prerequisites

To follow this quickstart, ensure you have VV4ML installed, along with `scikit-learn` and `pandas` to generate a simple
test model:
```bash
pip install vv4ml scikit-learn pandas
```

## Basic Smoothness Test Example

A practical way to see VV4ML in action is by running a **Smoothness Test**. This test evaluates how stable your model's
outputs are when subjected to small perturbations in the input features.

Create a new Python file (e.g. `smoothness_demo.py`) and paste the following example:
```python
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from vv4ml.verification.smoothness import run_smoothness_test

# Prepare evaluation data as a pandas DataFrame
X_raw, y_raw = make_regression(n_samples=2500, n_features=3, random_state=42)
# Shift target values away from 0. scikit-learn's make_regression centers outputs around 0,
# which causes numerical instability (division-by-zero effects) in the relative error metric.
y_raw = y_raw + 1000
feature_names = ["feat_A", "feat_B", "feat_C"]
x_eval = pd.DataFrame(X_raw, columns=feature_names)

# Train a baseline model
model = LinearRegression()
model.fit(x_eval, y_raw)

# Define the input ranges (min, max) for each dimension
input_ranges = [(x_eval[col].min(), x_eval[col].max()) for col in x_eval.columns]

# Run the test
# In this example we sample 50% of the data, apply a 1% perturbation, and set an error ratio threshold 0f 0.5
result = run_smoothness_test(
    model=model,
    x_eval=x_eval,
    input_ranges=input_ranges,
    sample_fraction=0.5,
    perturbation_scaled_magnitude=0.01,
    threshold=0.5,
    random_state=42,
)

# Inspect the verification results
print(f"Test is passed: {result.passed}")
print(f"Maximum Error Ration found: {result.max_error_ratio}")
print(f"Threshold is set to {result.threshold}")
```

### Output

When you run the script, VV4ML analyses the output and verify if they remain proportional and stable compared to the
input perturbations. If so, test will pass.

Let's roll these out:

* ```result.passed``` is ```True```: your model behaviour is smooth within the defined threshold.
* ```result.passed``` is ```False```: your model might exhibit sudden jumps or high sensitivity to specific feature changes

## Next Steps
* Learn how to drive your tests using YAML in the [Configuration Guide](config.md).
* Explore other verification methods in the [Verification module](verification)