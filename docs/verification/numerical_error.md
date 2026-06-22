# Numerical Error Test

Machine Learning models can suffer from **floating-point instabilities**, precision degradation, or extreme sensitivity
to random shuffling. The `numerical_error` module evaluates the numerical robustness of your model across multiple
execution sets.

---

## How it works

The test runs your evaluation suite across a configured number of independent validation rounds (`sets`). For each
round, it introduces microscopic perturbations or different data shuffles to catch:

* Floating-point underflows/overflows.
* Excessive variance in predictions due to bad numerical conditioning (e.g., in matrix inversions or high-frequency
weights).

The test passes if the maximum standard deviation of the outputs across all test sets remains below the threshold.

!!! note
    
    `convergence_rows` parameter refers to the minimum k found in the [convergence test](convergence.md)

---

## YAML Configuration

Configure the numerical error gate and the number of evaluation loops in your YAML file:

```yaml
gate_thresholds:
  verification:
    numerical_error:
      - field: "your_field"
        metric: "MaxStandardDeviation" # Strict check on prediction variance
        value: 1e-6                   # Tolerance for precision loss

test_configurations:
  verification:
    numerical_error:
      sets: 10                         # Number of independent validation rounds
```

## Usage

Here's a programmatic test that underlines the logic behind this test:

```python
from vv4ml.verification.numerical_error import run_numerical_error_test

# Assume that all parameters are already prepared
# Evaluate model predictions consistency across 10 different perturbed sets
result = run_numerical_error_test(
    x_train=your_training_input_features,
    y_train=your_training_outout_features,
    x_eval=your_evaluation_data,
    model_type=your_model_type,
    best_params=your_model_best_parameters,
    n_repetitions=10,
    convergence_rows=100,
    threshold=1e-6,
    random_state=42
)
```