# Convergence Test

The Convergence module evaluates **data volume convergence**. It determines whether your model's performance has
stabilized or if it would significantly benefit from more training data. 

By training the model on progressively larger subsets of your data, VV4ML analyses the stability of the resulting
evaluation metrics.

---

## How It Works

Given a list of dataset sizes $[N_1, N_2, \dots, N_k]$ configured via YAML, VV4ML:

1. Randomly Subsamples the training dataset (the sampling is seeded by `random_state`, default `42`, which is also 
used as the seed of the refitted models).
2. Trains the chosen model architecture on each subset.
3. Computes the target metric (e.g., `MeanRelativeError` or `MaxAbsoluteError`) for each step.
4. Find the minimum k at which convergence is observed.
5. Checks if the metric stabilizes. If the metric fluctuation across the largest data sizes is below your threshold, 
the test passes

---

## YAML Configuration

To activate and configure this test, define the `convergence` blocks under both `gate_thresholds` and `test_configurations`:

```yaml
gate_thresholds:
  verification:
    convergence:
      - field: "your_field_name"
        metric: "MeanRelativeError"
        value: 0.05                 

test_configurations:
  verification:
    convergence:
      n_rows: [500, 1000, 2500]     # Subsets of data to evaluate
```

## Usage

Here's a programmatic test that underlines the logic behind this test:

```python
from vv4ml.verification.convergence import run_convergence_test

# Assuming all parameters are already prepared
result = run_convergence_test(
    x_train=your_training_input_features,
    y_train=your_training_output_features,
    x_eval=your_evaluation_data,
    ref_predictions=your_reference_predictions,
    model_type=your_model_type,
    best_params=your_model_best_parameters,
    n_rows=[500, 1000, 2500],
    threshold=0.01,
    metric=your_metric,
    random_state=42             # Optional, defaults to 42
)

print(f"Convergence Test Passed: {result.passed}")
print(f"Found minimum k: {result.min_k}")
print(f"Tested rows: {result.n_rows_tested}")
```