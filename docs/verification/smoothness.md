# Smoothness Test

The Smoothness module evaluates the **local regularity** and adversarial robustness of a Machine Learning model. It
verifies that the model does not exhibit jumps in its predictions when subjected to minor perturbations in the input
features.

This test is critical for ensuring that your model behaves like a continuous, well-conditioned function.

---

## How It Works

For a given evaluation dataset, the test selects a representative subset and applies localized shifts along each
individual feature dimension.

1. **Perturbation Calculation:** For each input dimension $j$, a small step size $\delta$ is determined based on the
feature's total operating range $[\text{min}, \text{max}]$ defined in the configuration:

    $$
        \delta = (\text{max}_j - \text{min}_j) \cdot \text{perturbation_scaled_magnitude}
    $$

2. **Prediction Shifts:** The model evaluates two symmetric perturbed points around the reference point $x$:

    $$
    y_+ = f(x_1, \dots, x_j + \delta, \dots, x_n)
    $$

    $$
    y_- = f(x_1, \dots, x_j - \delta, \dots, x_n)
    $$

3. **Metric Evaluation:** It computes the **Maximum Relative Error Ratio** (a normalized approximation of the local
partial derivative):

    $$
    \text{Ratio} = \left\| \frac{(y_+ - y_-) \cdot x_j}{2 \cdot \delta \cdot y_{\text{ref}}} \right\|
    $$

The test passes only if the highest recorded ratio across all sampled points and all dimensions is less than or equal
to the configured `threshold`.

!!! warning

    If your target variable is naturally centered around 0, a near-zero $y_{\text{ref}}$ in the denominator might cause this ratio to artificially explode. Ensure your evaluation baseline is appropriately scaled or shifted away from zero.

---

## YAML Configuration

To activate the smoothness check, declare the corresponding gate metric and test parameters in your configuration file:

```yaml
gate_thresholds:
  verification:
    smoothness:
      - field: "your_field"
        metric: "MaximumRelativeErrorRatio"
        value: 0.5 # Maximum allowed derivative ratio

test_configurations:
  verification:
    smoothness:
      sample_fraction: 0.5               # Evaluates 50% of the dataset
      perturbation_scaled_magnitude: 0.01 # 1% shift relative to the feature range
```

## Usage

Here's a programmatic test that underlines the logic behind this test:

```python
import pandas as pd
from vv4ml.verification.smoothness import run_smoothness_test

# Assume all parameters are already defined
result = run_smoothness_test(
    model=your_trained_model,
    x_eval=your_evaluation_data,
    input_ranges=your_feature_boundaries,
    sample_fraction=0.3,
    perturbation_scaled_magnitude=0.01,
    threshold=0.5,
    random_state=42
)

print(f"Smoothness Test Passed: {result.passed}")
print(f"Peak Error Ratio: {result.max_error_ratio:.4f}")
print(f"Threshold: {result.threshold}")
```