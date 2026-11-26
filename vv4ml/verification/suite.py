"""Configuration-driven execution of the verification tests."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from vv4ml.config import Config, ThresholdCriterion
from vv4ml.utils.latin_hypercube import generate_lhs_samples
from vv4ml.verification.convergence import run_convergence_test
from vv4ml.verification.existence import run_existence_test
from vv4ml.verification.numerical_error import run_numerical_error_test
from vv4ml.verification.smoothness import run_smoothness_test


@dataclass
class VerificationReport:
    """Results of all verification tests executed for one output."""

    output: str
    model_type: str
    results: dict[str, Any] = field(default_factory=dict)
    runtime_s: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "model_type": self.model_type,
            "passed": self.passed,
            "results": {
                k: {**asdict(v), "outcome": v.outcome} for k, v in self.results.items()
            },
            "runtime_s": self.runtime_s,
        }


def _threshold(crits: list[ThresholdCriterion], output: str, test: str) -> ThresholdCriterion:
    for c in crits:
        if c.field == output:
            return c
    raise ValueError(f"No {test} gate threshold configured for output {output!r}")


def build_evaluation_dataset(config: Config) -> pd.DataFrame:
    """Generate the evaluation dataset D from the configuration."""
    vcfg = config.test_configurations.verification
    ev = vcfg.evaluation_dataset
    if ev is None:
        if vcfg.existence is None or vcfg.existence.samples is None:
            raise ValueError(
                "configure test_configurations.verification.evaluation_dataset"
            )
        return generate_lhs_samples(config.dataset.inputs, vcfg.existence.samples)
    return generate_lhs_samples(
        config.dataset.inputs,
        ev.samples,
        random_state=ev.random_state,
        optimization=ev.optimization,
        iter_num=ev.iter_num,
    )


def run_verification(
    config: Config,
    output: str,
    model,
    model_type: str,
    best_params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame | None = None,
    random_state: int = 42,
    tests: tuple[str, ...] = ("existence", "convergence", "numerical_error", "smoothness"),
) -> VerificationReport:
    """Run the configured verification tests for one output.

    Tests are executed in the order existence, convergence, numerical error,
    smoothness. The numerical error test uses the convergence size K found by the
    convergence test; if convergence is not established, it uses the largest
    evaluated training-set size and the report records this fallback.

    Args:
        config: Validated VV4ML configuration.
        output: Name of the output variable being verified.
        model: Reference model trained on the complete training set.
        model_type: Model type key accepted by ``recreate_model``.
        best_params: Hyperparameters of the reference model.
        x_train, y_train: Complete training set.
        x_eval: Evaluation dataset D; generated from the configuration if omitted.
        random_state: Seed for the stochastic steps of the tests.
        tests: Subset of tests to execute (those not configured are skipped).

    Returns:
        VerificationReport with one result per executed test.
    """
    vcfg = config.test_configurations.verification
    gates = config.gate_thresholds.verification
    out_field = next(o for o in config.dataset.outputs if o.name == output)
    input_names = [i.name for i in config.dataset.inputs]

    if x_eval is None:
        x_eval = build_evaluation_dataset(config)
    x_eval = x_eval[input_names]

    report = VerificationReport(output=output, model_type=model_type)
    k: int | None = None

    if "existence" in tests and vcfg.existence is not None:
        t0 = time.perf_counter()
        max_viol = vcfg.existence.max_violations
        if gates.existence:
            max_viol = int(_threshold(gates.existence, output, "existence").value)
        report.results["existence"] = run_existence_test(
            model, x_eval, out_field.range, max_violations=max_viol
        )
        report.runtime_s["existence"] = time.perf_counter() - t0

    ref_pred: np.ndarray | None = None
    if "convergence" in tests and vcfg.convergence is not None:
        t0 = time.perf_counter()
        gate = _threshold(gates.convergence, output, "convergence")
        ref_pred = np.asarray(model.predict(x_eval)).ravel()
        res = run_convergence_test(
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            ref_predictions=ref_pred,
            model_type=model_type,
            best_params=best_params,
            n_rows=sorted(vcfg.convergence.n_rows),
            threshold=gate.value,
            metric=gate.metric,
            random_state=random_state,
        )
        report.results["convergence"] = res
        report.runtime_s["convergence"] = time.perf_counter() - t0
        k = res.min_k

    if "numerical_error" in tests and vcfg.numerical_error is not None:
        t0 = time.perf_counter()
        gate = _threshold(gates.numerical_error, output, "numerical_error")
        if k is None:
            k = max(vcfg.convergence.n_rows) if vcfg.convergence else len(x_train)
            report.runtime_s["numerical_error_fallback_k"] = float(min(k, len(x_train)))
        k = min(k, len(x_train))
        report.results["numerical_error"] = run_numerical_error_test(
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            model_type=model_type,
            best_params=best_params,
            n_repetitions=vcfg.numerical_error.sets,
            convergence_rows=k,
            threshold=gate.value,
            random_state=random_state,
        )
        report.runtime_s["numerical_error"] = time.perf_counter() - t0

    if "smoothness" in tests and vcfg.smoothness is not None:
        t0 = time.perf_counter()
        gate = _threshold(gates.smoothness, output, "smoothness")
        report.results["smoothness"] = run_smoothness_test(
            model=model,
            x_eval=x_eval,
            input_ranges=[tuple(i.range) for i in config.dataset.inputs],
            sample_fraction=vcfg.smoothness.sample_fraction,
            perturbation_scaled_magnitude=vcfg.smoothness.perturbation_scaled_magnitude,
            threshold=gate.value,
            random_state=random_state,
            normalization=vcfg.smoothness.normalization,
        )
        report.runtime_s["smoothness"] = time.perf_counter() - t0

    return report
