"""Verification tests for ML models."""

from vv4ml.utils.latin_hypercube import generate_lhs_samples
from vv4ml.verification.existence import ExistenceResult, run_existence_test
from vv4ml.verification.convergence import ConvergenceResult, run_convergence_test
from vv4ml.verification.numerical_error import (
    NumericalErrorResult,
    run_numerical_error_test,
)
from vv4ml.verification.smoothness import SmoothnessResult, run_smoothness_test
from vv4ml.verification.suite import (
    VerificationReport,
    build_evaluation_dataset,
    run_verification,
)

__all__ = [
    "ExistenceResult",
    "run_existence_test",
    "VerificationReport",
    "build_evaluation_dataset",
    "run_verification",
    "ConvergenceResult",
    "SmoothnessResult",
    "NumericalErrorResult",
    "generate_lhs_samples",
    "run_convergence_test",
    "run_smoothness_test",
    "run_numerical_error_test",
]
