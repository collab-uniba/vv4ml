from dataclasses import dataclass


@dataclass
class Result:
    """Test result for a single model/input combination"""

    passed: bool  # If the test passes

    @property
    def outcome(self) -> str:
        return "PASSED" if self.passed else "FAILED"
