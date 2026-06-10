from dataclasses import dataclass


@dataclass
class Result:
    output_name: str
    model_type: str
    passed: bool

    @property
    def outcome(self) -> str:
        return "PASSED" if self.passed else "FAILED"
