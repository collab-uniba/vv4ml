from typing import Literal
from pydantic import BaseModel, field_validator


class InputField(BaseModel):
    """Input field specification."""

    name: str
    data_type: Literal["float", "int"]
    range: tuple[float, float]
    units: str
    missing_values_allowed: bool = False

    @field_validator("range", mode="before")
    @classmethod
    def convert_range(cls, v: list[float] | tuple[float, float]) -> tuple[float, float]:
        """Convert list to tuple for range."""
        if isinstance(v, list) and len(v) == 2:
            return (v[0], v[1])
        return v  # type: ignore[return-value]
