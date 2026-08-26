from dataclasses import dataclass
from math import isfinite


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not isfinite(self.temperature)
            or self.temperature <= 1e-10
        ):
            raise ValueError("temperature must be finite and greater than 1e-10")
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise TypeError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not isinstance(self.ignore_eos, bool):
            raise TypeError("ignore_eos must be a boolean")
