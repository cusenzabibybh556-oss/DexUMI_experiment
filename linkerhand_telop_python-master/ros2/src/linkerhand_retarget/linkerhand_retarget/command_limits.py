"""Reusable validation and limits for byte-based hand commands."""

import math
from typing import Iterable, List, Sequence, Tuple


class CommandLimiter:
    def __init__(
        self,
        minimum: Sequence[float],
        maximum: Sequence[float],
    ):
        self.minimum = [float(value) for value in minimum]
        self.maximum = [float(value) for value in maximum]
        if not self.minimum or len(self.minimum) != len(self.maximum):
            raise ValueError(
                "command_min and command_max must have equal length"
            )
        if not all(
            math.isfinite(low)
            and math.isfinite(high)
            and 0.0 <= low <= high <= 255.0
            for low, high in zip(self.minimum, self.maximum)
        ):
            raise ValueError(
                "each command limit must satisfy 0 <= min <= max <= 255"
            )

    @property
    def channel_count(self) -> int:
        return len(self.minimum)

    def apply(self, values: Iterable[float]) -> Tuple[List[float], int]:
        command = [float(value) for value in values]
        if len(command) != self.channel_count:
            raise ValueError(
                f"expected {self.channel_count} command values, "
                f"got {len(command)}"
            )
        if not all(math.isfinite(value) for value in command):
            raise ValueError("command contains NaN or infinity")

        limited = [
            max(low, min(high, value))
            for value, low, high in zip(
                command, self.minimum, self.maximum
            )
        ]
        return limited, sum(a != b for a, b in zip(command, limited))
