"""Conservative six-channel Inspire force to five-channel glove feedback."""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from typing import Sequence


INSPIRE_TO_GLOVE = (4, 3, 2, 1, 0)


class InspireFeedbackMapper:
    def __init__(self, baseline_samples: int = 20, threshold: float = 30.0,
                 gain: float = 0.05, alpha: float = 0.25,
                 max_output: float = 30.0, max_step: float = 5.0,
                 stale_ns: int = 150_000_000):
        self.baseline_samples = baseline_samples
        self.threshold = threshold
        self.gain = gain
        self.alpha = alpha
        self.max_output = max_output
        self.max_step = max_step
        self.stale_ns = stale_ns
        self.baseline_window = [deque(maxlen=baseline_samples) for _ in range(5)]
        self.baseline = None
        self.output = [0.0] * 5
        self.last_receive_ns = 0

    def update(self, forces: Sequence[float], receive_ns: int | None = None) -> list[float]:
        if len(forces) != 6 or not all(math.isfinite(float(x)) for x in forces):
            raise ValueError("Inspire feedback requires six finite force values")
        if any(abs(float(x)) > 2000 for x in forces):
            raise ValueError("Inspire force outside expected device range")
        now = time.monotonic_ns() if receive_ns is None else int(receive_ns)
        ordered = [float(forces[index]) for index in INSPIRE_TO_GLOVE]
        self.last_receive_ns = now
        if self.baseline is None:
            for window, value in zip(self.baseline_window, ordered):
                window.append(value)
            if len(self.baseline_window[0]) == self.baseline_samples:
                self.baseline = [statistics.median(window) for window in self.baseline_window]
            return self.output.copy()
        for index, value in enumerate(ordered):
            target = min(self.max_output, max(0.0,
                (value - self.baseline[index] - self.threshold) * self.gain))
            filtered = self.output[index] + self.alpha * (target - self.output[index])
            delta = max(-self.max_step, min(self.max_step, filtered - self.output[index]))
            self.output[index] = max(0.0, min(self.max_output, self.output[index] + delta))
        return self.output.copy()

    def current(self, now_ns: int | None = None) -> list[float]:
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        if (self.baseline is None or not self.last_receive_ns
                or now - self.last_receive_ns > self.stale_ns):
            self.output = [0.0] * 5
        return self.output.copy()
