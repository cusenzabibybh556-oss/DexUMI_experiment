"""Direct calibrated mapping from LinkerForce joints to Inspire commands."""

from __future__ import annotations

import math
from typing import Sequence


COMMAND_OPEN = 255
COMMAND_CLOSED = 0
MIN_CALIBRATION_SPAN = 1e-4
CHANNEL_COUNT = 6


def _expand_channels(value, name: str) -> list[int]:
    values = list(value) if isinstance(value, (list, tuple)) else [value] * CHANNEL_COUNT
    if len(values) != CHANNEL_COUNT:
        raise ValueError(f"{name} must be a scalar or contain six values")
    return [int(item) for item in values]


def extract_six_features(values: Sequence[float]) -> list[float]:
    """Return Inspire features in LinkerForce command order."""
    if len(values) != 21:
        raise ValueError(f"expected 21 glove joints, received {len(values)}")
    joints = [float(value) for value in values]
    if not all(math.isfinite(value) for value in joints):
        raise ValueError("glove joints contain a non-finite value")
    return [
        0.2 * joints[2] + 0.8 * joints[4],
        joints[1],
        joints[6],
        joints[10],
        joints[14],
        joints[18],
    ]


def map_calibrated_command(
    current: Sequence[float],
    opened: Sequence[float],
    fist: Sequence[float],
) -> list[int]:
    """Map the user's calibrated open/fist range directly to 255/0."""
    current_features = extract_six_features(current)
    open_features = extract_six_features(opened)
    fist_features = extract_six_features(fist)

    command = []
    for index, (value, open_value, fist_value) in enumerate(zip(
        current_features, open_features, fist_features
    )):
        span = fist_value - open_value
        if abs(span) < MIN_CALIBRATION_SPAN:
            raise ValueError(
                f"Inspire calibration span is too small for channel {index}")
        ratio = min(1.0, max(0.0, (value - open_value) / span))
        command.append(int(round(COMMAND_OPEN * (1.0 - ratio))))
    return command


class InspireCommandGuard:
    """Reject one-frame command spikes before they reach the Inspire bridge.

    Values are in LinkerForce command order:
    thumb bend, thumb rotate, index, middle, ring, little.
    """

    def __init__(
        self,
        max_step=(60, 6, 60, 60, 60, 60),
        jump_threshold=(80, 18, 80, 80, 80, 80),
        confirm_frames=(1, 3, 1, 1, 1, 1),
    ):
        self.max_step = [max(1, value) for value in _expand_channels(max_step, "max_step")]
        self.jump_threshold = [
            max(0, value) for value in _expand_channels(jump_threshold, "jump_threshold")
        ]
        self.confirm_frames = [
            max(1, value) for value in _expand_channels(confirm_frames, "confirm_frames")
        ]
        self.current: list[int] | None = None
        self.pending_direction = [0] * CHANNEL_COUNT
        self.pending_count = [0] * CHANNEL_COUNT
        self.rejected_spikes = [0] * CHANNEL_COUNT

    @staticmethod
    def _clip(value: float) -> int:
        return int(max(COMMAND_CLOSED, min(COMMAND_OPEN, round(value))))

    def reset(self, values: Sequence[float]) -> None:
        if len(values) != CHANNEL_COUNT:
            raise ValueError("Inspire command guard state must contain six values")
        self.current = [self._clip(value) for value in values]
        self.pending_direction = [0] * CHANNEL_COUNT
        self.pending_count = [0] * CHANNEL_COUNT
        self.rejected_spikes = [0] * CHANNEL_COUNT

    def _step_toward(self, previous: int, requested: int, index: int) -> int:
        delta = requested - previous
        if abs(delta) <= self.max_step[index]:
            return requested
        return previous + (self.max_step[index] if delta > 0 else -self.max_step[index])

    def update(self, values: Sequence[float]) -> list[int]:
        if len(values) != CHANNEL_COUNT:
            raise ValueError("Inspire command guard input must contain six values")
        requested = [self._clip(value) for value in values]
        if self.current is None:
            self.reset(requested)
            return list(self.current)

        guarded = []
        for index, raw in enumerate(requested):
            previous = self.current[index]
            delta = raw - previous
            direction = 1 if delta > 0 else -1 if delta < 0 else 0
            if abs(delta) > self.jump_threshold[index]:
                if self.pending_direction[index] == direction:
                    self.pending_count[index] += 1
                else:
                    self.pending_direction[index] = direction
                    self.pending_count[index] = 1
                if self.pending_count[index] < self.confirm_frames[index]:
                    self.rejected_spikes[index] += 1
                    guarded.append(previous)
                    continue
            else:
                self.pending_direction[index] = 0
                self.pending_count[index] = 0

            next_value = self._step_toward(previous, raw, index)
            if next_value == raw:
                self.pending_direction[index] = 0
                self.pending_count[index] = 0
            guarded.append(next_value)

        self.current = guarded
        return list(self.current)
