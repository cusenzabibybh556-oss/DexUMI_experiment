import math

import numpy as np
import pytest

from tron2_restore_trial import validate_recovery_start


class Limits:
    lower = np.full(16, -1.0)
    upper = np.full(16, 1.0)

    def check_joints(self, q):
        q = np.asarray(q)
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise RuntimeError("limit")


def test_recovery_accepts_small_overrun_that_retreats_inside():
    q = np.zeros(16)
    q[4] = -1.0 - math.radians(0.66)
    target = np.zeros(16)
    overrun, joints = validate_recovery_start(Limits(), q, target)
    assert joints == [4]
    assert overrun[4] == pytest.approx(math.radians(0.66))


def test_recovery_rejects_large_overrun():
    q = np.zeros(16)
    q[4] = -1.0 - math.radians(2.01)
    with pytest.raises(RuntimeError, match="大于恢复模式"):
        validate_recovery_start(Limits(), q, np.zeros(16))


def test_recovery_rejects_target_outside_limits():
    q = np.zeros(16)
    q[4] = -1.0 - math.radians(0.5)
    target = np.zeros(16)
    target[4] = -1.1
    with pytest.raises(RuntimeError, match="limit"):
        validate_recovery_start(Limits(), q, target)
