"""Pure bounded 6D residual pose feedback, for simulation-first evaluation.

All rotations and translation corrections use the robot base frame. Callers
must match nominal pose to the measurement time, independently of command
preview. No robot SDK, network, torque publishing, or filesystem access.
"""
from dataclasses import dataclass, fields
import math
from numbers import Integral, Real

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseConfig:
    gain_per_s: float = .8
    max_translation_m: float = .01
    max_rotation_deg: float = 2.
    max_translation_rate_m_s: float = .003
    max_rotation_rate_deg_s: float = 1.
    max_translation_acceleration_m_s2: float = .02
    max_rotation_acceleration_deg_s2: float = 4.
    translation_deadband_m: float = .0003
    rotation_deadband_deg: float = .05
    max_feedback_age_s: float = .15
    max_feedback_gap_s: float = .15
    max_rtt_s: float = .1
    max_error_m: float = .05
    max_error_deg: float = 15.

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'{f.name} must be positive and finite')


def bounded(v, maximum):
    n = float(np.linalg.norm(v))
    return v*min(1., maximum/max(n, 1e-30))


def pose_values(position, rotation):
    p, r = np.asarray(position, float), np.asarray(rotation, float)
    if (p.shape != (3,) or r.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(r).all()
            or not np.allclose(r.T@r, np.eye(3), atol=1e-6, rtol=0)
            or abs(np.linalg.det(r)-1) > 1e-6):
        raise ValueError('Pose must contain a finite position and SO(3) rotation')
    return p, r


class PoseCorrection:
    def __init__(self, config=None):
        self.config = config or PoseConfig()
        if not isinstance(self.config, PoseConfig):
            raise TypeError('Expected PoseConfig')
        self.translation = np.zeros(3)
        self.rotation = np.eye(3)
        self._target_p = self.translation.copy()
        self._target_r = self.rotation.copy()
        self._velocity_p = np.zeros(3)
        self._velocity_r = np.zeros(3)
        self._now = self._advance_time = self._last_received = self._last_midpoint = None
        self._last_sent = None
        self._seen = set()
        self.status = 'initial'

    def _clock(self, now):
        if isinstance(now, bool) or not isinstance(now, Integral) or now < 0 or (self._now is not None and now < self._now):
            self.freeze()
            raise ValueError('Expected nondecreasing integer monotonic nanoseconds')
        self._now = int(now)

    def freeze(self):
        self._target_p = self.translation.copy()
        self._target_r = self.rotation.copy()
        self._velocity_p[:] = 0
        self._velocity_r[:] = 0
        self._last_midpoint = None
        self.status = 'frozen'

    def observe(self, sample_id, sent_ns, received_ns, now_ns, nominal_p, nominal_r, actual_p, actual_r):
        self._clock(now_ns)
        if (isinstance(sent_ns, bool) or isinstance(received_ns, bool)
                or not isinstance(sent_ns, Integral) or not isinstance(received_ns, Integral)
                or not 0 <= sent_ns <= received_ns <= now_ns):
            self.freeze()
            raise ValueError('Invalid pose feedback times')
        np_, nr = pose_values(nominal_p, nominal_r)
        ap, ar = pose_values(actual_p, actual_r)
        if not isinstance(sample_id, (str, Integral)) or isinstance(sample_id, bool):
            raise ValueError('Invalid pose feedback sample ID')
        if sample_id in self._seen:
            self.status = 'duplicate'
            return False
        self._seen.add(sample_id)
        c = self.config
        if now_ns-received_ns > c.max_feedback_age_s*1e9 or received_ns-sent_ns > c.max_rtt_s*1e9:
            self.freeze();self.status = 'stale_or_high_rtt'
            return False
        if self._last_sent is not None and (sent_ns <= self._last_sent or received_ns <= self._last_received):
            self.freeze();self.status = 'out_of_order'
            return False
        ep = np_-ap
        er = Rotation.from_matrix(nr@ar.T).as_rotvec()
        if np.linalg.norm(ep) > c.max_error_m or np.linalg.norm(er) > np.deg2rad(c.max_error_deg):
            self.freeze()
            raise RuntimeError('6D feedback error exceeds experimental guard')
        mid = (sent_ns+received_ns)//2
        previous = self._last_midpoint
        self._last_sent, self._last_received, self._last_midpoint = sent_ns, received_ns, mid
        if previous is None or mid-previous > c.max_feedback_gap_s*1e9:
            self._target_p = self.translation.copy();self._target_r = self.rotation.copy()
            self.status = 'primed'
            return True
        dt = min((mid-previous)/1e9, .1)
        ep *= max(0., 1-c.translation_deadband_m/max(np.linalg.norm(ep), 1e-30))
        er *= max(0., 1-np.deg2rad(c.rotation_deadband_deg)/max(np.linalg.norm(er), 1e-30))
        vp = bounded(c.gain_per_s*ep, c.max_translation_rate_m_s)
        vr = bounded(c.gain_per_s*er, np.deg2rad(c.max_rotation_rate_deg_s))
        self._target_p = bounded(self.translation+vp*dt, c.max_translation_m)
        rr = Rotation.from_rotvec(vr*dt)*Rotation.from_matrix(self.rotation)
        self._target_r = Rotation.from_rotvec(bounded(rr.as_rotvec(), np.deg2rad(c.max_rotation_deg))).as_matrix()
        self.status = 'tracking'
        return True

    def advance(self, now_ns):
        self._clock(now_ns)
        previous = self._advance_time
        self._advance_time = now_ns
        if previous is None:
            return self.translation.copy(), self.rotation.copy()
        dt = (now_ns-previous)/1e9
        if not dt:
            return self.translation.copy(), self.rotation.copy()
        c = self.config
        if dt > .1 or self._last_received is None or now_ns-self._last_received > c.max_feedback_age_s*1e9:
            self.freeze()
            return self.translation.copy(), self.rotation.copy()
        difference = self._target_p-self.translation
        wanted = bounded(difference/dt, c.max_translation_rate_m_s)
        self._velocity_p += bounded(wanted-self._velocity_p, c.max_translation_acceleration_m_s2*dt)
        step = self._velocity_p*dt
        if np.linalg.norm(step) >= np.linalg.norm(difference):
            step = difference;self._velocity_p[:] = 0
        self.translation = bounded(self.translation+step, c.max_translation_m)
        difference_r = Rotation.from_matrix(self._target_r@self.rotation.T).as_rotvec()
        wanted_r = bounded(difference_r/dt, np.deg2rad(c.max_rotation_rate_deg_s))
        self._velocity_r += bounded(wanted_r-self._velocity_r, np.deg2rad(c.max_rotation_acceleration_deg_s2)*dt)
        step_r = self._velocity_r*dt
        if np.linalg.norm(step_r) >= np.linalg.norm(difference_r):
            step_r = difference_r;self._velocity_r[:] = 0
        rr = Rotation.from_rotvec(step_r)*Rotation.from_matrix(self.rotation)
        self.rotation = Rotation.from_rotvec(bounded(rr.as_rotvec(), np.deg2rad(c.max_rotation_deg))).as_matrix()
        return self.translation.copy(), self.rotation.copy()
