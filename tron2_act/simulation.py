"""Simulation-only streaming adapter for the existing Cartesian ServoP reference.

The reference/feedback code is shared with replay; the plant's IK/PD is not the
vendor's ServoP implementation. All clocks in this module are simulation time.
"""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tron2_act.geometry import matrix_to_rotation6d, rotation6d_to_matrix
from tron2_pose_compensation import PoseCorrection
from tron2_pose_reference import PoseReferenceModel, limit_correction, load_profile
from tron2_pose_compensation import bounded


HAND_JOINTS = ["thumb_cmc_pitch", "thumb_cmc_yaw", "index_mcp_pitch",
               "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]
HAND_LIMITS = np.array([.52, 1.54, 1.57, 1.57, 1.57, 1.57])


def hand_to_angles(command):
    command = np.asarray(command, float)
    if command.shape != (6,) or not np.isfinite(command).all():
        raise ValueError("Expected six finite O6 SDK commands")
    return (1 - np.clip(command, 0, 255) / 255) * HAND_LIMITS


def angles_to_hand(angles):
    return np.clip(255 * (1 - np.asarray(angles) / HAND_LIMITS), 0, 255)


class PoseHistory:
    def __init__(self):
        self.samples = deque()

    def append(self, t, p, r):
        self.samples.append((float(t), np.array(p), np.array(r)))
        while len(self.samples) > 2 and self.samples[1][0] < t - 1:
            self.samples.popleft()

    def state(self, t, p, r, hand, steps=2, hz=20):
        times = np.array([x[0] for x in self.samples])
        at = t - np.arange(steps, 0, -1) / hz
        if at.min(initial=t) < times[0] - 1e-8:
            raise ValueError("Insufficient measured wrist history")
        ps = np.array([x[1] for x in self.samples])
        rs = np.array([x[2] for x in self.samples])
        past_p = np.column_stack([np.interp(at, times, ps[:, j]) for j in range(3)])
        past_r = Slerp(times, Rotation.from_matrix(rs))(np.clip(at, times[0], times[-1])).as_matrix()
        return np.r_[np.concatenate([past_p-p, matrix_to_rotation6d(past_r @ r.T)], -1).ravel(), hand].astype(np.float32)


class ActionChunk:
    def __init__(self, t, p, r, hand, action, hz=20):
        action = np.asarray(action, float)
        if action.ndim != 2 or action.shape[1] != 15 or not np.isfinite(action).all():
            raise ValueError("Invalid ACT chunk")
        self.times = t + np.arange(len(action)+1)/hz
        self.positions = np.vstack([p, p+action[:, :3]])
        rotations = np.concatenate([r[None], rotation6d_to_matrix(action[:, 3:9]) @ r])
        self.rotations = Slerp(self.times, Rotation.from_matrix(rotations))
        self.hands = np.vstack([hand, np.clip(action[:, 9:], 0, 255)])

    def target(self, t):
        t = float(np.clip(t, self.times[0], self.times[-1]))
        p = np.array([np.interp(t, self.times, self.positions[:, j]) for j in range(3)])
        r = self.rotations(t).as_matrix()
        # O6 targets are command events, not interpolated future feedback.
        index = np.searchsorted(self.times, t+1e-9, side="right")-1
        return p, r, self.hands[index].copy()


class HandCommandClock:
    """Hold sent O6 targets across replans; measured q_pos never replaces a target."""
    def __init__(self, initial):
        self.command = np.array(initial, float)
        self.chunk = None

    def submit(self, chunk):
        self.chunk = chunk

    def advance(self, t):
        if self.chunk is not None and t >= self.chunk.times[1]-1e-9:
            self.command = self.chunk.target(t)[2]
        return self.command.copy()


class SimServoP:
    """Same pose feedback, model feedforward, preview and correction slew as replay."""
    def __init__(self, q0, feedback=True, continuous_reference=True, preview_s=None, ik_solver='legacy'):
        self.profile = load_profile()
        self.preview_s = self.profile['preview_s'] if preview_s is None else float(preview_s)
        if not np.isfinite(self.preview_s) or not 0 <= self.preview_s <= .2:
            raise ValueError('Preview must be within 0..200ms')
        if ik_solver not in {'legacy','bounded'}:
            raise ValueError('Expected legacy or bounded simulator IK')
        if ik_solver == 'bounded':
            from tron2_act.simulation_ik import BoundedPoseReferenceModel
            self.engine = BoundedPoseReferenceModel(self.profile,q0)
        else:
            self.engine = PoseReferenceModel(self.profile, q0)
        self.controller = PoseCorrection()
        self.feedback = feedback
        self.last_p = np.zeros(3)
        self.last_r = np.eye(3)
        self.last_t = None
        self.continuous_reference = continuous_reference
        self.filtered_p, self.filtered_r, _ = self.engine.forward(q0)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    def observe(self, t, nominal_p, nominal_r, p, r, now=None):
        if self.feedback:
            ns = round((t+1)*1e9)
            now_ns = round(((t if now is None else now)+1)*1e9)
            # Direct physics snapshots have a known capture time and zero transport RTT.
            self.controller.observe(ns, ns, ns, now_ns, nominal_p, nominal_r, p, r)

    def step(self, t, nominal, ramp=1.):
        dt = 1/300 if self.last_t is None else t-self.last_t
        ns = round((t+1)*1e9)
        fp, fr = self.controller.advance(ns) if self.feedback else (np.zeros(3), np.eye(3))
        pp, pr, _ = nominal(t+self.preview_s)
        requested_p, requested_r = pp.copy(), pr.copy()
        if self.continuous_reference:
            # Online ACT chunks can change at every query. Bound reference velocity
            # and acceleration before computing g/Kp + Kd*dq/Kp feedforward.
            ep = pp-self.filtered_p
            er = Rotation.from_matrix(pr@self.filtered_r.T).as_rotvec()
            wanted_p = bounded(8*ep,.08)
            wanted_r = bounded(8*er,np.deg2rad(30))
            self.linear_velocity += bounded(wanted_p-self.linear_velocity,.3*dt)
            self.angular_velocity += bounded(wanted_r-self.angular_velocity,np.deg2rad(90)*dt)
            self.filtered_p += self.linear_velocity*dt
            self.filtered_r = Rotation.from_rotvec(self.angular_velocity*dt).as_matrix()@self.filtered_r
            pp, pr = self.filtered_p.copy(), self.filtered_r.copy()
        issued_p, issued_r, _ = self.engine.step(pp+fp, fr@pr, dt, weight=ramp)
        cp, cr = issued_p-pp, issued_r@pr.T
        if (np.linalg.norm(cp) > self.profile["maximum_total_translation_m"] or
                Rotation.from_matrix(cr).magnitude() > np.deg2rad(self.profile["maximum_total_rotation_deg"])):
            raise RuntimeError("Combined correction exceeds 40mm / 5deg")
        cp, cr, limited = limit_correction(self.last_p, self.last_r, cp, cr, dt,
            self.profile["maximum_correction_rate_m_s"], self.profile["maximum_correction_rate_deg_s"])
        self.engine.qissued = self.engine.solve(pp+cp, cr@pr, self.engine.qissued)
        self.last_p, self.last_r, self.last_t = cp, cr, t
        return self.engine.qissued.copy(), {"correction_m": cp.tolist(),
            "feedback_m": fp.tolist(), "correction_limited": limited,
            "requested_preview_p":requested_p.tolist(), "conditioned_preview_p":pp.tolist(),
            "reference_conditioning_error_m":float(np.linalg.norm(pp-requested_p))}
