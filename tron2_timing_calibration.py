"""Offline SE(3) timing calibration from a separate excitation trajectory.

Fits effective command-to-measured-pose lag, including any uncorrected sensor
age. A fit alone cannot separate actuation delay from clock or feedback age.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class CalibrationReference:
    duration: float = 20.

    @property
    def metadata(self):
        return dict(episode='synthetic_6D_timing_excitation', duration_s=self.duration,
                    definition='Independent small multi-axis sinusoidal translation and rotation, smooth envelope')

    def target(self, t, anchor, initial_rotation):
        t = float(np.clip(t, 0, self.duration))
        edge = min(t, self.duration-t, 1.)
        envelope = edge*edge*(3-2*edge)
        p = np.array([.015*np.sin(2*np.pi*.17*t), .012*np.sin(2*np.pi*.23*t), .015*np.sin(2*np.pi*.31*t)])*envelope
        rv = np.deg2rad([3*np.sin(2*np.pi*.19*t), 3*np.sin(2*np.pi*.27*t), 2*np.sin(2*np.pi*.13*t)])*envelope
        return np.asarray(anchor)+p, Rotation.from_rotvec(rv).as_matrix()@initial_rotation


def fit_effective_lag(command_t, command_p, command_r, measured_t, measured_p, measured_r,
                      *, fit_window=(2., 12.), validation_window=(12., 18.), max_lag_s=.3):
    ct, cp, cr = np.asarray(command_t, float), np.asarray(command_p, float), np.asarray(command_r, float)
    mt, mp, mr = np.asarray(measured_t, float), np.asarray(measured_p, float), np.asarray(measured_r, float)
    if (ct.ndim != 1 or mt.ndim != 1 or cp.shape != (len(ct), 3) or cr.shape != (len(ct), 3, 3)
            or mp.shape != (len(mt), 3) or mr.shape != (len(mt), 3, 3)
            or len(ct) < 5 or len(mt) < 5 or not np.all(np.diff(ct) > 0) or not np.all(np.diff(mt) > 0)
            or not all(np.isfinite(a).all() for a in (ct, cp, cr, mt, mp, mr))
            or not 0 < max_lag_s <= .3):
        raise ValueError('Invalid calibration pose/time arrays')
    for rr in (cr, mr):
        if not np.allclose(np.swapaxes(rr, 1, 2)@rr, np.eye(3), atol=1e-6, rtol=0) or not np.allclose(np.linalg.det(rr), 1, atol=1e-6):
            raise ValueError('Calibration rotations must be SO(3)')
    interp_r = Slerp(ct, Rotation.from_matrix(cr))
    def residual(lag, window):
        keep = (mt >= window[0]) & (mt <= window[1]) & (mt-lag >= ct[0]) & (mt-lag <= ct[-1])
        if keep.sum() < 30:
            raise ValueError('Too few independent calibration measurements')
        query = mt[keep]-lag
        target_p = np.column_stack([np.interp(query, ct, cp[:, j]) for j in range(3)])
        target_r = interp_r(query)
        rot_error = (Rotation.from_matrix(mr[keep])*target_r.inv()).as_rotvec()
        return np.column_stack((mp[keep]-target_p, .25*rot_error))
    lags = np.linspace(0, max_lag_s, round(max_lag_s*1000)+1)
    scores = []
    for lag in lags:
        e = residual(lag, fit_window)
        e -= np.median(e, axis=0)
        scores.append(float(np.sqrt(np.mean(np.sum(e*e, axis=1)))))
    best = int(np.argmin(scores))
    def score(lag, window):
        e = residual(lag, window)
        bias = np.median(e, axis=0)
        centered = e-bias
        return dict(samples=len(e), centered_position_RMS_mm=float(np.sqrt(np.mean(np.sum(centered[:, :3]**2, axis=1)))*1000),
                    centered_rotation_RMS_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum((centered[:, 3:]/.25)**2, axis=1))))),
                    translation_bias_mm=(bias[:3]*1000).tolist())
    profile = dict(schema='tron2-effective-pose-lag-v1', simulation_only=True,
        effective_preview_s=float(lags[best]), lag_grid_resolution_s=.001,
        fit_window_s=list(fit_window), validation_window_s=list(validation_window),
        fit_before=score(0, fit_window), fit_after=score(lags[best], fit_window),
        validation_before=score(0, validation_window), validation_after=score(lags[best], validation_window),
        boundary_optimum=best in (0, len(lags)-1),
        semantics='Effective command-to-pose lag. Includes uncorrected measurement age or clock offset; not proof of firmware delay.',
        pose_bias_is_not_applied_to_task_targets=True)
    return profile, dict(lag_s=lags, centered_pose_score_m=np.array(scores))
