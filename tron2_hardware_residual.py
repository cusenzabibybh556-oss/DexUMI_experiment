"""Offline/runtime representation of a hardware-identified ServoP residual model.

The model never opens a robot connection.  It predicts a bounded correction on
top of the existing MuJoCo pose-reference feed-forward from quantities that are
already available inside :class:`tron2_pose_reference.PoseReferenceModel`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "tron2-servop-hardware-residual-v1"
BASE_FEATURE_NAMES = [
    *(f"model_shift_{axis}_m" for axis in "xyz"),
    *(f"model_rotation_{axis}_rad" for axis in "xyz"),
    *(f"reference_joint_velocity_{i}_rad_s" for i in range(7)),
    *(f"current_total_translation_{axis}_m" for axis in "xyz"),
    *(f"current_total_rotation_{axis}_rad" for axis in "xyz"),
    *(f"feedback_residual_translation_{axis}_m" for axis in "xyz"),
    *(f"feedback_residual_rotation_{axis}_rad" for axis in "xyz"),
]
FEATURE_NAMES = [*(f"linear:{name}" for name in BASE_FEATURE_NAMES),
                 *(f"square:{name}" for name in BASE_FEATURE_NAMES),
                 *(f"absolute:{name}" for name in BASE_FEATURE_NAMES)]
OUTPUT_NAMES = [*(f"additional_translation_{axis}_m" for axis in "xyz"),
                *(f"additional_rotation_{axis}_rad" for axis in "xyz")]


def feature_vector(model_shift, model_rotation, joint_velocity, current_translation,
                   current_rotation, feedback_translation, feedback_rotation):
    """Return the fixed feature order used by fitting, replay, and deployment."""
    arrays = [np.asarray(model_shift, float), np.asarray(model_rotation, float),
              np.asarray(joint_velocity, float), np.asarray(current_translation, float),
              np.asarray(current_rotation, float), np.asarray(feedback_translation, float),
              np.asarray(feedback_rotation, float)]
    expected = [(3,), (3,), (7,), (3,), (3,), (3,), (3,)]
    if [value.shape for value in arrays] != expected or not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Invalid hardware residual feature values")
    base = np.concatenate(arrays)
    return np.r_[base, base*base, np.abs(base)]


class HardwareResidualModel:
    """Small, auditable robust-ridge model fitted only from real hardware logs."""

    def __init__(self, payload, *, path=None, sha256=None):
        if payload.get("schema") != SCHEMA:
            raise ValueError("Unexpected ServoP hardware model schema")
        if payload.get("feature_names") != FEATURE_NAMES or payload.get("output_names") != OUTPUT_NAMES:
            raise ValueError("ServoP hardware model feature/output contract changed")
        self.mean = np.asarray(payload["feature_mean"], float)
        self.scale = np.asarray(payload["feature_scale"], float)
        self.coefficients = np.asarray(payload["adjustment_coefficients"], float)
        self.intercept = np.asarray(payload["adjustment_intercept"], float)
        n = len(FEATURE_NAMES)
        if (self.mean.shape != (n,) or self.scale.shape != (n,)
                or self.coefficients.shape != (6, n) or self.intercept.shape != (6,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or not np.isfinite(self.coefficients).all() or not np.isfinite(self.intercept).all()
                or np.any(self.scale <= 0)):
            raise ValueError("Invalid ServoP hardware model arrays")
        limits = payload["prediction_limits"]
        self.max_translation = float(limits["maximum_total_translation_m"])
        self.max_rotation = np.deg2rad(float(limits["maximum_total_rotation_deg"]))
        if not 0 < self.max_translation <= .04 or not 0 < self.max_rotation <= np.deg2rad(5):
            raise ValueError("ServoP hardware model widens the established correction limits")
        gate = payload["feedback_gate"]
        self.gate_translation = float(gate["full_scale_translation_m"])
        self.gate_rotation = np.deg2rad(float(gate["full_scale_rotation_deg"]))
        if not 0 < self.gate_translation <= .01 or not 0 < self.gate_rotation <= np.deg2rad(2):
            raise ValueError("Invalid ServoP hardware residual gate")
        self.payload = payload
        self.path = None if path is None else str(Path(path).resolve())
        self.sha256 = sha256

    @classmethod
    def load(cls, path):
        path = Path(path)
        raw = path.read_bytes()
        return cls(json.loads(raw), path=path, sha256=hashlib.sha256(raw).hexdigest())

    def predict_adjustment(self, features):
        x = np.asarray(features, float)
        if x.shape[-1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("Invalid ServoP hardware model input")
        prediction = (x-self.mean)/self.scale @ self.coefficients.T + self.intercept
        # Linear base features 19:22 and 22:25 are the measured-feedback
        # translation/rotation residual.  An ideal or absent residual must never
        # cause this learned layer to invent a correction.
        gate = np.maximum(np.linalg.norm(x[..., 19:22], axis=-1)/self.gate_translation,
                          np.linalg.norm(x[..., 22:25], axis=-1)/self.gate_rotation)
        gate = np.clip(gate, 0, 1)
        return prediction*gate[..., None] if np.ndim(gate) else prediction*gate

    def predict_total(self, model_shift, model_rotation, joint_velocity,
                      current_translation, current_rotation,
                      feedback_translation, feedback_rotation):
        x = feature_vector(model_shift, model_rotation, joint_velocity,
                           current_translation, current_rotation,
                           feedback_translation, feedback_rotation)
        adjustment = self.predict_adjustment(x)
        translation = np.asarray(current_translation, float)+adjustment[:3]
        rotation = (Rotation.from_rotvec(adjustment[3:]).as_matrix()
                    @ Rotation.from_rotvec(current_rotation).as_matrix())
        translation_norm = float(np.linalg.norm(translation))
        if translation_norm > self.max_translation:
            translation *= self.max_translation/translation_norm
        vector = Rotation.from_matrix(rotation).as_rotvec()
        rotation_norm = float(np.linalg.norm(vector))
        if rotation_norm > self.max_rotation:
            rotation = Rotation.from_rotvec(vector*(self.max_rotation/rotation_norm)).as_matrix()
        return translation, rotation, adjustment

