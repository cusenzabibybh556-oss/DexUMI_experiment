"""Same spatial-delta convention as calibrated Tracker replay, never body deltas."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def matrix_to_rotation6d(matrix):
    r = np.asarray(matrix)
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def rotation6d_to_matrix(value):
    x = np.asarray(value, dtype=np.float64)
    a, b = x[..., :3], x[..., 3:6]
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(na < 1e-8):
        raise ValueError("Degenerate rotation6D first column")
    a = a / na
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    nb = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(nb < 1e-8):
        raise ValueError("Degenerate rotation6D second column")
    b = b / nb
    return np.stack([a, b, np.cross(a, b)], axis=-1)


def decode_quaternions(values, source_format):
    q = np.asarray(values, dtype=np.float64).copy()
    if source_format == "pyvut-legacy":
        q = q[..., [1, 0, 3, 2]]
    elif source_format != "wxyz":
        raise ValueError("Unknown quaternion source format")
    return Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()


def interpolate(times, values, targets, max_gap, rotations=False):
    """Return interpolation plus validity, never extrapolate across a long gap."""
    t = np.asarray(times, float)
    at = np.asarray(targets, float)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Interpolation needs >=2 strictly increasing times")
    right = np.clip(np.searchsorted(t, at, side="left"), 1, len(t) - 1)
    left = right - 1
    exact = np.isclose(at, t[left], rtol=0, atol=1e-8) | np.isclose(at, t[right], rtol=0, atol=1e-8)
    valid = (at >= t[0] - 1e-8) & (at <= t[-1] + 1e-8) & ((t[right] - t[left] <= max_gap) | exact)
    clipped = np.clip(at, t[0], t[-1])
    if rotations:
        result = Slerp(t, Rotation.from_matrix(values))(clipped).as_matrix()
    else:
        result = np.column_stack([np.interp(clipped, t, values[:, j]) for j in range(values.shape[1])])
    return result, valid


def hold_commands(times, values, targets, record_end, max_age_s=0.):
    """Causal zero-order hold of sent targets; never mix commands by interpolation.

    A zero age limit preserves the collector's stateful latest-before semantics:
    absence of a new command means the old target remains active.
    """
    t = np.asarray(times, float)
    at = np.asarray(targets, float)
    if not len(t) or np.any(np.diff(t) <= 0):
        raise ValueError("Commands need strictly increasing event timestamps")
    indices = np.searchsorted(t, at + 1e-9, side="right") - 1
    valid = (indices >= 0) & (at <= record_end + 1e-9)
    indices = np.clip(indices, 0, len(t) - 1)
    if max_age_s > 0:
        valid &= at - t[indices] <= max_age_s + 1e-9
    return np.asarray(values)[indices], valid
