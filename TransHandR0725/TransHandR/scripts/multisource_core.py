"""Synchronization and HDF5 primitives for the quality-first recorder."""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class TimedSample:
    value: Any
    source_timestamp: float
    unified_timestamp_ns: int
    host_receive_timestamp_ns: int
    valid: bool = True
    sequence: int = -1
    stream_index: int = -1
    stream_epoch: int = -1


class SampleBuffer:
    """Small thread-safe sample buffer retaining a bounded time window."""

    def __init__(self, maxlen: int = 512, retention_ns: int = 1_000_000_000):
        self._samples: Deque[TimedSample] = deque(maxlen=maxlen)
        self.retention_ns = int(retention_ns)
        self._lock = threading.Lock()

    def add(self, sample: TimedSample) -> None:
        with self._lock:
            self._samples.append(sample)
            newest = max(item.unified_timestamp_ns for item in self._samples)
            cutoff = newest - self.retention_ns
            if self._samples and any(
                self._samples[index].unified_timestamp_ns > self._samples[index + 1].unified_timestamp_ns
                for index in range(len(self._samples) - 1)
            ):
                self._samples = deque(
                    sorted(self._samples, key=lambda item: item.unified_timestamp_ns),
                    maxlen=self._samples.maxlen,
                )
            while self._samples and self._samples[0].unified_timestamp_ns < cutoff:
                self._samples.popleft()

    def snapshot(self) -> Tuple[TimedSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def latest(self) -> Optional[TimedSample]:
        with self._lock:
            return max(self._samples, key=lambda item: item.unified_timestamp_ns) if self._samples else None


class ClockMapper:
    """Affine source-clock to host CLOCK_MONOTONIC mapper.

    V1 estimates offset from the first observation and accepts a configured
    source unit.  The explicit scale/offset are exposed so a calibrated drift
    estimate can be supplied later without changing the recorder.
    """

    UNIT_SCALE = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}

    def __init__(self, unit: str = "ns", scale: float = 1.0, offset_ns: Optional[float] = None):
        if unit not in self.UNIT_SCALE:
            raise ValueError(f"Unknown clock unit {unit}")
        self.source_to_ns = self.UNIT_SCALE[unit]
        self.scale = float(scale)
        self.offset_ns = offset_ns

    def map(self, source_timestamp: float, host_receive_timestamp_ns: int) -> int:
        source_ns = float(source_timestamp) * self.source_to_ns
        if self.offset_ns is None:
            self.offset_ns = host_receive_timestamp_ns - self.scale * source_ns
        return int(round(self.scale * source_ns + self.offset_ns))


def visionpro_hand_points(finger_matrices: np.ndarray, wrist_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return world-frame and true wrist-local XYZ points from AVP transforms."""
    fingers = np.asarray(finger_matrices, np.float64)
    wrist = np.asarray(wrist_matrix, np.float64)
    if fingers.shape != (25, 4, 4) or wrist.shape != (4, 4):
        raise ValueError(f"invalid Vision Pro transforms: fingers={fingers.shape}, wrist={wrist.shape}")

    def xyz(matrices: np.ndarray) -> np.ndarray:
        return np.stack((-matrices[:, 1, 3], matrices[:, 2, 3], -matrices[:, 0, 3]), axis=1)

    local = xyz(fingers)
    world = xyz(wrist @ fingers)
    return world.astype(np.float32), (local - local[0]).astype(np.float32)


def nearest(samples: Sequence[TimedSample], timestamp_ns: int, max_gap_ms: float) -> Tuple[Optional[TimedSample], float]:
    valid = [s for s in samples if s.valid]
    if not valid:
        return None, math.nan
    item = min(valid, key=lambda s: abs(s.unified_timestamp_ns - timestamp_ns))
    dt_ms = (item.unified_timestamp_ns - timestamp_ns) / 1e6
    return (item if abs(dt_ms) <= max_gap_ms else None), dt_ms


def bracket(samples: Sequence[TimedSample], timestamp_ns: int) -> Tuple[Optional[TimedSample], Optional[TimedSample]]:
    valid = sorted((s for s in samples if s.valid), key=lambda s: s.unified_timestamp_ns)
    before = next((s for s in reversed(valid) if s.unified_timestamp_ns <= timestamp_ns), None)
    after = next((s for s in valid if s.unified_timestamp_ns >= timestamp_ns), None)
    return before, after


def interpolate_vector(samples: Sequence[TimedSample], timestamp_ns: int, max_gap_ms: float) -> Tuple[Optional[TimedSample], float]:
    left, right = bracket(samples, timestamp_ns)
    if left is None or right is None:
        return None, math.nan
    left_gap = (timestamp_ns - left.unified_timestamp_ns) / 1e6
    right_gap = (right.unified_timestamp_ns - timestamp_ns) / 1e6
    gap = max(left_gap, right_gap)
    if left_gap > max_gap_ms or right_gap > max_gap_ms:
        return None, gap
    span = right.unified_timestamp_ns - left.unified_timestamp_ns
    alpha = 0.0 if span == 0 else (timestamp_ns - left.unified_timestamp_ns) / span
    value = (1.0 - alpha) * np.asarray(left.value, np.float32) + alpha * np.asarray(right.value, np.float32)
    source_timestamp = (1.0 - alpha) * left.source_timestamp + alpha * right.source_timestamp
    return TimedSample(value, source_timestamp, timestamp_ns, max(left.host_receive_timestamp_ns, right.host_receive_timestamp_ns)), gap


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, np.float64); q1 = np.asarray(q1, np.float64)
    q0 /= np.linalg.norm(q0); q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return (result / np.linalg.norm(result)).astype(np.float32)
    theta = math.acos(dot)
    result = math.sin((1.0 - alpha) * theta) / math.sin(theta) * q0 + math.sin(alpha * theta) / math.sin(theta) * q1
    return result.astype(np.float32)


def interpolate_pose(samples: Sequence[TimedSample], timestamp_ns: int, max_gap_ms: float) -> Tuple[Optional[TimedSample], float]:
    left, right = bracket(samples, timestamp_ns)
    if left is None or right is None:
        return None, math.nan
    lg = (timestamp_ns - left.unified_timestamp_ns) / 1e6
    rg = (right.unified_timestamp_ns - timestamp_ns) / 1e6
    if lg > max_gap_ms or rg > max_gap_ms:
        return None, max(lg, rg)
    span = right.unified_timestamp_ns - left.unified_timestamp_ns
    alpha = 0.0 if span == 0 else (timestamp_ns - left.unified_timestamp_ns) / span
    lv, rv = left.value, right.value
    value = {
        "pos": ((1 - alpha) * np.asarray(lv["pos"]) + alpha * np.asarray(rv["pos"])).astype(np.float32),
        "quat": slerp(np.asarray(lv["quat"]), np.asarray(rv["quat"]), alpha),
    }
    source_timestamp = (1.0 - alpha) * left.source_timestamp + alpha * right.source_timestamp
    return TimedSample(value, source_timestamp, timestamp_ns, max(left.host_receive_timestamp_ns, right.host_receive_timestamp_ns)), max(lg, rg)


def latest_before(samples: Sequence[TimedSample], timestamp_ns: int, max_age_ms: float) -> Tuple[Optional[TimedSample], float]:
    """Return the newest causal sample.

    A non-positive ``max_age_ms`` disables expiry.  This is useful for
    stateful event streams such as the last command successfully applied to a
    robot: silence means the command is unchanged, not that it became invalid.
    """
    candidates = [s for s in samples if s.valid and s.unified_timestamp_ns <= timestamp_ns]
    if not candidates:
        return None, math.nan
    item = max(candidates, key=lambda s: s.unified_timestamp_ns)
    age_ms = (timestamp_ns - item.unified_timestamp_ns) / 1e6
    return (item if max_age_ms <= 0 or age_ms <= max_age_ms else None), age_ms


def tactile_at_master(
    episode: h5py.File, master_index: int, max_age_ms: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], float, int]:
    """Resolve tactile data for one D405 frame across v2 and v3 episodes."""
    schema = str(episode.attrs.get("schema_version", ""))
    master = episode["timestamps/master_timestamp_ns"]
    if master_index < 0 or master_index >= len(master):
        raise IndexError(f"master index out of range: {master_index}")
    if schema.startswith("quality-first-v3"):
        timestamps = episode["dexterous_hand/tactile_stream/timestamp_ns"][:]
        if not len(timestamps):
            return None, math.nan, -1
        target = int(master[master_index])
        index = int(np.searchsorted(timestamps, target, side="right") - 1)
        if index < 0:
            return None, math.nan, -1
        age_ms = (target - int(timestamps[index])) / 1e6
        limit = (float(episode.attrs.get("tactile_max_gap_ms", 120.0))
                 if max_age_ms is None else float(max_age_ms))
        if age_ms < 0 or age_ms > limit:
            return None, age_ms, -1
        return np.asarray(
            episode["dexterous_hand/tactile_stream/matrix"][index]
        ), age_ms, index
    if "dexterous_hand/tactile/matrix" not in episode:
        return None, math.nan, -1
    valid = bool(episode["dexterous_hand/tactile/valid"][master_index])
    if not valid:
        return None, math.nan, -1
    age_ms = float(episode["sync/tactile_dt_ms"][master_index])
    if max_age_ms is not None and abs(age_ms) > float(max_age_ms):
        return None, age_ms, -1
    return np.asarray(
        episode["dexterous_hand/tactile/matrix"][master_index]
    ), age_ms, master_index


def palm_at_master(
    episode: h5py.File, master_index: int, max_age_ms: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], float, int]:
    """Resolve one B1 palm frame; a custom age limit reselects from the raw stream."""
    master = episode["timestamps/master_timestamp_ns"]
    if master_index < 0 or master_index >= len(master):
        raise IndexError(f"master index out of range: {master_index}")
    if "dexterous_hand/palm_stream/timestamp_ns" not in episode:
        return None, math.nan, -1
    timestamps = episode["dexterous_hand/palm_stream/timestamp_ns"][:]
    if max_age_ms is None:
        index = int(episode["dexterous_hand/palm/index"][master_index])
    else:
        index = int(np.searchsorted(timestamps, int(master[master_index]), side="right") - 1)
    if index < 0 or index >= len(timestamps):
        return None, math.nan, -1
    age_ms = (int(master[master_index]) - int(timestamps[index])) / 1e6
    limit = (float(episode.attrs.get("tactile_max_gap_ms", 120.0))
             if max_age_ms is None else float(max_age_ms))
    if age_ms < 0 or age_ms > limit:
        return None, age_ms, -1
    return np.asarray(episode["dexterous_hand/palm_stream/matrix"][index]), age_ms, index


class EpisodeWriter:
    """Append-only HDF5 writer with independent D405 and tactile axes."""

    def __init__(self, path: Path, config: Any, camera_meta: Dict[str, Any], episode_idx: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.episode_idx = int(episode_idx)
        self.f = h5py.File(self.path, "w")
        self.count = 0
        self.tactile_count = 0
        self.palm_count = 0
        self.force_count = 0
        self._build(camera_meta, episode_idx)

    def _dataset(self, name: str, tail: Tuple[int, ...], dtype: str, image: bool = False) -> h5py.Dataset:
        chunks = (max(1, self.config.hdf5_chunk_frames),) + tail
        kwargs = {"maxshape": (None,) + tail, "chunks": chunks, "dtype": dtype}
        if image and self.config.compression:
            kwargs["compression"] = self.config.compression
        return self.f.create_dataset(name, shape=(0,) + tail, **kwargs)

    def _build(self, meta: Dict[str, Any], episode_idx: int) -> None:
        c, f = self.config, self.f
        head_meta = meta.get("d435i", {})
        attrs = {
            "schema_version": "quality-first-v3.2",
            "episode_idx": episode_idx,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task_name": c.task_name,
            "master_source": "d405",
            "camera_fps": c.camera_fps,
            "actual_capture_hz": np.nan,
            "hand_type": c.hand_type,
            "control_source": c.control_source,
            "target_hand_model": c.target_hand_model,
            "hand_side": c.hand_side,
            "telemetry_protocol_version": c.telemetry_protocol_version,
            "action_semantics": "executed_action",
            "vision_hand_side": c.vision_hand_side,
            "d405_serial": meta.get("serial", c.d405_serial),
            "image_height": c.image_height,
            "image_width": c.image_width,
            "quat_order": "[w, x, y, z]",
            "transform_convention": "p_A = T_A_B @ p_B",
            "right_hand_relative_frame": "vision_wrist_local",
            "right_hand_max_gap_ms": c.right_hand_max_gap_ms,
            "hand_state_max_gap_ms": c.hand_state_max_gap_ms,
            "tracker_max_gap_ms": c.tracker_max_gap_ms,
            "action_max_age_ms": c.action_max_age_ms,
            "operator_max_gap_ms": c.operator_max_gap_ms,
            "tactile_max_gap_ms": c.tactile_max_gap_ms,
            "tactile_contact_threshold": c.tactile_contact_threshold,
            "enable_vive_tracker": c.enable_vive_tracker,
            "max_episode_duration_s": c.max_episode_duration_s,
            "max_episode_size_gb": c.max_episode_size_gb,
            "depth_scale": meta.get("depth_scale", np.nan),
            "rgb_intrinsics": json.dumps(meta.get("rgb_intrinsics", {})),
            "depth_intrinsics": json.dumps(meta.get("depth_intrinsics", {})),
            "depth_aligned_to_color": c.depth_aligned_to_color,
            "enable_head_camera": c.enable_head_camera,
            "d435i_serial": head_meta.get("serial", c.head_camera_serial),
            "d435i_camera_fps": c.head_camera_fps,
            "d435i_image_height": c.head_image_height,
            "d435i_image_width": c.head_image_width,
            "d435i_depth_scale": head_meta.get("depth_scale", np.nan),
            "d435i_rgb_intrinsics": json.dumps(head_meta.get("rgb_intrinsics", {})),
            "d435i_depth_intrinsics": json.dumps(head_meta.get("depth_intrinsics", {})),
            "d435i_depth_aligned_to_color": c.head_depth_aligned_to_color,
            "d435i_max_gap_ms": c.head_camera_max_gap_ms,
            "selected_channels": json.dumps(c.selected_channels),
        }
        for key, value in attrs.items(): f.attrs[key] = value
        h, w = c.image_height, c.image_width
        js, ja = int(c.hand_spec["state_dim"]), int(c.hand_spec["action_dim"])
        self._dataset("timestamps/master_timestamp_ns", (), "int64")
        self._dataset("observations/images/d405_rgb", (h, w, 3), "uint8", True)
        self._dataset("observations/depth/d405_depth", (h, w), "uint16", True)
        if c.enable_head_camera:
            self._dataset("observations/images/d435i_rgb", (c.head_image_height, c.head_image_width, 3), "uint8", True)
            self._dataset("observations/depth/d435i_depth", (c.head_image_height, c.head_image_width), "uint16", True)
        for name, tail, dtype in [
            ("observations/right_hand/world_keypoints", (25, 3), "float32"),
            ("observations/right_hand/relative_keypoints", (25, 3), "float32"),
            ("observations/right_hand/valid", (), "bool"),
            ("observations/right_hand/source_timestamp", (), "float64"),
            ("observations/right_hand/unified_timestamp_ns", (), "int64"),
            ("observations/right_hand/host_receive_timestamp_ns", (), "int64"),
            ("operator/exoskeleton/joint_positions", (21,), "float32"),
            ("operator/exoskeleton/valid", (), "bool"),
            ("operator/exoskeleton/source_timestamp_ns", (), "int64"),
            ("operator/exoskeleton/unified_timestamp_ns", (), "int64"),
            ("operator/exoskeleton/host_receive_timestamp_ns", (), "int64"),
            ("dexterous_hand/joint_positions", (js,), "float32"),
            ("dexterous_hand/state_valid", (), "bool"),
            ("dexterous_hand/source_timestamp", (), "float64"),
            ("dexterous_hand/unified_timestamp_ns", (), "int64"),
            ("dexterous_hand/host_receive_timestamp_ns", (), "int64"),
            ("action/command", (ja,), "float32"), ("action/valid", (), "bool"),
            ("action/command_timestamp_ns", (), "int64"),
            ("dexterous_hand/tactile/index", (), "int64"),
            ("dexterous_hand/tactile/age_ms", (), "float32"),
            ("dexterous_hand/tactile/valid", (), "bool"),
            ("dexterous_hand/palm/index", (), "int64"),
            ("dexterous_hand/palm/age_ms", (), "float32"),
            ("dexterous_hand/palm/valid", (), "bool"),
            ("vive_tracker/pos", (1, 3), "float32"), ("vive_tracker/quat", (1, 4), "float32"),
            ("vive_tracker/valid", (1,), "bool"), ("vive_tracker/source_timestamp", (1,), "float64"),
            ("vive_tracker/unified_timestamp_ns", (1,), "int64"),
            ("vive_tracker/host_receive_timestamp_ns", (1,), "int64"),
            ("camera/d405/frame_number", (), "int64"), ("camera/d405/device_timestamp", (), "float64"),
            ("camera/d405/host_receive_timestamp_ns", (), "int64"), ("camera/d405/valid", (), "bool"),
            ("sync/right_hand_dt_ms", (), "float32"), ("sync/hand_state_interpolation_gap_ms", (), "float32"),
            ("sync/operator_interpolation_gap_ms", (), "float32"),
            ("sync/tracker_interpolation_gap_ms", (), "float32"), ("sync/action_age_ms", (), "float32"),
        ]: self._dataset(name, tail, dtype)
        self._dataset("dexterous_hand/tactile_stream/matrix", (5, 12, 6), "int16")
        self._dataset("dexterous_hand/tactile_stream/timestamp_ns", (), "int64")
        self._dataset("dexterous_hand/tactile_stream/sequence", (), "int64")
        self._dataset("dexterous_hand/palm_stream/matrix", (14, 15), "int16")
        self._dataset("dexterous_hand/palm_stream/timestamp_ns", (), "int64")
        self._dataset("dexterous_hand/palm_stream/sequence", (), "int64")
        self._dataset("dexterous_hand/force_stream/value", (6,), "int16")
        self._dataset("dexterous_hand/force_stream/timestamp_ns", (), "int64")
        self._dataset("dexterous_hand/force_stream/sequence", (), "int64")
        if c.enable_head_camera:
            for name, tail, dtype in [
                ("camera/d435i/frame_number", (), "int64"), ("camera/d435i/device_timestamp", (), "float64"),
                ("camera/d435i/unified_timestamp_ns", (), "int64"), ("camera/d435i/host_receive_timestamp_ns", (), "int64"),
                ("camera/d435i/valid", (), "bool"), ("sync/d435i_dt_ms", (), "float32"),
            ]: self._dataset(name, tail, dtype)
        rh = f["observations/right_hand"]
        rh.attrs["right_hand_coordinate_rule"] = "x=-mat[1][3], y=mat[2][3], z=-mat[0][3]"
        rh.attrs["source_hand_side"] = c.vision_hand_side
        rh.attrs["world_transform_rule"] = "world_joint_transform = source_wrist @ source_finger_local_transform"
        rh.attrs["relative_keypoints_rule"] = "world_keypoints - world_keypoints[wrist_index]"
        rh.attrs["keypoint_names"] = json.dumps([
            "wrist",
            "thumb_knuckle", "thumb_intermediate_base", "thumb_intermediate_tip", "thumb_tip",
            "index_metacarpal", "index_knuckle", "index_intermediate_base", "index_intermediate_tip", "index_tip",
            "middle_metacarpal", "middle_knuckle", "middle_intermediate_base", "middle_intermediate_tip", "middle_tip",
            "ring_metacarpal", "ring_knuckle", "ring_intermediate_base", "ring_intermediate_tip", "ring_tip",
            "little_metacarpal", "little_knuckle", "little_intermediate_base", "little_intermediate_tip", "little_tip",
        ])
        rh.attrs["wrist_index"] = 0; rh.attrs["coordinate_unit"] = "meter"
        hand = f["dexterous_hand"]
        hand.attrs["state_joint_names"] = json.dumps(c.state_joint_names)
        hand.attrs["state_unit"] = c.hand_spec["state_unit"]
        hand.attrs["feedback_semantics"] = "real_device_feedback"
        action = f["action"]
        action.attrs["action_names"] = json.dumps(c.action_names)
        action.attrs["action_unit"] = c.hand_spec["action_unit"]
        action.attrs["action_semantics"] = "executed_joint_position_target"
        operator = f["operator/exoskeleton"]
        operator.attrs["joint_names"] = json.dumps([f"glove_joint_{i + 1}" for i in range(21)])
        operator.attrs["source"] = "linkerforce"
        tactile = f["dexterous_hand/tactile"]
        tactile.attrs["finger_names"] = json.dumps(["thumb", "index", "middle", "ring", "little"])
        tactile.attrs["matrix_shape"] = "[5, 12, 6]"
        tactile.attrs["contact_threshold"] = c.tactile_contact_threshold
        tactile.attrs["alignment"] = "latest-before"
        tactile.attrs["matrix_storage"] = "dexterous_hand/tactile_stream/matrix"
        tactile_stream = f["dexterous_hand/tactile_stream"]
        tactile_stream.attrs["time_axis"] = "asynchronous"
        tactile_stream.attrs["timestamp_clock"] = "host_monotonic"
        palm = f["dexterous_hand/palm"]
        palm.attrs["alignment"] = "latest-before"
        palm.attrs["matrix_storage"] = "dexterous_hand/palm_stream/matrix"
        palm.attrs["max_age_ms"] = c.tactile_max_gap_ms
        palm_stream = f["dexterous_hand/palm_stream"]
        palm_stream.attrs["time_axis"] = "asynchronous"
        palm_stream.attrs["timestamp_clock"] = "host_monotonic"
        palm_stream.attrs["matrix_shape"] = "[14, 15]"
        force_stream = f["dexterous_hand/force_stream"]
        force_stream.attrs["time_axis"] = "asynchronous"
        force_stream.attrs["timestamp_clock"] = "host_monotonic"
        force_stream.attrs["unit"] = "device_reported_g"
        force_stream.attrs["joint_names"] = json.dumps(
            ["little", "ring", "middle", "index", "thumb_bend", "thumb_rotate"])
        tracker_group = f["vive_tracker"]
        tracker_group.attrs["position_unit"] = "meter"
        tracker_group.attrs["rotation_representation"] = "unit_quaternion"
        tracker_group.attrs["quat_order"] = "[w, x, y, z]"
        cal = f.require_group("calibration/d405")
        cal.create_dataset("T_wrist_camera", data=np.eye(4, dtype=np.float32))
        cal.create_dataset("T_tracker_camera", data=np.eye(4, dtype=np.float32))
        cal.create_dataset("calibrated", data=np.bool_(False))
        if c.enable_head_camera:
            head_cal = f.require_group("calibration/d435i")
            head_cal.create_dataset("T_d405_d435i", data=np.eye(4, dtype=np.float32))
            head_cal.create_dataset("calibrated", data=np.bool_(False))

    def append(self, row: Dict[str, Any]) -> None:
        i = self.count
        for name, value in row.items():
            ds = self.f[name]; ds.resize((i + 1,) + ds.shape[1:]); ds[i] = value
        self.count += 1
        if self.count % 30 == 0: self.f.flush()

    def append_tactile(self, matrix: Any, timestamp_ns: int, sequence: int) -> int:
        """Append one real tactile event and return its episode-local index."""
        values = np.asarray(matrix)
        if values.shape != (5, 12, 6):
            raise ValueError(f"tactile matrix must be 5x12x6, got {values.shape}")
        if not np.all(np.isfinite(values)) or not np.allclose(values, np.rint(values)):
            raise ValueError("tactile matrix must contain finite integer values")
        rounded = np.rint(values)
        limits = np.iinfo(np.int16)
        if np.any(rounded < limits.min) or np.any(rounded > limits.max):
            raise ValueError("tactile matrix exceeds int16 range")
        timestamp_ns = int(timestamp_ns)
        timestamps = self.f["dexterous_hand/tactile_stream/timestamp_ns"]
        if self.tactile_count and timestamp_ns <= int(timestamps[self.tactile_count - 1]):
            raise ValueError("out-of-order tactile timestamp")
        index = self.tactile_count
        entries = {
            "dexterous_hand/tactile_stream/matrix": rounded.astype(np.int16),
            "dexterous_hand/tactile_stream/timestamp_ns": timestamp_ns,
            "dexterous_hand/tactile_stream/sequence": int(sequence),
        }
        for name, value in entries.items():
            dataset = self.f[name]
            dataset.resize((index + 1,) + dataset.shape[1:])
            dataset[index] = value
        self.tactile_count += 1
        return index

    def append_palm(self, matrix: Any, timestamp_ns: int, sequence: int) -> int:
        values = np.asarray(matrix)
        if values.shape != (14, 15) or not np.all(np.isfinite(values)):
            raise ValueError("palm matrix must be finite 14x15 values")
        if not np.allclose(values, np.rint(values)) or np.any(values < -1) or np.any(values > 255):
            raise ValueError("palm matrix must contain integer values in [-1, 255]")
        timestamp_ns = int(timestamp_ns)
        timestamps = self.f["dexterous_hand/palm_stream/timestamp_ns"]
        if self.palm_count and timestamp_ns <= int(timestamps[self.palm_count - 1]):
            raise ValueError("out-of-order palm timestamp")
        index = self.palm_count
        for name, value in {
            "dexterous_hand/palm_stream/matrix": np.rint(values).astype(np.int16),
            "dexterous_hand/palm_stream/timestamp_ns": timestamp_ns,
            "dexterous_hand/palm_stream/sequence": int(sequence),
        }.items():
            dataset = self.f[name]
            dataset.resize((index + 1,) + dataset.shape[1:])
            dataset[index] = value
        self.palm_count += 1
        return index

    def append_force(self, values: Any, timestamp_ns: int, sequence: int) -> int:
        forces = np.asarray(values, dtype=np.float64)
        if forces.shape != (6,) or not np.all(np.isfinite(forces)):
            raise ValueError("force sample must contain six finite values")
        if not np.allclose(forces, np.rint(forces)) or np.any(np.abs(forces) > 2000):
            raise ValueError("force sample exceeds device range or is not integral")
        timestamp_ns = int(timestamp_ns)
        timestamps = self.f["dexterous_hand/force_stream/timestamp_ns"]
        if self.force_count and timestamp_ns <= int(timestamps[self.force_count - 1]):
            raise ValueError("out-of-order force timestamp")
        index = self.force_count
        for name, value in {
            "dexterous_hand/force_stream/value": np.rint(forces).astype(np.int16),
            "dexterous_hand/force_stream/timestamp_ns": timestamp_ns,
            "dexterous_hand/force_stream/sequence": int(sequence),
        }.items():
            dataset = self.f[name]
            dataset.resize((index + 1,) + dataset.shape[1:])
            dataset[index] = value
        self.force_count += 1
        return index

    def close(self) -> Dict[str, Any]:
        ts = self.f["timestamps/master_timestamp_ns"][:]
        duration = (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0.0
        hz = (len(ts) - 1) / duration if duration > 0 else 0.0
        self.f.attrs["actual_capture_hz"] = hz
        tactile_ts = self.f["dexterous_hand/tactile_stream/timestamp_ns"][:]
        tactile_duration = ((tactile_ts[-1] - tactile_ts[0]) / 1e9
                            if len(tactile_ts) > 1 else 0.0)
        tactile_hz = ((len(tactile_ts) - 1) / tactile_duration
                      if tactile_duration > 0 else 0.0)
        summary = {"frames": len(ts), "duration_s": duration, "actual_capture_hz": hz,
                   "tactile_frames": len(tactile_ts), "tactile_capture_hz": tactile_hz,
                   "palm_frames": self.palm_count, "force_frames": self.force_count}
        selected = set(self.config.selected_channels)
        valid_datasets = [("d405", "camera/d405/valid")]
        for channel, label, name in [
            ("vision_hand", "right_hand", "observations/right_hand/valid"),
            ("hand_state", "hand_state", "dexterous_hand/state_valid"),
            ("action", "action", "action/valid"),
            ("operator_exoskeleton", "operator_exoskeleton", "operator/exoskeleton/valid"),
            ("tactile", "tactile", "dexterous_hand/tactile/valid"),
            ("vive_tracker", "tracker", "vive_tracker/valid"),
            ("d435i", "d435i", "camera/d435i/valid"),
        ]:
            if channel in selected and name in self.f: valid_datasets.append((label, name))
        for label, name in valid_datasets:
            values = self.f[name][:]
            summary[f"{label}_valid_ratio"] = float(np.mean(values)) if values.size else 0.0
        metric_datasets = [("right_hand_temporal_error", "sync/right_hand_dt_ms"), ("operator_interpolation_gap", "sync/operator_interpolation_gap_ms"), ("hand_state_interpolation_gap", "sync/hand_state_interpolation_gap_ms"), ("tactile_age", "dexterous_hand/tactile/age_ms"), ("tracker_interpolation_gap", "sync/tracker_interpolation_gap_ms"), ("action_age", "sync/action_age_ms")]
        if "sync/d435i_dt_ms" in self.f: metric_datasets.insert(0, ("d435i_temporal_error", "sync/d435i_dt_ms"))
        for label, name in metric_datasets:
            values = np.asarray(self.f[name][:], dtype=float)
            values = np.abs(values[np.isfinite(values)])
            summary[f"{label}_mean_ms"] = float(np.mean(values)) if values.size else math.nan
            summary[f"{label}_max_ms"] = float(np.max(values)) if values.size else math.nan
        self.f.require_group("quality")
        for key, value in summary.items(): self.f["quality"].attrs[key] = value
        self.f.close()
        summary["file_size_mb"] = self.path.stat().st_size / 1024 ** 2
        return summary
