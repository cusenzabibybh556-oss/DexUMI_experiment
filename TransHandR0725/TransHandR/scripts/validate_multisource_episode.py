#!/usr/bin/env python3
"""Validate a TransHandR episode and report synchronization quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import h5py
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "缺少校验依赖，请先运行: python3 -m pip install --user -r requirements-collector.txt"
    ) from exc


BASE_DATASETS = [
    "timestamps/master_timestamp_ns",
    "observations/images/d405_rgb", "observations/depth/d405_depth",
    "observations/right_hand/world_keypoints", "observations/right_hand/relative_keypoints",
    "observations/right_hand/valid", "observations/right_hand/source_timestamp",
    "observations/right_hand/unified_timestamp_ns", "observations/right_hand/host_receive_timestamp_ns",
    "dexterous_hand/joint_positions", "dexterous_hand/state_valid",
    "dexterous_hand/source_timestamp", "dexterous_hand/unified_timestamp_ns",
    "dexterous_hand/host_receive_timestamp_ns",
    "action/command", "action/valid", "action/command_timestamp_ns",
    "vive_tracker/pos", "vive_tracker/quat", "vive_tracker/valid",
    "vive_tracker/source_timestamp", "vive_tracker/unified_timestamp_ns",
    "vive_tracker/host_receive_timestamp_ns",
    "camera/d405/frame_number", "camera/d405/device_timestamp",
    "camera/d405/host_receive_timestamp_ns", "camera/d405/valid",
    "sync/right_hand_dt_ms", "sync/hand_state_interpolation_gap_ms",
    "sync/tracker_interpolation_gap_ms", "sync/action_age_ms",
]

HEAD_CAMERA_DATASETS = [
    "observations/images/d435i_rgb", "observations/depth/d435i_depth",
    "camera/d435i/frame_number", "camera/d435i/device_timestamp",
    "camera/d435i/unified_timestamp_ns", "camera/d435i/host_receive_timestamp_ns",
    "camera/d435i/valid", "sync/d435i_dt_ms",
]

V2_DATASETS = [
    "operator/exoskeleton/joint_positions", "operator/exoskeleton/valid",
    "operator/exoskeleton/source_timestamp_ns", "operator/exoskeleton/unified_timestamp_ns",
    "operator/exoskeleton/host_receive_timestamp_ns",
    "dexterous_hand/tactile/matrix", "dexterous_hand/tactile/max",
    "dexterous_hand/tactile/mean", "dexterous_hand/tactile/contact_area",
    "dexterous_hand/tactile/valid", "dexterous_hand/tactile/source_timestamp_ns",
    "dexterous_hand/tactile/unified_timestamp_ns",
    "dexterous_hand/tactile/host_receive_timestamp_ns",
    "sync/operator_interpolation_gap_ms", "sync/tactile_dt_ms",
]

V3_MASTER_DATASETS = [
    "operator/exoskeleton/joint_positions", "operator/exoskeleton/valid",
    "operator/exoskeleton/source_timestamp_ns", "operator/exoskeleton/unified_timestamp_ns",
    "operator/exoskeleton/host_receive_timestamp_ns",
    "dexterous_hand/tactile/index", "dexterous_hand/tactile/age_ms",
    "dexterous_hand/tactile/valid", "sync/operator_interpolation_gap_ms",
]

V3_TACTILE_STREAM_DATASETS = [
    "dexterous_hand/tactile_stream/matrix",
    "dexterous_hand/tactile_stream/timestamp_ns",
    "dexterous_hand/tactile_stream/sequence",
]

# label, valid dataset, aligned timestamp, error metric, configured limit
BASE_SOURCES: List[Tuple[str, str, str, str, str]] = [
    ("right_hand", "observations/right_hand/valid", "observations/right_hand/unified_timestamp_ns",
     "sync/right_hand_dt_ms", "right_hand_max_gap_ms"),
    ("hand_state", "dexterous_hand/state_valid", "dexterous_hand/unified_timestamp_ns",
     "sync/hand_state_interpolation_gap_ms", "hand_state_max_gap_ms"),
    ("action", "action/valid", "action/command_timestamp_ns",
     "sync/action_age_ms", "action_max_age_ms"),
    ("tracker", "vive_tracker/valid", "vive_tracker/unified_timestamp_ns",
     "sync/tracker_interpolation_gap_ms", "tracker_max_gap_ms"),
]

HEAD_SOURCE = (
    "d435i", "camera/d435i/valid", "camera/d435i/unified_timestamp_ns",
    "sync/d435i_dt_ms", "d435i_max_gap_ms",
)

V2_SOURCES = [
    ("operator_exoskeleton", "operator/exoskeleton/valid",
     "operator/exoskeleton/unified_timestamp_ns",
     "sync/operator_interpolation_gap_ms", "operator_max_gap_ms"),
    ("tactile", "dexterous_hand/tactile/valid",
     "dexterous_hand/tactile/unified_timestamp_ns",
     "sync/tactile_dt_ms", "tactile_max_gap_ms"),
]

V3_SOURCES = [
    ("operator_exoskeleton", "operator/exoskeleton/valid",
     "operator/exoskeleton/unified_timestamp_ns",
     "sync/operator_interpolation_gap_ms", "operator_max_gap_ms"),
]

SOURCE_CHANNELS = {
    "right_hand": "vision_hand",
    "hand_state": "hand_state",
    "action": "action",
    "tracker": "vive_tracker",
    "d435i": "d435i",
    "operator_exoskeleton": "operator_exoskeleton",
    "tactile": "tactile",
}


def _mask(dataset: h5py.Dataset, frames: int) -> np.ndarray:
    return np.asarray(dataset[:], dtype=bool).reshape(frames, -1).any(axis=1)


def _first_column(dataset: h5py.Dataset, frames: int) -> np.ndarray:
    return np.asarray(dataset[:]).reshape(frames, -1)[:, 0]


def _stats(values: np.ndarray) -> Dict[str, float]:
    values = np.abs(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if not values.size:
        return {"mean_abs_ms": float("nan"), "p95_abs_ms": float("nan"), "max_abs_ms": float("nan")}
    return {
        "mean_abs_ms": float(np.mean(values)),
        "p95_abs_ms": float(np.percentile(values, 95)),
        "max_abs_ms": float(np.max(values)),
    }


def _valid_rigid_transform(matrix: np.ndarray) -> bool:
    return (
        matrix.shape == (4, 4)
        and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
        and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3)
    )


def validate(path: Path) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    timing: Dict[str, Dict[str, Any]] = {}

    with h5py.File(path, "r") as f:
        schema_version = str(f.attrs.get("schema_version", ""))
        is_v2 = schema_version.startswith("quality-first-v2")
        is_v3 = schema_version.startswith("quality-first-v3")
        is_v31 = schema_version.startswith(("quality-first-v3.1", "quality-first-v3.2"))
        is_v32 = schema_version.startswith("quality-first-v3.2")
        selected_channels = None
        if "selected_channels" in f.attrs:
            raw_selected = f.attrs["selected_channels"]
            if isinstance(raw_selected, bytes):
                raw_selected = raw_selected.decode("utf-8", "replace")
            try:
                selected_channels = list(json.loads(raw_selected))
            except (TypeError, json.JSONDecodeError):
                warnings.append("selected_channels attr 无法解析，按旧数据的全通道模式校验")
        head_expected = bool(f.attrs.get("enable_head_camera", False))
        has_head = "camera/d435i" in f
        if head_expected and not has_head:
            errors.append("enable_head_camera=true 但缺少 D435i datasets")

        required = BASE_DATASETS + (V2_DATASETS if is_v2 else []) + (
            V3_MASTER_DATASETS if is_v3 else []) + (
            HEAD_CAMERA_DATASETS if head_expected or has_head else [])
        stream_required = V3_TACTILE_STREAM_DATASETS if is_v3 else []
        missing = [name for name in required + stream_required if name not in f]
        if missing:
            errors.append("缺少 datasets: " + ", ".join(missing))
        if "schema_version" not in f.attrs:
            errors.append("缺少 schema_version attr")
        if missing:
            return {"ok": False, "file": str(path), "errors": errors, "warnings": warnings}

        master = f["timestamps/master_timestamp_ns"][:]
        frames = len(master)
        wrong_lengths = {name: f[name].shape[0] for name in required if f[name].shape[0] != frames}
        if wrong_lengths:
            errors.append(f"时间维 T 不一致（期望 {frames}）: {wrong_lengths}")
            return {"ok": False, "file": str(path), "frames": frames, "errors": errors, "warnings": warnings}
        if frames > 1 and not np.all(np.diff(master) > 0):
            errors.append("master_timestamp_ns 不是严格递增")

        duration = (master[-1] - master[0]) / 1e9 if frames > 1 else 0.0
        actual_fps = (frames - 1) / duration if duration > 0 else 0.0
        target_fps = float(f.attrs.get("camera_fps", 0))
        if frames > 2 and target_fps and not 0.8 * target_fps <= actual_fps <= 1.2 * target_fps:
            warnings.append(f"实际 FPS {actual_fps:.2f} 偏离配置 {target_fps:.2f}")

        sources = list(BASE_SOURCES)
        if is_v2:
            sources.extend(V2_SOURCES)
        if is_v3:
            sources.extend(V3_SOURCES)
        if head_expected or has_head:
            sources.append(HEAD_SOURCE)

        ratios = {"d405": float(np.mean(f["camera/d405/valid"][:])) if frames else 0.0}
        for label, valid_name, timestamp_name, metric_name, limit_attr in sources:
            valid = _mask(f[valid_name], frames) if frames else np.array([], dtype=bool)
            ratios[label] = float(np.mean(valid)) if frames else 0.0
            selected = selected_channels is None or SOURCE_CHANNELS[label] in selected_channels
            if not selected and np.any(valid):
                errors.append(f"未选择的通道 {SOURCE_CHANNELS[label]} 仍包含 valid 帧")

            timestamps = _first_column(f[timestamp_name], frames) if frames else np.array([])
            if valid.sum() > 1 and np.any(np.diff(timestamps[valid]) < 0):
                errors.append(f"{timestamp_name} 在 valid 帧中不是单调非递减")

            metric = _first_column(f[metric_name], frames)[valid] if frames else np.array([])
            source_stats = _stats(metric)
            source_stats.update(
                valid_ratio=ratios[label],
                valid_frames=int(valid.sum()),
                limit_ms=float(f.attrs[limit_attr]),
                selected=selected,
            )
            timing[label] = source_stats

            finite_metric = np.abs(metric[np.isfinite(metric)])
            limit_ms = source_stats["limit_ms"]
            exceeds_limit = limit_ms > 0 and np.any(finite_metric > limit_ms + 1e-4)
            if finite_metric.size != metric.size or exceeds_limit:
                errors.append(f"{metric_name} 的 valid 帧超出阈值 {source_stats['limit_ms']} ms")

        action_valid = np.asarray(f["action/valid"][:], dtype=bool)
        command_ts = f["action/command_timestamp_ns"][:]
        if np.any(command_ts[action_valid] > master[action_valid]):
            errors.append("检测到 future action 泄漏")

        if is_v3:
            tactile_matrix = f["dexterous_hand/tactile_stream/matrix"]
            tactile_ts = np.asarray(
                f["dexterous_hand/tactile_stream/timestamp_ns"][:], dtype=np.int64)
            tactile_sequence = f["dexterous_hand/tactile_stream/sequence"]
            tactile_frames = len(tactile_ts)
            if tactile_matrix.shape[0] != tactile_frames or len(tactile_sequence) != tactile_frames:
                errors.append("异步触觉流的 matrix/timestamp/sequence 长度不一致")
            if tactile_matrix.shape[1:] != (5, 12, 6):
                errors.append("异步触觉矩阵维度不是 5x12x6")
            if tactile_matrix.dtype != np.dtype("int16"):
                errors.append("异步触觉矩阵 dtype 不是 int16")
            if tactile_frames > 1 and np.any(np.diff(tactile_ts) <= 0):
                errors.append("异步触觉 timestamp_ns 不是严格递增")

            if is_v32:
                palm_paths = (
                    "dexterous_hand/palm_stream/matrix",
                    "dexterous_hand/palm_stream/timestamp_ns",
                    "dexterous_hand/palm_stream/sequence",
                    "dexterous_hand/palm/index",
                    "dexterous_hand/palm/age_ms",
                    "dexterous_hand/palm/valid",
                )
                absent = [name for name in palm_paths if name not in f]
                if absent:
                    errors.append("缺少掌心触觉 datasets: " + ", ".join(absent))
                else:
                    palm_matrix = f[palm_paths[0]]
                    palm_ts = np.asarray(f[palm_paths[1]][:], dtype=np.int64)
                    palm_sequence = f[palm_paths[2]]
                    palm_index = np.asarray(f[palm_paths[3]][:], dtype=np.int64)
                    palm_age = np.asarray(f[palm_paths[4]][:], dtype=float)
                    palm_valid = np.asarray(f[palm_paths[5]][:], dtype=bool)
                    if palm_matrix.shape != (len(palm_ts), 14, 15) or len(palm_sequence) != len(palm_ts):
                        errors.append("异步掌心流的 matrix/timestamp/sequence 维度不一致")
                    if palm_matrix.dtype != np.dtype("int16"):
                        errors.append("异步掌心矩阵 dtype 不是 int16")
                    if len(palm_ts) > 1 and np.any(np.diff(palm_ts) <= 0):
                        errors.append("异步掌心 timestamp_ns 不是严格递增")
                    if len(palm_index) != frames or len(palm_age) != frames or len(palm_valid) != frames:
                        errors.append("掌心主表长度与 D405 不一致")
                    elif np.any(palm_valid):
                        refs = palm_index[palm_valid]
                        if np.any(refs < 0) or np.any(refs >= len(palm_ts)):
                            errors.append("掌心引用 index 超出异步流范围")
                        else:
                            age = (master[palm_valid] - palm_ts[refs]) / 1e6
                            if np.any(age < 0) or np.any(age > float(f.attrs["tactile_max_gap_ms"])):
                                errors.append("掌心引用不是未过期的历史帧")
                            if not np.allclose(palm_age[palm_valid], age, atol=1e-3, rtol=0):
                                errors.append("掌心 age_ms 与引用时间戳不一致")
                    if np.any(palm_index[~palm_valid] != -1) or np.any(~np.isnan(palm_age[~palm_valid])):
                        errors.append("无效掌心引用必须为 index=-1、age_ms=NaN")
                    if not selected_channels or "tactile" not in selected_channels:
                        if selected_channels is not None and (len(palm_ts) or np.any(palm_valid)):
                            errors.append("未选择触觉通道但掌心流非空")
                    if str(f.attrs.get("target_hand_model", "")) != "o6" and len(palm_ts):
                        errors.append("非 O6 Episode 的掌心流必须为空")

            if is_v31:
                force_paths = (
                    "dexterous_hand/force_stream/value",
                    "dexterous_hand/force_stream/timestamp_ns",
                    "dexterous_hand/force_stream/sequence",
                )
                absent = [name for name in force_paths if name not in f]
                if absent:
                    errors.append("缺少异步力流 datasets: " + ", ".join(absent))
                else:
                    values = f[force_paths[0]]
                    force_ts = np.asarray(f[force_paths[1]][:], dtype=np.int64)
                    sequences = f[force_paths[2]]
                    if values.shape != (len(force_ts), 6) or len(sequences) != len(force_ts):
                        errors.append("异步力流的 value/timestamp/sequence 维度不一致")
                    if values.dtype != np.dtype("int16"):
                        errors.append("异步力流 dtype 不是 int16")
                    if len(force_ts) > 1 and np.any(np.diff(force_ts) <= 0):
                        errors.append("异步力流 timestamp_ns 不是严格递增")
                    if selected_channels is not None and "inspire_force" not in selected_channels and len(force_ts):
                        errors.append("未选择 Inspire Force 通道但异步力流非空")
                    if str(f.attrs.get("target_hand_model", "")) != "inspire" and len(force_ts):
                        errors.append("非 Inspire Episode 的异步力流必须为空")

            tactile_valid = np.asarray(
                f["dexterous_hand/tactile/valid"][:], dtype=bool)
            tactile_index = np.asarray(
                f["dexterous_hand/tactile/index"][:], dtype=np.int64)
            tactile_age = np.asarray(
                f["dexterous_hand/tactile/age_ms"][:], dtype=float)
            selected = selected_channels is None or "tactile" in selected_channels
            ratios["tactile"] = float(np.mean(tactile_valid)) if frames else 0.0
            if not selected and (tactile_frames or np.any(tactile_valid)):
                errors.append("未选择触觉通道但异步触觉流或有效引用非空")
            if np.any(tactile_index[~tactile_valid] != -1):
                errors.append("无效触觉引用的 index 必须为 -1")
            if np.any(~np.isnan(tactile_age[~tactile_valid])):
                errors.append("无效触觉引用的 age_ms 必须为 NaN")
            valid_indices = tactile_index[tactile_valid]
            if np.any(valid_indices < 0) or np.any(valid_indices >= tactile_frames):
                errors.append("触觉引用 index 超出异步触觉流范围")
            if np.any(tactile_valid):
                referenced_ts = tactile_ts[valid_indices]
                valid_master = master[tactile_valid]
                expected_age = (valid_master - referenced_ts) / 1e6
                if np.any(referenced_ts > valid_master):
                    errors.append("触觉引用使用了 D405 之后的未来帧")
                if np.any(tactile_age[tactile_valid] < 0):
                    errors.append("触觉 age_ms 出现负数")
                if not np.allclose(
                    tactile_age[tactile_valid], expected_age, atol=1e-3, rtol=0):
                    errors.append("触觉 age_ms 与引用时间戳不一致")
                if np.any(tactile_age[tactile_valid] >
                          float(f.attrs["tactile_max_gap_ms"]) + 1e-4):
                    errors.append("有效触觉引用超过 tactile_max_gap_ms")
            timing["tactile"] = {
                **_stats(tactile_age[tactile_valid]),
                "valid_ratio": ratios["tactile"],
                "valid_frames": int(tactile_valid.sum()),
                "stream_frames": tactile_frames,
                "limit_ms": float(f.attrs["tactile_max_gap_ms"]),
                "selected": selected,
            }

        tracker_valid = np.asarray(f["vive_tracker/valid"][:, 0], dtype=bool)
        if np.any(tracker_valid):
            norms = np.linalg.norm(f["vive_tracker/quat"][:, 0][tracker_valid], axis=1)
            if np.any(np.abs(norms - 1) > 1e-3):
                errors.append("Tracker quaternion norm 不接近 1")

        nan_placeholders = [
            ("observations/right_hand/valid", "observations/right_hand/world_keypoints"),
            ("dexterous_hand/state_valid", "dexterous_hand/joint_positions"),
            ("action/valid", "action/command"),
            ("vive_tracker/valid", "vive_tracker/pos"),
        ]
        if is_v2 or is_v3:
            nan_placeholders.extend([
                ("operator/exoskeleton/valid", "operator/exoskeleton/joint_positions"),
            ])
        if is_v2:
            nan_placeholders.extend([
                ("dexterous_hand/tactile/valid", "dexterous_hand/tactile/matrix"),
            ])
        for valid_name, data_name in nan_placeholders:
            valid = _mask(f[valid_name], frames) if frames else np.array([], dtype=bool)
            if np.any(~valid) and not np.all(np.isnan(f[data_name][:][~valid])):
                errors.append(f"{data_name} 的 invalid 帧未使用 NaN")

        if is_v2 or is_v3:
            if f["operator/exoskeleton/joint_positions"].shape[1:] != (21,):
                errors.append("外骨骼原始关节维度不是 21")
            required_attrs = ["control_source", "target_hand_model", "hand_side",
                              "telemetry_protocol_version"]
            missing_attrs = [name for name in required_attrs if name not in f.attrs]
            if missing_attrs:
                errors.append("v2/v3 缺少 attrs: " + ", ".join(missing_attrs))
        if is_v2:
            if f["dexterous_hand/tactile/matrix"].shape[1:] != (5, 12, 6):
                errors.append("触觉矩阵维度不是 5x12x6")

        rigid_transforms = [
            "calibration/d405/T_wrist_camera",
            "calibration/d405/T_tracker_camera",
        ]
        if head_expected or has_head:
            rigid_transforms.append("calibration/d435i/T_d405_d435i")
        for name in rigid_transforms:
            if name not in f:
                errors.append(f"缺少 {name}")
            elif not _valid_rigid_transform(f[name][:]):
                errors.append(f"{name} 不是合法刚体变换")

    return {
        "ok": not errors,
        "file": str(path),
        "frames": frames,
        "duration_s": duration,
        "actual_capture_hz": actual_fps,
        "selected_channels": selected_channels,
        "valid_ratios": ratios,
        "timing_ms": timing,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = validate(args.episode)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("PASS" if report["ok"] else "FAIL", report.get("file", args.episode))
        print(
            f"frames={report.get('frames', 0)}  duration={report.get('duration_s', 0):.3f}s  "
            f"fps={report.get('actual_capture_hz', 0):.2f}"
        )
        for label, stats in report.get("timing_ms", {}).items():
            if not stats.get("selected", True):
                print(f"{label:>10}: SKIPPED (not selected)")
                continue
            print(
                f"{label:>10}: valid={stats['valid_ratio']:.3f}  "
                f"|dt| mean/p95/max={stats['mean_abs_ms']:.2f}/"
                f"{stats['p95_abs_ms']:.2f}/{stats['max_abs_ms']:.2f} ms  "
                f"limit={stats['limit_ms']:.1f} ms"
            )
        for kind in ("errors", "warnings"):
            for item in report.get(kind, []):
                print(f"{kind[:-1].upper()}: {item}")

    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
