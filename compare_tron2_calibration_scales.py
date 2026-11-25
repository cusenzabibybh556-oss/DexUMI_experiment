#!/usr/bin/env python3
"""只读诊断同朝向跨中心距离比、返回漂移及跨会话验证；不拟合或应用尺度。

输入须为已结束的 samples.jsonl。status=18 可保留作诊断，原标记不修改。
plan 中的 *_group_indices 均为原始 sample 事件的零起点顺序，不能当成 id。
跨会话清单示例：
{"same_map_mount_as":"/绝对路径/本次/samples.jsonl", "evidence":"地图与安装连续性说明",
 "datasets":[{"input":"/旧会话/samples.jsonl", "group_ids":[25,26,27,28,30,32]}]}
"""
from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from dexumi_wrist_solver import tracker_pose_for_calibration
from solve_tron2_tracker_calibration import PoseGroup, group_errors, inverse, load_samples, statistics, transform


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_transform(value, name):
    T = np.asarray(value, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-6)):
        raise ValueError(name + " 不是有效刚体矩阵")
    return T


def angle_deg(R):
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def load_diagnostic(path, convention):
    meta, groups, rejected, digest = load_samples(path, diagnostic_statuses={18})
    if convention == "raw-xyzw" and meta.get("tracker_transport", {}).get("kind") != "passive_hidraw":
        raise ValueError("raw-xyzw 仅用于明确保留 HID 原始报文的 passive_hidraw 会话")
    groups = [PoseGroup(g.id, g.label, g.robot, tracker_pose_for_calibration(g.tracker, convention),
                        g.statuses, g.pair_count) for g in groups]
    events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [event["id"] for event in events if event.get("event") == "sample"]
    return meta, groups, rejected, digest, ids


def physical_identity(meta):
    model = meta.get("robot_kinematics", {})
    return {
        "arm": meta.get("arm"), "robot_id": meta.get("robot_id"),
        "tracker_mac": meta.get("tracker_transport", {}).get("mac"),
        "tracker_index": meta.get("tracker_index"), "model_sha256": model.get("model_sha256"),
        "base_frame": model.get("base_frame"), "wrist_frame": model.get("wrist_frame"),
        "tcp_offset_m": model.get("tcp_offset_m"),
    }


def check_compatible(a, b):
    ia, ib = physical_identity(a), physical_identity(b)
    if any(value is None for value in ia.values()) or any(value is None for value in ib.values()):
        raise ValueError("缺少机器人、Tracker、模型或 TCP 身份信息")
    different = [key for key in ia if ia[key] != ib[key]]
    if different:
        raise ValueError("会话硬件/模型身份不同：" + ", ".join(different))


def matrix_set(calibration, offset):
    X_E = validate_transform(calibration["T_ET"], "T_ET")
    C = transform(np.eye(3), -np.asarray(offset, dtype=float))
    X_F = validate_transform(calibration.get("T_FT", inverse(C) @ X_E), "T_FT")
    if not np.allclose(C @ X_F, X_E, atol=1e-6):
        raise ValueError("T_FT 与 T_ET 的 SDK TCP 转换不一致")
    Y = validate_transform(calibration["T_WB"], "T_WB")
    if "T_BW" in calibration and not np.allclose(inverse(Y), calibration["T_BW"], atol=1e-6):
        raise ValueError("T_WB 与 T_BW 不互逆")
    return C, X_F, X_E, Y


def pair_diagnostic(first, second, C, X_F, Y, min_baseline_m=.05):
    """比值方向为 Tracker 距离 / 机器人模型法兰距离；不将比值当补偿参数。"""
    F0, F1 = first.robot @ C, second.robot @ C
    Q0, Q1 = first.tracker, second.tracker
    # 在 Tracker 世界系中还原法兰原点，剔除实际旋转产生的安装杠杆位移。
    W_F0, W_F1 = Q0 @ inverse(X_F), Q1 @ inverse(X_F)
    flange_delta = F1[:3, 3] - F0[:3, 3]
    tracker_delta = Q1[:3, 3] - Q0[:3, 3]
    corrected_delta = W_F1[:3, 3] - W_F0[:3, 3]
    denominator = float(np.linalg.norm(flange_delta))
    raw_distance = float(np.linalg.norm(tracker_delta))
    corrected_distance = float(np.linalg.norm(corrected_delta))
    robot_relative_R = F0[:3, :3].T @ F1[:3, :3]
    tracker_relative_R = Q0[:3, :3].T @ Q1[:3, :3]
    predicted_relative_R = X_F[:3, :3] @ tracker_relative_R @ X_F[:3, :3].T
    predicted_Q0, predicted_Q1 = Y @ F0 @ X_F, Y @ F1 @ X_F
    unexplained_delta = tracker_delta - (predicted_Q1[:3, 3] - predicted_Q0[:3, 3])
    ratio_valid = denominator >= min_baseline_m
    return {
        "group_ids": [first.id, second.id], "raw_tracking_statuses": [first.statuses, second.statuses],
        "contains_status18": 18 in first.statuses or 18 in second.statuses,
        "robot_flange_distance_mm": denominator * 1000,
        "tracker_distance_mm": raw_distance * 1000,
        "tracker_rotation_corrected_flange_distance_mm": corrected_distance * 1000,
        "removed_rotation_lever_delta_mm": ((tracker_delta - corrected_delta) * 1000).tolist(),
        "robot_orientation_change_deg": angle_deg(robot_relative_R),
        "tracker_orientation_change_deg": angle_deg(tracker_relative_R),
        "relative_rotation_mapping_error_deg": angle_deg(robot_relative_R.T @ predicted_relative_R),
        "raw_distance_ratio_tracker_over_robot": raw_distance / denominator if ratio_valid else None,
        "corrected_distance_ratio_tracker_over_robot": corrected_distance / denominator if ratio_valid else None,
        "ratio_usable": ratio_valid,
        "ratio_rejection_reason": None if ratio_valid else "法兰基线短于阈值；不对返回起点等近零距离计算比例",
        "unexplained_tracker_translation_delta_mm": (unexplained_delta * 1000).tolist(),
        "unexplained_tracker_translation_norm_mm": float(np.linalg.norm(unexplained_delta) * 1000),
    }


def ratio_summary(rows):
    selected = [row for row in rows if row["ratio_usable"]]
    summary = {"pair_count": len(rows), "usable_pair_count": len(selected),
               "status18_pair_count": sum(row["contains_status18"] for row in rows)}
    for field in ("raw_distance_ratio_tracker_over_robot", "corrected_distance_ratio_tracker_over_robot"):
        values = np.asarray([row[field] for row in selected])
        summary[field] = ({"min": float(values.min()), "median": float(np.median(values)),
                           "max": float(values.max())} if len(values) else None)
    return summary


def analyze(groups, sample_ids, plan, calibration, offset, min_baseline_m=.05):
    C, X_F, X_E, Y = matrix_set(calibration, offset)
    plan_groups = plan["groups"]
    if len(plan_groups) != len(sample_ids) or plan.get("sample_count", len(sample_ids)) != len(sample_ids):
        raise ValueError("完成样本数与计划不一致；不能猜测采样序号对应")
    indices = plan["same_orientation_group_indices"]
    if (len(set(indices)) != len(indices)
            or any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(sample_ids)
                   for index in indices)):
        raise ValueError("same_orientation_group_indices 不是有效且不重复的零起点采样顺序")
    available = {g.id: g for g in groups}
    selected = [(index, available[sample_ids[index]]) for index in indices if sample_ids[index] in available]
    report = {"diagnostic_only": True, "scale_fitted": False, "applied_to_mapping": False,
              "ratio_definition": "Tracker distance / model zero-offset flange distance",
              "minimum_ratio_baseline_m": min_baseline_m,
              "unavailable_reference_groups": [{"sample_index": i, "id": sample_ids[i]} for i in indices
                                               if sample_ids[i] not in available],
              "reference_groups": [{"sample_index": i, "id": g.id, "center": plan_groups[i]["center"],
                                    "role": plan_groups[i]["role"], "raw_tracking_statuses": g.statuses}
                                   for i, g in selected],
              "training_center_pairs": [], "independent_center_pairs": [], "return_checks": []}
    for (i, first), (j, second) in combinations(selected, 2):
        role_i, role_j = plan_groups[i]["role"], plan_groups[j]["role"]
        # 抬升过渡与返回检查可以检查重复性，但不属于标定训练/留出基线。
        if role_i not in {"fit", "held_out"} or role_j not in {"fit", "held_out"}:
            continue
        if plan_groups[i]["center"] == plan_groups[j]["center"]:
            continue
        pair = pair_diagnostic(first, second, C, X_F, Y, min_baseline_m)
        pair.update(sample_indices=[i, j], centers=[plan_groups[i]["center"], plan_groups[j]["center"]])
        key = "independent_center_pairs" if "held_out" in (role_i, role_j) else "training_center_pairs"
        report[key].append(pair)
    for j, returned in selected:
        if plan_groups[j]["role"] != "return_check":
            continue
        start = next(((i, g) for i, g in selected if i < j and plan_groups[i]["center"] == plan_groups[j]["center"]
                      and plan_groups[i]["role"] != "return_check"), None)
        if start is None:
            report["return_checks"].append({"sample_index": j, "error": "缺少同中心的返回前参考组"})
            continue
        i, first = start
        row = pair_diagnostic(first, returned, C, X_F, Y, min_baseline_m)
        row.update(sample_indices=[i, j], interpretation="相同计划姿态前后差；未被模型末端变化解释的部分可能含地图漂移、安装形变和模型误差")
        report["return_checks"].append(row)
    report["ratio_summaries"] = {key: ratio_summary(report[key])
                                 for key in ("training_center_pairs", "independent_center_pairs")}
    report["warnings"] = [
        "不优化尺度、不写入主矩阵；距离比不是经独立长度真值验证的修正系数。",
        "旋转杠杆校正使用候选 T_FT，具有模型依赖；须同时保留原始距离比。",
        "含status18的结果仅用于诊断，未把18重写为2，也不由此宣称跟踪状态已被官方确认。",
        "共享中心的多个距离比不是独立样本；汇总不是统计置信区间。",
    ]
    return report


def cross_validate(manifest_path, main_path, main_meta, calibration):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    reference = Path(manifest["same_map_mount_as"])
    if not reference.is_absolute():
        reference = manifest_path.parent / reference
    if reference.resolve() != Path(main_path).resolve() or not str(manifest.get("evidence", "")).strip():
        raise ValueError("跨会话清单必须明确指向本次 samples.jsonl 并给出地图/安装连续性说明")
    output = []
    for item in manifest["datasets"]:
        path = Path(item["input"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        meta, groups, rejected, digest, _ = load_diagnostic(path, calibration["tracker_quaternion_convention"])
        check_compatible(main_meta, meta)
        ids = item.get("group_ids", [g.id for g in groups])
        if len(set(ids)) != len(ids) or any(isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in ids):
            raise ValueError("跨会话group_ids必须为不重复的正整数id，不是采样顺序")
        selected = [g for g in groups if g.id in ids]
        rows = [group_errors(g, np.asarray(calibration["T_ET"]), np.asarray(calibration["T_WB"])) for g in selected]
        normal_rows = [r for r in rows if r["tracking_statuses"] == [2]]
        output.append({"input": str(path), "input_sha256": digest,
                       "same_map_mount_basis": "explicit manifest attestation plus matching hardware/model identity",
                       "continuity_evidence": manifest["evidence"],
                       "selected_group_ids": ids, "unavailable_group_ids": sorted(set(ids) - {g.id for g in selected}),
                       "rejected_groups": rejected, "refitted": False,
                       "metrics_status2_only": statistics(normal_rows),
                       "metrics_including_status18_diagnostic": statistics(rows), "per_group": rows})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--cross-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-baseline", type=float, default=.05, help="距离比最小法兰基线，单位m")
    args = parser.parse_args()
    if not np.isfinite(args.min_baseline) or args.min_baseline <= 0:
        parser.error("min-baseline 须为有限正数")
    try:
        calibration = read_json(args.calibration)
        convention = calibration["tracker_quaternion_convention"]
        meta, groups, rejected, digest, sample_ids = load_diagnostic(args.input, convention)
        if "session" not in calibration:
            raise ValueError("calibration缺少源会话信息，不能核对模型与设备身份")
        check_compatible(meta, calibration["session"])
        if calibration.get("input_sha256") != digest:
            raise ValueError("候选calibration的input_sha256不是本次samples；跨会话候选请显式另行验证")
        plan = read_json(args.plan)
        if "sampling_plan" in meta and Path(meta["sampling_plan"]).resolve() != args.plan.resolve():
            raise ValueError("samples引用的采样计划不是指定plan")
        report = analyze(groups, sample_ids, plan, calibration, meta["robot_kinematics"]["tcp_offset_m"], args.min_baseline)
        report.update(input_file=str(args.input.resolve()), input_sha256=digest,
                      plan_file=str(args.plan.resolve()), calibration_file=str(args.calibration.resolve()),
                      calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                      tracker_quaternion_convention=convention, rejected_groups=rejected,
                      hardware_connected=False)
        if args.cross_manifest:
            report["cross_dataset_validation"] = cross_validate(args.cross_manifest, args.input, meta, calibration)
        output = args.output or args.input.parent / "diagnostics" / (
            "scale_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8] + ".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        print("只读诊断已保存：" + str(output.resolve()))
        print(json.dumps(report["ratio_summaries"], ensure_ascii=False, indent=2))
    except (ValueError, KeyError, TypeError, OSError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
