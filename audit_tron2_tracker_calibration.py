#!/usr/bin/env python3
"""离线审计 DexUMI 腕部标定的采样范围和参数敏感性，不连接或控制硬件。

保持 dexumi_wrist_solver.fit_dexumi 的目标、初值、边界和优化方法不变。
留出组在所有试验中固定；不能根据留出误差挑选首帧或删除样本。
逐组剔除和换参考帧是敏感性分析，不是统计置信区间或外部真值精度。
"""
from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations
import json
from pathlib import Path
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from dexumi_wrist_solver import fit_dexumi, tracker_pose_for_calibration
from solve_tron2_tracker_calibration import PoseGroup, inverse, load_samples, transform


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)), "max": float(values.max()),
    }


def pose_coverage(poses, ids):
    """位置范围及 SO(3) 测地旋转范围；包含原版 rotvec 分支诊断。"""
    positions = np.asarray([p[:3, 3] for p in poses])
    centered = positions - positions.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    rotvecs = [Rotation.from_matrix(p[:3, :3]).as_rotvec() for p in poses]
    distances, angles, relative_rotvecs, branch_pairs = [], [], [], []
    for i, j in combinations(range(len(poses)), 2):
        distances.append(float(np.linalg.norm(positions[i] - positions[j]) * 1000))
        rv = Rotation.from_matrix(poses[i][:3, :3].T @ poses[j][:3, :3]).as_rotvec()
        angle = float(np.linalg.norm(rv))
        angles.append(float(np.degrees(angle)))
        relative_rotvecs.append(rv)
        # 原版 loss 直接相减 rotvec，靠近 ±π 时可能将小转角变为大误差。
        vector_distance = float(np.linalg.norm(rotvecs[i] - rotvecs[j]))
        if angle < np.pi / 2 and vector_distance > np.pi:
            branch_pairs.append({"ids": [ids[i], ids[j]],
                                 "geodesic_angle_deg": float(np.degrees(angle)),
                                 "rotvec_difference_deg": float(np.degrees(vector_distance))})
    rotation_singular = (np.linalg.svd(np.asarray(relative_rotvecs), compute_uv=False)
                         if relative_rotvecs else np.zeros(3))
    return {
        "group_count": len(poses), "group_ids": list(ids),
        "position_min_m": positions.min(axis=0).tolist(),
        "position_max_m": positions.max(axis=0).tolist(),
        "position_extent_mm": (np.ptp(positions, axis=0) * 1000).tolist(),
        "centered_position_singular_values_mm": (singular * 1000).tolist(),
        "position_smallest_largest_singular_ratio":
            float(singular[-1] / singular[0]) if singular[0] > 1e-12 else 0.0,
        "pairwise_distance_mm": distribution(distances),
        "pairwise_rotation_deg": distribution(angles),
        "relative_rotation_singular_values_rad": rotation_singular.tolist(),
        "rotvec_branch_pair_count": len(branch_pairs),
        "rotvec_branch_pairs": branch_pairs,
    }


def coverage(groups, offset):
    C = transform(np.eye(3), -np.asarray(offset, float))
    ids = [g.id for g in groups]
    return {
        "sdk_tcp": pose_coverage([g.robot for g in groups], ids),
        "zero_offset_flange": pose_coverage([g.robot @ C for g in groups], ids),
        "tracker": pose_coverage([g.tracker for g in groups], ids),
    }


def transform_difference(reference, candidate):
    a, b = np.asarray(reference), np.asarray(candidate)
    return {
        "translation_mm": float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000),
        "rotation_deg": float(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())),
    }


def prediction_difference(reference, candidate, groups):
    """固定观测点上的映射变化；比地图原点相关的 T_BW 平移更易解释。"""
    aX, bX = inverse(np.asarray(reference["T_ET"])), inverse(np.asarray(candidate["T_ET"]))
    aY, bY = np.asarray(reference["T_BW"]), np.asarray(candidate["T_BW"])
    differences = [transform_difference(aY @ g.tracker @ aX, bY @ g.tracker @ bX)
                   for g in groups]
    return {key: distribution([row[key] for row in differences])
            for key in ("translation_mm", "rotation_deg")}


def brief_fit(result):
    keys = ("status", "T_ET", "T_FT", "T_BW", "metrics", "observability",
            "fit_group_ids", "validation_group_ids", "translation_bound_hit",
            "optimizer_objective_mixed_m_rad")
    return {key: result[key] for key in keys}


def sensitivity_summary(trials):
    success = [trial for trial in trials if "error" not in trial]
    return {
        "trial_count": len(trials), "success_count": len(success),
        "failed_trials": [trial for trial in trials if "error" in trial],
        "difference_from_baseline": {
            name: {metric: distribution([trial["difference_from_baseline"][name][metric]
                                         for trial in success])
                   for metric in ("translation_mm", "rotation_deg")}
            for name in ("T_ET", "T_BW")},
        "validation_position_rmse_mm": distribution([
            trial["fit"]["metrics"]["validation"]["wrist_position_error_mm"]["rmse"]
            for trial in success]),
        "validation_rotation_rmse_deg": distribution([
            trial["fit"]["metrics"]["validation"]["rotation_error_deg"]["rmse"]
            for trial in success]),
        "worst_validation_prediction_change_mm": max([
            trial["validation_prediction_change"]["translation_mm"]["max"]
            for trial in success], default=None),
        "worst_validation_prediction_change_deg": max([
            trial["validation_prediction_change"]["rotation_deg"]["max"]
            for trial in success], default=None),
    }


def audit(groups, offset, validation_ids, max_anchor_trials=0, skip_leave_one_out=False,
          progress=print, fit_function=fit_dexumi):
    """可注入离线求解函数测试；只重排/剔除拟合组，不修改留出组。"""
    validation_ids = set(validation_ids)
    train = [g for g in groups if g.id not in validation_ids]
    held = [g for g in groups if g.id in validation_ids]
    if len(train) < 4 or len(held) < 3 or not validation_ids <= {g.id for g in groups}:
        raise ValueError("审计至少需要4组拟合和3组有效留出数据；留出组号须全部可用")
    if max_anchor_trials < 0:
        raise ValueError("max_anchor_trials 不能为负数")
    progress("拟合原始顺序基准；所有敏感性试验保持相同独立验证组。")
    baseline = fit_function(groups, offset, validation_ids)
    report = {
        "baseline": brief_fit(baseline),
        "coverage": {"all": coverage(groups, offset), "fit": coverage(train, offset),
                     "validation": coverage(held, offset)},
        "leave_one_fit_group_out": [], "reference_anchor_trials": [],
    }

    def run_trial(trial_groups, description):
        try:
            result = fit_function(trial_groups, offset, validation_ids)
            return {**description, "fit": brief_fit(result),
                    "difference_from_baseline": {
                        key: transform_difference(baseline[key], result[key]) for key in ("T_ET", "T_BW")},
                    "validation_prediction_change": prediction_difference(baseline, result, held)}
        except (ValueError, np.linalg.LinAlgError) as exc:
            return {**description, "error": str(exc)}

    if not skip_leave_one_out:
        for index, omitted in enumerate(train, 1):
            progress(f"逐组剔除 {index}/{len(train)}：拟合组 {omitted.id}")
            report["leave_one_fit_group_out"].append(run_trial(
                [g for g in groups if g.id != omitted.id], {"omitted_fit_group_id": omitted.id}))
    # 需要抽样时只按采集顺序等间隔选取，绝不根据验证误差选参考帧。
    indices = (np.linspace(0, len(train) - 1, max_anchor_trials, dtype=int)
               if 0 < max_anchor_trials < len(train) else range(len(train)))
    anchors = [train[index] for index in indices]
    for index, anchor in enumerate(anchors, 1):
        progress(f"参考帧试验 {index}/{len(anchors)}：拟合组 {anchor.id}")
        trial_groups = [anchor] + [g for g in groups if g.id != anchor.id]
        report["reference_anchor_trials"].append(run_trial(trial_groups, {"reference_group_id": anchor.id}))
    report["sensitivity"] = {key: sensitivity_summary(report[key])
                             for key in ("leave_one_fit_group_out", "reference_anchor_trials")}
    report["warnings"] = [
        "敏感性不是置信区间；静态残差不等于动态精度，也不是外部测量真值。",
        "所有试验使用同一批留出组；不能按留出误差挑选锚点或剔除组后仍称独立验证。",
        "T_BW 平移受 Tracker 地图原点位置影响，应同时查看固定留出点的预测变化。",
        "原版 DexUMI loss 混合米/弧度且直接相减 rotvec；分支诊断不改变其求解方法。",
        "TCP 范围可能主要来自旋转杠杆臂；需同时查看真实法兰和 Tracker 位置范围。",
    ]
    return report


def parse_ids(value):
    try:
        ids = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("请给出逗号分隔的正整数组号") from exc
    if not ids or any(item < 1 for item in ids) or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("留出组号须为互不重复的正整数")
    return set(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation-ids", type=parse_ids,
                        help="采样前指定的独立验证组号，如 4,8,12,16")
    parser.add_argument("--calibration", type=Path,
                        help="沿用本次候选矩阵的fit/held组；排除返回和过渡检查，避免误混入拟合")
    parser.add_argument("--tracker-quaternion", choices=["legacy-wzyx", "raw-xyzw"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-anchor-trials", type=int, default=0, help="0=全部拟合组；否则按采样顺序均匀选取")
    parser.add_argument("--skip-leave-one-out", action="store_true")
    args = parser.parse_args()
    source = args.input.resolve()
    metadata, groups, rejected, digest = load_samples(source)
    if args.tracker_quaternion == "raw-xyzw" and metadata.get("tracker_transport", {}).get("kind") != "passive_hidraw":
        parser.error("raw-xyzw 只用于已记录 HID 原始报文的 passive_hidraw 会话")
    offset = metadata.get("robot_kinematics", {}).get("tcp_offset_m")
    if offset is None:
        parser.error("缺少明确的 SDK TCP 偏移，不能猜测法兰转换")
    groups = [PoseGroup(g.id, g.label, g.robot,
                        tracker_pose_for_calibration(g.tracker, args.tracker_quaternion), g.statuses, g.pair_count)
              for g in groups]
    if args.calibration:
        calibration = json.loads(args.calibration.read_text())
        if (calibration.get("input_sha256") != digest
                or calibration.get("tracker_quaternion_convention") != args.tracker_quaternion):
            parser.error("候选矩阵的样本校验值或四元数约定不匹配")
        held = set(calibration["validation_group_ids"])
        fit = set(calibration["fit_group_ids"])
        if fit & held or not (fit | held) <= {g.id for g in groups}:
            parser.error("候选矩阵的fit/held分组重叠或包含不可用数据")
        if args.validation_ids is not None and args.validation_ids != held:
            parser.error("validation-ids 与候选矩阵预留分组不同")
        args.validation_ids = held
        groups = [g for g in groups if g.id in fit | held]
    if args.validation_ids is None:
        parser.error("需要 --validation-ids 或 --calibration，不能自动挑留出组")
    output = (args.output.resolve() if args.output else source.parent / "audits" /
              ("sampling_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8] + ".json"))
    if output.exists():
        parser.error("输出文件已存在；请换新路径以保留之前的审计结果")
    report = audit(groups, offset, args.validation_ids, args.max_anchor_trials, args.skip_leave_one_out,
                   progress=lambda text: print(text, flush=True))
    report.update(input_file=str(source), input_sha256=digest, session=metadata,
                  rejected_groups=rejected, tracker_quaternion_convention=args.tracker_quaternion,
                  applied_to_replay=False, hardware_connected=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print("离线审计已保存：" + str(output))


if __name__ == "__main__":
    main()
