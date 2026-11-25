"""从水平小臂起点规划广域左腕标定；纯离线，不连接机器人或 Tracker。

四个拟合位置、三个完全留出位置；各位置保持零TCP法兰中心进行多轴
旋转，并保留共同参考朝向，用于隔离平移尺度与安装杆臂项。
"""
from __future__ import annotations

import math
import itertools
import numpy as np
from scipy.spatial.transform import Rotation

from tron2_large_calibration_plan import _flange, _solve, _check_path, _disable_removed_graspers


def _path_geometry(model, start, target):
    n = max(2, 1+int(math.ceil(np.max(abs(target[:7]-start[:7]))/math.radians(.5))))
    rows = []
    for alpha in np.linspace(0, 1, n):
        q = start + (target-start)*alpha
        p, r = _flange(model, q)
        base = model.data.xpos[model.base]
        elbow = model.data.xpos[model.model.body("elbow_L_Link").id]-base
        wrist_yaw = model.data.xpos[model.model.body("wrist_yaw_L_Link").id]-base
        rows.append(np.r_[p, elbow, wrist_yaw])
    return np.array(rows)


def _bounded_path(model, start, target, lower, upper):
    reason = _check_path(model, start, target, lower, upper)
    if reason:
        return reason
    path = _path_geometry(model, start, target)
    if path[:, 2].min() < -.10:
        return "flange path goes below base Z=-0.10m"
    if path[:, 0].min() < .25 or path[:, 1].min() < .05:
        return "flange path approaches torso region (X<0.25m or Y<0.05m)"
    return None


def _order_center_visits(model, targets, groups, lower, upper):
    """仅重排同中心的已求解姿态；共同 r0 始终第一，逐条检查新连线。"""
    targets, groups = list(targets), list(groups)
    i = 0
    while i < len(groups):
        if groups[i]["role"] not in ("fit", "held_out"):
            i += 1
            continue
        j = i+1
        while j < len(groups) and groups[j]["center"] == groups[i]["center"] and groups[j]["role"] == groups[i]["role"]:
            j += 1
        block_q, block_g = targets[i:j], groups[i:j]
        after = targets[j] if j < len(targets) else None

        def segments(a, b):
            return math.ceil(float(np.max(abs(a-b)))/math.radians(5))

        choices = []
        for tail in itertools.permutations(range(1, len(block_q))):
            order = (0,)+tail
            cost = sum(segments(block_q[a], block_q[b]) for a, b in zip(order[:-1], order[1:]))
            if after is not None:
                cost += segments(block_q[order[-1]], after)
            choices.append((cost, order))
        for cost, order in sorted(choices):
            route = [block_q[k] for k in order]
            if after is not None:
                route.append(after)
            if all(_bounded_path(model, a, b, lower, upper) is None for a, b in zip(route[:-1], route[1:])):
                targets[i:j] = [block_q[k] for k in order]
                groups[i:j] = [block_g[k] for k in order]
                break
        else:
            raise ValueError(f"中心{groups[i]['center']}没有完整通过路径检查的访问次序")
        i = j
    return targets, groups


def build_broad_plan(model, start):
    """返回 targets, stages, report；索引均为从零开始的目标/采样索引。"""
    start = np.asarray(start, float).copy()
    model.check_joints(start)
    removed = _disable_removed_graspers(model)
    lower, upper = model.lower.copy(), model.upper.copy()
    lower[[5, 6, 12, 13]] = np.maximum(lower[[5, 6, 12, 13]], -.7505)
    upper[[5, 6, 12, 13]] = np.minimum(upper[[5, 6, 12, 13]], .7505)
    if np.any(start < lower) or np.any(start > upper):
        raise ValueError("起点超出模型/SDK/MoveJ 限位交集")
    start_p, r0 = _flange(model, start)
    p0 = start_p.copy()
    calibration_start = start.copy()
    needs_lift = bool(start_p[2] < -.10)
    if needs_lift:
        p0[2] = -.08
        calibration_start = _solve(model, p0, r0, start, lower, upper)
        if calibration_start is None:
            raise ValueError("无法保持初始腕朝向抬升到标定高度")
        reason = _check_path(model, start, calibration_start, lower, upper)
        lift_path = _path_geometry(model, start, calibration_start)
        if reason or lift_path[:, 2].min() < start_p[2]-.005:
            raise ValueError(f"独立抬升过渡未通过：{reason or '腕部下探超过5mm'}")
    # 每个留出位置整组排除于拟合，穿插在不同空间区域，而不是只留一个角落。
    centers = [
        ([0., 0., 0.], "fit", 6),
        ([-.13, -.12, .10], "fit", 6),
        ([-.09, -.04, .15], "held_out", 3),
        ([-.10, .20, .10], "fit", 6),
        ([-.05, .12, .14], "held_out", 3),
        ([.05, -.11, .12], "fit", 6),
        ([.03, .08, .08], "held_out", 3),
    ]
    if needs_lift:
        # 用户恢复了较低的原准备姿态：先原地抬升，再向前/左/上展开。
        centers = [([0., 0., 0.], "fit", 6),
                   ([.14, -.10, .10], "fit", 6),
                   ([.10, -.03, .18], "held_out", 3),
                   ([.12, .21, .14], "fit", 6),
                   ([.18, .12, .20], "held_out", 3),
                   ([.22, -.08, .04], "fit", 6),
                   ([.15, .06, .13], "held_out", 3)]
    rotvecs = np.array([[0, 0, 0], [35, 0, 0], [-35, 0, 0],
                       [0, 35, 0], [0, -35, 0], [0, 0, 35], [0, 0, -35],
                       [25, 25, 0], [-25, 25, 0], [25, -25, 0], [-25, -25, 0],
                       [0, 25, 25], [0, -25, -25], [25, 0, 25], [-25, 0, -25],
                       [40, 0, 0], [-40, 0, 0], [0, 40, 0], [0, -40, 0],
                       [0, 0, 40], [0, 0, -40]], float)
    targets, groups, rejected, same_orientation = [], [], [], []
    if needs_lift:
        targets.append(start.copy())
        groups.append({"center": -1, "center_offset_m": (start_p-p0).tolist(),
                       "rotation_vector_deg": [0., 0., 0.], "role": "transit_check"})
        same_orientation.append(0)
    previous = calibration_start.copy()
    for center_index, (offset, role, count) in enumerate(centers):
        offset = np.array(offset)
        selected = 0
        for rv in rotvecs:
            desired_r = r0 @ Rotation.from_rotvec(np.deg2rad(rv)).as_matrix()
            target = _solve(model, p0+offset, desired_r, previous, lower, upper)
            reason = "IK joint margin / full pose tolerance infeasible" if target is None else _bounded_path(model, previous, target, lower, upper)
            if reason:
                if not np.any(rv):
                    raise ValueError(f"空间中心 {center_index} 无法保持共同 r0 朝向：{reason}")
                rejected.append({"center": center_index, "rotation_vector_deg": rv.tolist(), "reason": reason})
                continue
            if not np.any(rv):
                same_orientation.append(len(targets))
            targets.append(target)
            groups.append({"center": center_index, "center_offset_m": offset.tolist(),
                           "rotation_vector_deg": rv.tolist(), "role": role})
            previous = target
            selected += 1
            if selected == count:
                break
        if selected < count:
            raise ValueError(f"中心 {center_index} 只找到 {selected}/{count} 个完整可达姿态，未生成计划：{rejected}")
    error = _bounded_path(model, previous, calibration_start, lower, upper)
    if error:
        raise ValueError(f"返回起点的路径不满足要求：{error}")
    same_orientation.append(len(targets))
    targets.append(calibration_start.copy())
    groups.append({"center": 0, "center_offset_m": [0., 0., 0.],
                   "rotation_vector_deg": [0., 0., 0.], "role": "return_check"})
    if needs_lift:
        # 完全反向复用已检查的抬升关节路径，不另找下降捷径。
        same_orientation.append(len(targets))
        targets.append(start.copy())
        groups.append({"center": -1, "center_offset_m": (start_p-p0).tolist(),
                       "rotation_vector_deg": [0., 0., 0.], "role": "return_check"})
    targets, groups = _order_center_visits(model, targets, groups, lower, upper)
    same_orientation = [i for i, g in enumerate(groups) if not np.any(g["rotation_vector_deg"])]
    stages, path_rows, previous = [], [], start.copy()
    calibration_path_rows = []
    for index, target in enumerate(targets):
        # 对最终实际 assembled targets 再统一验收，包含真正首个高位目标和归位。
        reason = _check_path(model, previous, target, lower, upper)
        if reason:
            raise ValueError(f"最终路径第{index}段自碰撞/限位检查失败：{reason}")
        if np.minimum(target[:7]-lower[:7], upper[:7]-target[:7]).min() < math.radians(5)-1e-8:
            raise ValueError(f"最终路径第{index}段关节余量不足5°")
        actual_path = _path_geometry(model, previous, target)
        transit = needs_lift and (index <= 1 or index == len(targets)-1)
        minimum_z = start_p[2]-.005 if transit else -.10
        if actual_path[:, 2].min() < minimum_z:
            raise ValueError(f"最终路径第{index}段低于其高度下界")
        if not transit:
            calibration_path_rows.extend(actual_path)
        path_rows.extend(actual_path)
        n = int(math.ceil(np.max(abs(target-previous))/math.radians(5)))
        stages.append([previous+(target-previous)*(i+1)/n for i in range(n)])
        previous = target
    targets, path_rows = np.asarray(targets), np.asarray(path_rows)
    calibration_path_rows = np.asarray(calibration_path_rows)
    pr = [_flange(model, q) for q in targets]
    pos = np.array([p for p, r in pr])
    orientations = [np.rad2deg(Rotation.from_matrix(a @ b.T).magnitude()) for _, a in pr for _, b in pr]
    margins = np.rad2deg(np.minimum(targets[:, :7]-lower[:7], upper[:7]-targets[:, :7]))
    fit = [i for i, g in enumerate(groups) if g["role"] == "fit"]
    hold = [i for i, g in enumerate(groups) if g["role"] == "held_out"]
    calibration_pos = pos[fit+hold]
    fit_rotvecs = np.array([groups[i]["rotation_vector_deg"] for i in fit])
    axis_counts = {name: int(np.sum((np.count_nonzero(fit_rotvecs, axis=1) == 1)
                                  & (abs(fit_rotvecs[:, axis]) >= 30)))
                   for axis, name in enumerate(("x", "y", "z"))}
    if min(axis_counts.values()) == 0:
        raise ValueError("拟合组缺少至少一个独立旋转轴，不生成标定计划")
    report = {
        "start_q_rad": start.tolist(), "target_q_rad": targets.tolist(), "groups": groups,
        "sample_count": len(targets), "sample_indices": list(range(len(targets))),
        "fit_group_indices": fit, "held_out_group_indices": hold,
        "return_check_group_indices": [i for i, g in enumerate(groups) if g["role"] == "return_check"],
        "same_orientation_group_indices": same_orientation,
        "transit_check_group_indices": [i for i, g in enumerate(groups) if g["role"] == "transit_check"],
        "calibration_origin_flange_m": p0.tolist(), "calibration_start_q_rad": calibration_start.tolist(),
        "initial_lift_required": needs_lift,
        "held_out_rule": "三个空间位置的全部姿态(中心2/4/6)从拟合中排除；返回起点仅检查重复性。",
        "method": "DexUMI zero-TCP fixed-center rotations extended across seven spatial centers",
        "center_offsets_m": [o for o, role, n in centers], "flange_positions_m": pos.tolist(),
        "flange_xyz_span_m": np.ptp(calibration_pos, axis=0).tolist(),
        "flange_max_pair_distance_m": float(np.max(np.linalg.norm(calibration_pos[:, None]-calibration_pos, axis=2))),
        "including_transit_flange_xyz_span_m": np.ptp(pos, axis=0).tolist(),
        "maximum_pair_orientation_deg": float(max(orientations)),
        "fit_pure_rotation_axis_sample_counts": axis_counts,
        "fit_rotation_vector_min_deg": fit_rotvecs.min(axis=0).tolist(),
        "fit_rotation_vector_max_deg": fit_rotvecs.max(axis=0).tolist(),
        "fit_rotation_vector_rank": int(np.linalg.matrix_rank(fit_rotvecs)),
        "left_joint_span_deg": np.rad2deg(np.ptp(targets[:, :7], axis=0)).tolist(),
        "minimum_left_joint_margin_deg": margins.min(axis=0).tolist(),
        "minimum_wrist_joint_margin_deg": float(margins[:, 4:7].min()),
        "path_flange_min_m": path_rows[:, :3].min(axis=0).tolist(),
        "path_flange_max_m": path_rows[:, :3].max(axis=0).tolist(),
        "path_elbow_min_m": path_rows[:, 3:6].min(axis=0).tolist(),
        "path_elbow_max_m": path_rows[:, 3:6].max(axis=0).tolist(),
        "path_forearm_min_z_m": float(path_rows[:, [2, 5, 8]].min()),
        "calibration_path_flange_min_m": calibration_path_rows[:, :3].min(axis=0).tolist(),
        "calibration_path_flange_max_m": calibration_path_rows[:, :3].max(axis=0).tolist(),
        "calibration_path_elbow_min_m": calibration_path_rows[:, 3:6].min(axis=0).tolist(),
        "calibration_path_elbow_max_m": calibration_path_rows[:, 3:6].max(axis=0).tolist(),
        "path_height_note": "-0.10m只约束高位标定阶段的法兰；起始肘低于此值，肘/小臂另报告并按模型碰撞检查。",
        "maximum_step_deg": 5, "stage_duration_seconds": 2., "path_collision_resolution_deg": .5,
        "segment_count": sum(map(len, stages)), "model_self_collision_passed": True,
        "right_and_head_fixed": bool(np.array_equal(targets[:, 7:], np.tile(start[7:], (len(targets), 1)))),
        "removed_model_geometry": removed, "missing_geometry": ["Tracker及支架", "线缆", "桌面及外部环境"],
        "rejected_candidates": rejected,
        "tracker_coverage_note": "实际 Tracker 空间跨度须由采集测量；报告为真实零 TCP 法兰范围。",
    }
    if report["flange_max_pair_distance_m"] < .30 or report["maximum_pair_orientation_deg"] < 60:
        raise ValueError("计划未达到30cm/60°覆盖要求，不生成运动计划")
    return targets, stages, report
