"""TRON2 左臂大范围手眼标定的离线规划，不连接设备、不发送控制命令。

DexUMI 核心：固定零 TCP 法兰位置，变化完整朝向；扩展到三个拟合中心
和一个完全独立的验证中心。辅助运动只用于到达采样中心，不参与拟合。
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _flange(model, q):
    model._forward(q)
    return (model.data.xpos[model.body] - model.data.xpos[model.base]).copy(), model.data.xmat[model.body].reshape(3, 3).copy()


def _disable_removed_graspers(model):
    removed = []
    for g in range(model.model.ngeom):
        name = model.model.body(int(model.model.geom_bodyid[g])).name
        if name.startswith("grasper_"):
            model.model.geom_contype[g] = model.model.geom_conaffinity[g] = 0
            removed.append(name)
    return sorted(set(removed))


def _collision(model, q):
    m, d = model.model, model.data
    d.qpos[model.qadr] = q
    model.mj.mj_forward(m, d)
    for c in d.contact:
        names = [m.body(int(m.geom_bodyid[g])).name for g in (c.geom1, c.geom2)]
        if c.dist < -1e-6 and any(n.endswith("_L_Link") for n in names):
            return names, float(c.dist)
    return None


def _check_path(model, previous, target, lower, upper):
    n = max(2, 1 + int(math.ceil(np.max(abs(target[:7]-previous[:7])) / math.radians(.5))))
    for alpha in np.linspace(0, 1, n):
        q = (1-alpha)*previous + alpha*target
        if np.any(q < lower-1e-9) or np.any(q > upper+1e-9):
            return "path outside MoveJ joint limits"
        hit = _collision(model, q)
        if hit:
            return f"model collision: {hit}"
    return None


def _solve(model, position, rotation, previous, lower, upper):
    """连续 IK 保留 5° 余量；不裁剪目标、不放宽误差和关节范围。"""
    margin = math.radians(5)
    lo, hi = lower[:7]+margin, upper[:7]-margin
    seed = previous[:7].copy()
    if np.any(seed < lo-1e-10) or np.any(seed > hi+1e-10):
        return None

    def residual(left):
        q = previous.copy()
        q[:7] = left
        p, r = _flange(model, q)
        return np.r_[100*(p-position), 3*Rotation.from_matrix(rotation @ r.T).as_rotvec(),
                     .004*(left-seed)]

    result = least_squares(residual, seed, bounds=(lo, hi), max_nfev=200,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    q = previous.copy()
    q[:7] = result.x
    p, r = _flange(model, q)
    if np.linalg.norm(p-position) > .0003 or Rotation.from_matrix(rotation @ r.T).magnitude() > math.radians(.15):
        return None
    return q


def build_large_plan(model, start):
    """返回 targets, stages, report；最终归位额外采样，不参与拟合。

    sample_indices 指定 targets 中需要采样的目标。fit_group_indices 与
    held_out_group_indices 指采样后的从零开始的 group 索引。
    """
    start = np.asarray(start, float).copy()
    model.check_joints(start)
    removed = _disable_removed_graspers(model)
    lower, upper = model.lower.copy(), model.upper.copy()
    lower[[5, 6, 12, 13]] = np.maximum(lower[[5, 6, 12, 13]], -.7505)
    upper[[5, 6, 12, 13]] = np.minimum(upper[[5, 6, 12, 13]], .7505)
    if np.any(start < lower) or np.any(start > upper):
        raise ValueError("起点超出模型/SDK/MoveJ 限位交集")
    if _collision(model, start):
        raise ValueError(f"起点存在模型自碰撞：{_collision(model, start)}")
    p0, r0 = _flange(model, start)
    # 向前/向左扩展，三个拟合中心覆盖 31.6cm；独立中心同时改变 XYZ。
    offsets = np.array([[0., 0., 0.], [.18, .06, .04],
                        [.18, .24, .10], [.08, .18, .12]])
    rotations_deg = np.array([[0, 0, 0], [35, 0, 0], [-35, 0, 0],
                              [0, 35, 0], [0, -35, 0], [0, 0, 35],
                              [0, 0, -35], [25, 25, 0], [-25, 25, 0],
                              [25, -25, 0], [-25, -25, 0], [0, 25, 25],
                              [0, -25, -25], [25, 0, 25], [-25, 0, -25],
                              [40, 0, 0], [0, 40, 0], [0, 0, 40],
                              [-40, 0, 0], [0, -40, 0], [0, 0, -40]], float)
    targets, groups, rejected = [], [], []
    previous = start.copy()
    for center_index, offset in enumerate(offsets):
        selected = 0
        for rv in rotations_deg:
            desired_r = r0 @ Rotation.from_rotvec(np.deg2rad(rv)).as_matrix()
            target = _solve(model, p0+offset, desired_r, previous, lower, upper)
            reason = "IK infeasible at required pose tolerance / joint margin" if target is None else _check_path(model, previous, target, lower, upper)
            if reason:
                if not np.any(rv):
                    raise ValueError(f"中心 {center_index} 无法保持共同参考朝向，拒绝更换朝向补齐中心：{reason}")
                rejected.append({"center": center_index, "rotation_vector_deg": rv.tolist(), "reason": reason})
                continue
            targets.append(target)
            groups.append({"center": center_index, "center_offset_m": offset.tolist(),
                           "rotation_vector_deg": rv.tolist(),
                           "role": "held_out" if center_index == 3 else "fit"})
            previous = target
            selected += 1
            if selected == 5:
                break
        if selected < 5:
            raise ValueError(f"空间中心 {center_index} 只有 {selected}/5 个连续可达姿态；未生成执行计划。拒绝：{rejected}")
    error = _check_path(model, previous, start, lower, upper)
    if error:
        raise ValueError(f"直接返回起点的模型路径未通过：{error}")
    targets.append(start.copy())
    groups.append({"center": 0, "center_offset_m": [0., 0., 0.],
                   "rotation_vector_deg": [0., 0., 0.], "role": "return_check"})
    sample_count = len(targets)
    stages, previous = [], start.copy()
    for target in targets:
        n = int(math.ceil(np.max(abs(target-previous))/math.radians(5)))
        stages.append([previous+(target-previous)*(j+1)/n for j in range(n)])
        previous = target
    targets = np.asarray(targets)
    samples = targets[:sample_count]
    pr = [_flange(model, q) for q in samples]
    pos = np.array([p for p, _ in pr])
    sdk_pos = np.array([model.pose(q)[0] for q in samples])
    orientations = np.array([[np.rad2deg(Rotation.from_matrix(a @ b.T).magnitude()) for _, b in pr] for _, a in pr])
    margins = np.rad2deg(np.minimum(targets[:, :7]-lower[:7], upper[:7]-targets[:, :7]))
    joint_span = np.rad2deg(np.ptp(samples[:, :7], axis=0))
    # MoveJ 在关节空间插值；即使采样端点法兰位置相同，中间路径仍可能成弧。
    # 单独报告整段法兰包围盒，不能只用采样中心声明运动范围。
    dense_path = []
    previous = start.copy()
    for target in targets:
        n = max(2, 1 + int(math.ceil(np.max(abs(target[:7]-previous[:7])) / math.radians(.5))))
        dense_path.extend(_flange(model, previous+(target-previous)*a)[0] for a in np.linspace(0, 1, n))
        previous = target
    dense_path = np.asarray(dense_path)
    report = {
        "start_q_rad": start.tolist(), "target_q_rad": targets.tolist(), "groups": groups,
        "sample_count": sample_count, "sample_indices": list(range(sample_count)),
        "fit_group_indices": list(range(15)), "held_out_group_indices": list(range(15, 20)),
        "return_check_group_indices": [20], "same_orientation_group_indices": [0, 5, 10, 15, 20],
        "method": "DexUMI zero-TCP fixed-center orientation sampling; three fit centers and one entire held-out center",
        "center_offsets_m": offsets.tolist(), "flange_positions_m": pos.tolist(),
        "flange_xyz_span_m": np.ptp(pos, axis=0).tolist(),
        "flange_max_pair_distance_m": float(np.max(np.linalg.norm(pos[:, None]-pos, axis=2))),
        "sdk_tcp_positions_m": sdk_pos.tolist(),
        "maximum_pair_orientation_deg": float(orientations.max()),
        "left_joint_span_deg": joint_span.tolist(),
        "path_flange_min_m": dense_path.min(axis=0).tolist(),
        "path_flange_max_m": dense_path.max(axis=0).tolist(),
        "path_flange_offset_min_m": (dense_path.min(axis=0)-p0).tolist(),
        "path_flange_offset_max_m": (dense_path.max(axis=0)-p0).tolist(),
        "minimum_left_joint_margin_deg": margins.min(axis=0).tolist(),
        "minimum_wrist_joint_margin_deg": float(margins[:, 4:7].min()),
        "maximum_step_deg": 5, "path_collision_resolution_deg": .5,
        "segment_count": sum(map(len, stages)), "model_self_collision_passed": True,
        "right_and_head_fixed": bool(np.array_equal(targets[:, 7:], np.tile(start[7:], (len(targets), 1)))),
        "removed_model_geometry": removed,
        "missing_geometry": ["Tracker及支架", "线缆", "桌面及外部环境"],
        "rejected_candidates": rejected,
        "tracker_coverage_note": "实际 Tracker 空间跨度须由采集测量；不能用虚拟 SDK TCP 的旋转圆弧代替法兰平移。",
    }
    if report["flange_max_pair_distance_m"] < .30 or report["maximum_pair_orientation_deg"] < 60:
        raise ValueError("大范围计划未达到 30cm / 60° 覆盖要求，不生成执行计划")
    return targets, stages, report
