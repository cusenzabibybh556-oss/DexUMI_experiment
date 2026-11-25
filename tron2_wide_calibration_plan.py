"""仅离线生成 TRON2 左臂多中心标定轨迹；本模块没有硬件连接和控制接口。

保持每个空间中心的零 TCP 法兰位置，改变完整末端朝向；在多个中心
重复这一过程。关节限位使用模型、SDK 和 MoveJ 文档范围的交集。
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
            model.model.geom_contype[g] = 0
            model.model.geom_conaffinity[g] = 0
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
    # 每 0.5° 检查一次整段关节插值，包含起点和终点。
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
    """连续 IK：误差足够小时才接受；不裁剪求解结果或运动命令。"""
    margin = math.radians(5)
    lo, hi = lower[:7]+margin, upper[:7]-margin
    start = previous[:7].copy()
    if np.any(start < lo) or np.any(start > hi):
        return None

    def residual(left):
        q = previous.copy()
        q[:7] = left
        p, r = _flange(model, q)
        rv = Rotation.from_matrix(rotation @ r.T).as_rotvec()
        # 微弱连续性项用于选择冗余解，不允许牺牲下面的位姿验收精度。
        return np.r_[100*(p-position), 3*rv, .004*(left-start)]

    result = least_squares(residual, start, bounds=(lo, hi), max_nfev=160,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    q = previous.copy()
    q[:7] = result.x
    p, r = _flange(model, q)
    pe = float(np.linalg.norm(p-position))
    re = float(np.linalg.norm(Rotation.from_matrix(rotation @ r.T).as_rotvec()))
    if pe > .0003 or re > math.radians(.15):
        return None
    return q


def build_wide_plan(model, start):
    """返回 (targets, stages, report)；每个 target 都采样，含起点及最终归位。

    stages[i] 是到 targets[i] 的 MoveJ 子目标数组，每段最大关节变化 5°。
    report 的 fit_group_indices / held_out_group_indices 为从 0 开始的样本索引。
    不修改起始右臂、头部目标。规划失败时抛出异常，不返回部分运动计划。
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
    offsets = np.array([[0, 0, 0], [-.065, .065, -.040],
                        [.065, .045, .040], [.035, -.025, -.095]])
    # 依次尝试独立轴及混合轴旋转，失败目标不裁剪。每中心取八个可达朝向。
    rotations_deg = np.array([[0, 0, 0], [20, 0, 0], [-20, 0, 0],
                              [0, 20, 0], [0, -20, 0], [0, 0, 20],
                              [0, 0, -20], [16, 16, 0], [-16, 16, 0],
                              [16, -16, 0], [-16, -16, 0], [0, 16, 16],
                              [0, -16, -16], [16, 0, 16], [-16, 0, -16],
                              [0, 0, 30], [30, 0, 0], [0, 30, 0],
                              [0, 0, -30], [-30, 0, 0], [0, -30, 0]], float)
    targets, group_info = [], []
    previous = start.copy()
    rejected = []
    for center_index, offset in enumerate(offsets):
        selected = 0
        for rv in rotations_deg:
            desired_r = r0 @ Rotation.from_rotvec(np.deg2rad(rv)).as_matrix()
            target = _solve(model, p0+offset, desired_r, previous, lower, upper)
            reason = "IK pose tolerance or joint-margin infeasible" if target is None else _check_path(model, previous, target, lower, upper)
            if reason:
                rejected.append({"center": center_index, "rotation_vector_deg": rv.tolist(), "reason": reason})
                continue
            targets.append(target)
            group_info.append({"center": center_index, "center_offset_m": offset.tolist(),
                               "rotation_vector_deg": rv.tolist(),
                               "role": "held_out" if center_index == 3 else "fit"})
            previous = target
            selected += 1
            if selected == 8:
                break
        if selected < 8:
            raise ValueError(f"空间中心 {center_index} 仅找到 {selected}/8 个连续可达朝向；未生成执行计划。拒绝原因：{rejected}")
    # 单独检查直接返回路径；若不通过，沿原路径逆序返回。
    return_targets = ([start.copy()] if _check_path(model, previous, start, lower, upper) is None
                      else list(reversed(targets[:-1])) + [start.copy()])
    for target in return_targets:
        reason = _check_path(model, previous, target, lower, upper)
        if reason:
            raise ValueError(f"返回路径失败：{reason}")
        targets.append(target.copy())
        group_info.append({"role": "return", "sample_for_fit": False})
        previous = target
    stages, previous = [], start.copy()
    for target in targets:
        n = int(math.ceil(np.max(abs(target-previous))/math.radians(5)))
        stages.append([previous+(target-previous)*(j+1)/n for j in range(n)])
        previous = target
    targets = np.asarray(targets)
    fit = [i for i, g in enumerate(group_info) if g["role"] == "fit"]
    hold = [i for i, g in enumerate(group_info) if g["role"] == "held_out"]
    p_and_r = [_flange(model, q) for q in targets[:32]]
    positions = np.array([p for p, _ in p_and_r])
    sdk_positions = np.array([model.pose(q)[0] for q in targets[:32]])
    max_rotation = max(Rotation.from_matrix(a @ b.T).magnitude() for _, a in p_and_r for _, b in p_and_r)
    margins = np.rad2deg(np.minimum(targets[:, :7]-lower[:7], upper[:7]-targets[:, :7]))
    report = {
        "start_q_rad": start.tolist(), "target_q_rad": targets.tolist(), "groups": group_info,
        "fit_group_indices": fit, "held_out_group_indices": hold,
        "method": "DexUMI zero-TCP fixed-position orientation sampling extended to four spatial centers",
        "flange_positions_m": positions.tolist(), "sdk_tcp_positions_m": sdk_positions.tolist(),
        "flange_xyz_span_m": np.ptp(positions, axis=0).tolist(),
        "flange_max_pair_distance_m": float(np.max(np.linalg.norm(positions[:, None]-positions, axis=2))),
        "sdk_tcp_xyz_span_m": np.ptp(sdk_positions, axis=0).tolist(),
        "maximum_pair_orientation_deg": float(np.rad2deg(max_rotation)),
        "minimum_left_joint_margin_deg": margins.min(axis=0).tolist(),
        "minimum_wrist_joint_margin_deg": float(margins[:, 4:7].min()),
        "maximum_step_deg": 5, "path_collision_resolution_deg": .5,
        "segment_count": sum(map(len, stages)), "model_self_collision_passed": True,
        "right_and_head_fixed": bool(np.array_equal(targets[:, 7:], np.tile(start[7:], (len(targets), 1)))),
        "removed_model_geometry": removed,
        "missing_geometry": ["Tracker及支架", "线缆", "桌面及外部环境"],
        "rejected_candidates": rejected,
        "tracker_coverage_note": "Tracker 实际空间跨度需由采集结果计算；此处报告零TCP法兰跨度，不能用虚拟TCP圆弧替代。",
    }
    return targets, stages, report
