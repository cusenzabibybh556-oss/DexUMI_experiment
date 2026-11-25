"""冻结标定后执行的全新姿态验收规划；本模块仅离线计算，不连接硬件。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from tron2_large_calibration_plan import _flange, _solve, _check_path, _disable_removed_graspers
from tron2_broad_calibration_plan import _bounded_path, _path_geometry


REFERENCE_REPORT = Path(__file__).parent / "tracker_calibrations/broad_20261027_230150_0285f444/plan.json"


def build_validation_plan(model, start, reference_q):
    """返回 targets, stages, report；validation_target_indices 仅包含9个新姿态。

    朝向固定参照 broad 原始起点；当前起点只用于过渡与最终原样返回。
    起点、抬升和返回也采静态组，但不得加入拟合或新姿态精度评价。
    """
    start, reference_q = np.asarray(start, float), np.asarray(reference_q, float)
    model.check_joints(start)
    model.check_joints(reference_q)
    baseline = json.loads(REFERENCE_REPORT.read_text())
    if not np.allclose(reference_q, baseline["start_q_rad"], atol=1e-8, rtol=0):
        raise ValueError("reference_q 必须与刚完成 broad 计划的参考起点一致")
    removed = _disable_removed_graspers(model)
    lower, upper = model.lower.copy(), model.upper.copy()
    lower[[5, 6, 12, 13]] = np.maximum(lower[[5, 6, 12, 13]], -.7505)
    upper[[5, 6, 12, 13]] = np.minimum(upper[[5, 6, 12, 13]], .7505)
    if np.any(start < lower) or np.any(start > upper):
        raise ValueError("当前起点超出模型/SDK/MoveJ 限位交集")
    r0 = _flange(model, reference_q)[1]
    start_p, start_r = _flange(model, start)
    lift_position = start_p.copy()
    lift_position[2] = -.08
    lift = _solve(model, lift_position, start_r, start, lower, upper)
    if lift is None:
        raise ValueError("不能保持当前朝向抬升至验收高度")
    error = _check_path(model, start, lift, lower, upper)
    lift_path = _path_geometry(model, start, lift)
    if error or lift_path[:, 2].min() < start_p[2]-.005:
        raise ValueError(f"抬升路径不合格：{error or '下降超过5mm'}")
    centers = [[.35, .16, -.01], [.46, .32, .07], [.44, .13, -.015]]
    rotations = [
        [[20, -15, 10], [-15, -10, 15], [10, -15, -20]],
        [[20, -15, 10], [-20, 10, 15], [10, 20, -20]],
        [[20, -15, 10], [-20, 10, 15], [10, 20, -20]],
    ]
    goals, metadata, previous = [], [], lift.copy()
    for center_index, (position, rotvecs) in enumerate(zip(centers, rotations)):
        for rv in rotvecs:
            target_r = r0 @ Rotation.from_rotvec(np.deg2rad(rv)).as_matrix()
            q = _solve(model, np.asarray(position), target_r, previous, lower, upper)
            if q is None:
                raise ValueError(f"新验收中心{center_index}朝向{rv}不可达；不裁剪或放宽限位")
            goals.append(q)
            metadata.append({"role": "validation", "center": center_index,
                             "flange_position_base_m": position, "rotation_vector_deg": rv})
            previous = q
    old_q = np.asarray(baseline["target_q_rad"])
    envelope_lower, envelope_upper = old_q[:, :7].min(0), old_q[:, :7].max(0)
    new_q = np.array([start, lift]+goals)
    if (np.any(new_q[:, :7] < envelope_lower-math.radians(.5))
            or np.any(new_q[:, :7] > envelope_upper+math.radians(.5))):
        raise ValueError("新验收目标超出上一轮左臂关节包络，不扩大扫掠范围")

    # 只优化既定9个目标的访问次序，逐边验证；位姿本身不随排序变化。
    nodes = [lift]+goals
    costs = np.full((10, 10), np.inf)
    for a in range(10):
        for b in range(a+1, 10):
            if _bounded_path(model, nodes[a], nodes[b], lower, upper) is not None:
                continue
            path = _path_geometry(model, nodes[a], nodes[b])
            low, high = np.array(baseline["calibration_path_flange_min_m"]), np.array(baseline["calibration_path_flange_max_m"])
            if np.any(path[:, :3].min(0) < low-.005) or np.any(path[:, :3].max(0) > high+.005):
                continue
            n = math.ceil(float(np.max(abs(nodes[a]-nodes[b])))/math.radians(5))
            costs[a, b] = costs[b, a] = n
    # Held–Karp：全局最少分段，起点/终点均为同一个已验证的高位过渡姿态。
    dp, parent = {}, {}
    for j in range(9):
        if np.isfinite(costs[0, j+1]):
            dp[(1 << j, j)] = costs[0, j+1]
    for mask in range(1, 1 << 9):
        for last in range(9):
            key = (mask, last)
            if key not in dp:
                continue
            for nxt in range(9):
                if mask & (1 << nxt):
                    continue
                new_key = (mask | (1 << nxt), nxt)
                value = dp[key]+costs[last+1, nxt+1]
                if np.isfinite(value) and value < dp.get(new_key, np.inf):
                    dp[new_key], parent[new_key] = value, key
    completed = [(dp[(511, j)]+costs[j+1, 0], j) for j in range(9) if (511, j) in dp]
    if not completed or not np.isfinite(min(completed)[0]):
        raise ValueError("9个新姿态之间不存在完整通过路径检查的访问路线")
    _, last = min(completed)
    order, key = [], (511, last)
    while True:
        order.append(key[1])
        if key not in parent:
            break
        key = parent[key]
    order.reverse()
    targets = [start.copy(), lift.copy()] + [goals[i] for i in order] + [lift.copy(), start.copy()]
    groups = [{"role": "start_check"}, {"role": "transit_check"}] + [metadata[i] for i in order] + [{"role": "return_check"}, {"role": "return_check"}]
    previous, stages, dense_rows = start.copy(), [], []
    for i, target in enumerate(targets):
        error = _check_path(model, previous, target, lower, upper)
        if error:
            raise ValueError(f"最终实际第{i}段路径不合格：{error}")
        if np.minimum(target[:7]-lower[:7], upper[:7]-target[:7]).min() < math.radians(5)-1e-8:
            raise ValueError(f"最终第{i}目标关节余量不足5°")
        path = _path_geometry(model, previous, target)
        transit = i <= 1 or i == len(targets)-1
        if path[:, 2].min() < (start_p[2]-.005 if transit else -.10):
            raise ValueError(f"最终实际第{i}段路径低于高度约束")
        dense_rows.extend(path)
        n = int(math.ceil(np.max(abs(target-previous))/math.radians(5)))
        stages.append([previous+(target-previous)*(j+1)/n for j in range(n)])
        previous = target
    targets, dense_rows = np.asarray(targets), np.asarray(dense_rows)
    margin_deg = np.rad2deg(np.minimum(targets[:, :7]-lower[:7], upper[:7]-targets[:, :7]))
    report = {
        "purpose": "冻结候选参数后的9个全新位姿验收，不参与任何拟合或选模",
        "reference_report": str(REFERENCE_REPORT), "orientation_reference_q_rad": reference_q.tolist(),
        "start_q_rad": start.tolist(), "target_q_rad": targets.tolist(), "groups": groups,
        "sample_count": len(targets), "sample_indices": list(range(len(targets))),
        "validation_target_indices": list(range(2, 11)), "fit_group_indices": [],
        "held_out_group_indices": list(range(2, 11)), "return_check_target_indices": [11, 12],
        "center_positions_base_m": centers, "validation_visit_order": order,
        "maximum_step_deg": 5., "stage_duration_seconds": 2., "path_collision_resolution_deg": .5,
        "segment_count": sum(map(len, stages)), "planned_motion_seconds": 2*sum(map(len, stages)),
        "minimum_left_joint_margin_deg": margin_deg.min(0).tolist(),
        "path_flange_min_m": dense_rows[:, :3].min(0).tolist(), "path_flange_max_m": dense_rows[:, :3].max(0).tolist(),
        "path_elbow_min_m": dense_rows[:, 3:6].min(0).tolist(), "path_elbow_max_m": dense_rows[:, 3:6].max(0).tolist(),
        "previous_joint_envelope_passed": True, "previous_flange_envelope_passed": True,
        "envelope_note": "检查上一轮关节范围及法兰空间包围盒，不声称证明精确外部扫掠体积。",
        "right_and_head_fixed": bool(np.array_equal(targets[:, 7:], np.tile(start[7:], (len(targets), 1)))),
        "model_self_collision_passed": True, "removed_model_geometry": removed,
        "missing_geometry": ["Tracker及支架", "线缆", "桌面及外部环境"],
    }
    return targets, stages, report
