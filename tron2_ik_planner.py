"""在有限候选集合中选择关节限位余量最大的完整左臂轨迹。

只做模型运动学计算。这里的 q 不能作为 ServoP 的分支选择指令；不连接硬件，
不改准备姿态，不缩放输入位姿，不保证全局最优、碰撞安全或固件可跟踪性。
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MarginPlanOptions:
    # 属于本地规划参数，不是厂家发布的关节速度限制。
    max_joint_speed: float = 0.5
    min_margin_deg: float = 0.0
    knot_interval: float = 0.1
    position_tolerance: float = 0.0001
    angle_tolerance_deg: float = 0.05
    # 不同运动正则权重生成不同连续候选；最终按整段最差余量排序。
    motion_weights: tuple = (0.0, 0.1, 1.0, 10.0)

    def validate(self):
        for name in ("max_joint_speed", "knot_interval", "position_tolerance",
                     "angle_tolerance_deg"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"IK 参数 {name} 必须为有限正数")
        if not math.isfinite(self.min_margin_deg) or self.min_margin_deg < 0:
            raise ValueError("IK 最小限位余量必须为有限非负数")
        if not self.motion_weights or any(not math.isfinite(w) or w < 0
                                          for w in self.motion_weights):
            raise ValueError("IK 候选权重必须为非空的有限非负数集合")


def joint_path_metrics(plan, times, lower, upper):
    """余量取所有时间、所有运动关节的最小值，不能用平均值掩盖局部贴限。"""
    q = np.asarray(plan, dtype=float)
    times = np.asarray(times, dtype=float)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if (q.ndim != 2 or q.shape[1] != len(lower) or q.shape != (len(times), len(upper))
            or len(q) < 2 or not np.isfinite(q).all() or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower >= upper)):
        raise ValueError("IK 候选轨迹、限位或时间轴无效")
    margins = np.minimum(q - lower, upper - q)
    frame, joint = np.unravel_index(int(np.argmin(margins)), margins.shape)
    velocity = np.diff(q, axis=0) / np.diff(times)[:, None]
    dt = np.diff(times)
    # 相同余量时，比较积分速度平方；不是用该代价交换限位余量。
    energy = float(np.sum(velocity**2 * dt[:, None]))
    acceleration_energy = 0.0
    if len(velocity) > 1:
        mid_dt = (dt[1:] + dt[:-1]) / 2
        acceleration = np.diff(velocity, axis=0) / mid_dt[:, None]
        acceleration_energy = float(np.sum(acceleration**2 * mid_dt[:, None]))
    return {
        "minimum_margin_deg": float(np.rad2deg(margins[frame, joint])),
        "critical_frame": int(frame), "critical_joint_index": int(joint),
        "critical_time_s": float(times[frame]),
        "max_joint_speed_rad_s": float(np.abs(velocity).max()),
        "velocity_energy": energy, "acceleration_energy": acceleration_energy,
    }


def choose_margin_candidate(candidates):
    """只接收已经完整验收的候选；最大化最小余量，再比较平滑度。"""
    scores = ("minimum_margin_deg", "acceleration_energy", "velocity_energy")
    valid = [c for c in candidates if c.get("valid") is True
             and all(math.isfinite(c.get(k, math.nan)) for k in scores)
             and c["minimum_margin_deg"] >= 0]
    if not valid:
        raise RuntimeError("没有通过整段位姿、限位和连续性检查的 IK 候选")
    return min(valid, key=lambda c: (
        -c["minimum_margin_deg"], c["acceleration_energy"], c["velocity_energy"], c["id"],
    ))


def _knots(times, interval):
    indices = [0]
    for i in range(1, len(times) - 1):
        if times[i] - times[indices[-1]] >= interval:
            indices.append(i)
    if indices[-1] != len(times) - 1:
        indices.append(len(times) - 1)
    return np.asarray(indices)


def _pose_error(ik, q, target_p, target_r, Rotation):
    p, r, _ = ik._forward(q)
    # 使用 SO(3) 对数，避免叉积误差在 180 度时误判为零。
    return np.r_[p - target_p, 0.25 * Rotation.from_matrix(target_r @ r.T).as_rotvec()]


def _coarse_candidate(ik, start, ps, rs, times, indices, weight, options, minimize, Rotation):
    q = start.copy()
    knots = [q.copy()]
    for left, right in zip(indices[:-1], indices[1:]):
        dt = float(times[right] - times[left])
        previous = q[:7].copy()
        # 留少量速度余量给中间帧的位姿投影；不改变用户指定的时间轴。
        step = 0.95 * options.max_joint_speed * dt
        lo = np.maximum(ik.lower[:7] + 1e-7, previous - step)
        hi = np.minimum(ik.upper[:7] - 1e-7, previous + step)

        def equality(x):
            trial = q.copy()
            trial[:7] = x[:7]
            return _pose_error(ik, trial, ps[right], rs[right], Rotation)

        def inequalities(x):
            return np.r_[x[:7] - ik.lower[:7] - x[7],
                         ik.upper[:7] - x[:7] - x[7]]

        def objective(x):
            return -x[7] + weight * np.sum((x[:7] - previous)**2) / dt

        margin = min(float(np.min(previous - ik.lower[:7])),
                     float(np.min(ik.upper[:7] - previous)))
        x0 = np.r_[np.clip(previous, lo, hi), max(0.0, margin)]
        result = minimize(objective, x0, method="SLSQP",
                          bounds=list(zip(lo, hi)) + [(0, math.pi)],
                          constraints=[{"type": "eq", "fun": equality},
                                       {"type": "ineq", "fun": inequalities}],
                          options={"maxiter": 100, "ftol": 1e-10})
        error = equality(result.x)
        if (not np.isfinite(result.x).all() or np.min(inequalities(result.x)) < -1e-7
                or np.linalg.norm(error[:3]) > options.position_tolerance
                or np.rad2deg(np.linalg.norm(error[3:]) / .25) > options.angle_tolerance_deg
                or np.any(result.x[:7] < lo - 1e-8) or np.any(result.x[:7] > hi + 1e-8)):
            raise RuntimeError(f"第 {right} 帧在限位及速度约束内未找到位姿解：{result.message}")
        q[:7] = result.x[:7]
        knots.append(q.copy())
    return np.asarray(knots)


def _dense_projection(ik, start, coarse, times, indices, ps, rs, options, Rotation):
    plan = np.tile(start, (len(times), 1))
    for j in range(7):
        plan[:, j] = np.interp(times, times[indices], coarse[:, j])
    # 关节插值不等于笛卡尔插值；逐个输出帧重新投影到原始 6D 目标。
    for i in range(1, len(times)):
        q = plan[i].copy()
        for _ in range(15):
            p, r, point = ik._forward(q)
            angular = Rotation.from_matrix(rs[i] @ r.T).as_rotvec()
            error = np.r_[ps[i] - p, .25 * angular]
            if np.linalg.norm(error[:3]) < 1e-8 and np.linalg.norm(angular) < 1e-7:
                break
            ik.mj.mj_jac(ik.model, ik.data, ik.jacp, ik.jacr, point, ik.body)
            jac = np.vstack((ik.jacp[:, ik.dadr[:7]], .25 * ik.jacr[:, ik.dadr[:7]]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-8 * np.eye(6), error)
            delta *= min(1., .025 / max(float(np.max(np.abs(delta))), 1e-12))
            q[:7] += delta
            if np.any(q[:7] <= ik.lower[:7]) or np.any(q[:7] >= ik.upper[:7]):
                raise RuntimeError(f"第 {i} 帧插值后的位姿投影触及关节限位")
        plan[i] = q
    plan[0] = start  # 首帧必须保持用户提供的关节解，不能偷偷换准备姿态。
    return plan


def plan_with_joint_margin(ik, q_start, positions, quaternions, times, options=None, progress=None):
    """搜索多个连续完整候选，按整段最差关节余量选择；失败不返回半条轨迹。"""
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    from replay_tron2_tracker import quaternion_to_matrix

    options = options or MarginPlanOptions()
    options.validate()
    start = np.asarray(q_start, dtype=float).copy()
    ik.check_joints(start)
    ps, us, ts = (np.asarray(x, dtype=float) for x in (positions, quaternions, times))
    n = len(ts)
    if (ts.ndim != 1 or n < 2 or ps.shape != (n, 3) or us.shape != (n, 4)
            or not all(np.isfinite(x).all() for x in (ps, us, ts))
            or np.any(np.diff(ts) <= 0) or np.any(np.linalg.norm(us, axis=1) < 1e-12)):
        raise ValueError("IK 位姿或严格递增的时间轴无效")
    ik.check_anchor(start, ps[0], us[0])
    rs = np.array([quaternion_to_matrix(u) for u in us])
    indices = _knots(ts, options.knot_interval)
    candidates, paths = [], {}
    for index, weight in enumerate(options.motion_weights):
        record = {"id": index, "motion_weight": weight, "valid": False}
        if progress:
            progress(f"离线 IK 候选 {index+1}/{len(options.motion_weights)}：整段求解中")
        try:
            coarse = _coarse_candidate(ik, start, ps, rs, ts, indices, weight,
                                       options, minimize, Rotation)
            plan = _dense_projection(ik, start, coarse, ts, indices, ps, rs, options, Rotation)
            if not np.array_equal(plan[:, 7:], np.tile(start[7:], (n, 1))):
                raise RuntimeError("右臂或头部目标发生非计划变化")
            metrics = joint_path_metrics(plan[:, :7], ts, ik.lower[:7], ik.upper[:7])
            if metrics["minimum_margin_deg"] < options.min_margin_deg - 1e-8:
                raise RuntimeError(f"整段最小余量 {metrics['minimum_margin_deg']:.3f}° 未达要求")
            if metrics["max_joint_speed_rad_s"] > options.max_joint_speed + 1e-6:
                raise RuntimeError(f"插值后关节速度 {metrics['max_joint_speed_rad_s']:.4f} rad/s 超限")
            errors = np.array([_pose_error(ik, q, p, r, Rotation) for q, p, r in zip(plan, ps, rs)])
            pos_error = float(np.linalg.norm(errors[:, :3], axis=1).max())
            angle_error = float(np.rad2deg(np.linalg.norm(errors[:, 3:], axis=1).max() / .25))
            if pos_error > options.position_tolerance or angle_error > options.angle_tolerance_deg:
                raise RuntimeError(f"整段位姿误差超限：{pos_error*1000:.4f} mm / {angle_error:.4f}°")
            record.update(metrics, valid=True, max_position_error_mm=pos_error*1000,
                          max_angle_error_deg=angle_error)
            paths[index] = plan
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            record["reason"] = str(exc)
        candidates.append(record)
        if progress:
            progress(f"候选 {index+1}：" + (f"最小余量 {record['minimum_margin_deg']:.3f}°"
                     if record["valid"] else record["reason"]))
    try:
        selected = choose_margin_candidate(candidates)
    except RuntimeError as exc:
        raise RuntimeError(str(exc) + "；" + "；".join(c["reason"] for c in candidates)) from exc
    from tron2_left_ik import JOINT_NAMES
    report = {
        **selected, "frames": n, "knot_count": len(indices), "candidates": candidates,
        "selected_candidate": selected["id"],
        "critical_joint": JOINT_NAMES[selected["critical_joint_index"]],
        "selection": "max_min_joint_limit_margin_then_acceleration_then_velocity",
        "search_scope": "best among generated and fully validated candidates; not a global optimum proof",
        "max_joint_speed_limit_rad_s": options.max_joint_speed,
        "required_minimum_margin_deg": options.min_margin_deg,
        "first_frame_unchanged": True, "right_and_head_unchanged": True,
        "cartesian_trajectory_changed": False, "collision_checked": False,
        "continuity_check": "finite differences at all output frames; acceleration is a tie-break cost, not a hard bound",
        "firmware_branch_enforced": False, "hardware_motion_commands_sent": 0,
    }
    return paths[selected["id"]], report
