"""独立的左臂恢复路径试验；不接入 Replay，不自动批准后续自动恢复。

该入口只在现场明确确认后发分段 MoveJ，每段计划关节变化不超过 5 度。
小段指令和反馈检查都不是急停，已被机器人接受的 MoveJ 仍可能继续。
本机碰撞检查缺少实际 O6、转接件、桌面和线缆，不代表完整安全验证。
"""

from __future__ import annotations

import math

import numpy as np

import replay_tron2_fixed_start as fixed
from prepare_tron2_arms import wait_for_target


MAX_JOINT_STEP_RAD = math.radians(5)
TRIAL_JOINT_TOLERANCE_RAD = 0.015
START_CHANGE_TOLERANCE_RAD = 0.005
# A ServoP stop can leave the measured joint just beyond the conservative
# SDK/model intersection.  Recovery must be able to leave that state, but only
# along a straight path that monotonically reduces the existing violation.
MAX_RECOVERY_LIMIT_OVERRUN_RAD = math.radians(2)


def validate_recovery_start(model, q, target):
    """Allow a small pre-existing limit overrun only when the target retreats.

    This exception is deliberately local to the confirmed MoveJ restore path;
    normal IK, replay and ACT checks keep using the unmodified hard limits.
    """
    q = np.asarray(q, dtype=float)
    target = np.asarray(target, dtype=float)
    if q.shape != (16,) or target.shape != (16,):
        raise ValueError("恢复限位检查需要两个完整的 16 关节向量")
    model.check_joints(target)
    below = np.maximum(model.lower - q, 0.)
    above = np.maximum(q - model.upper, 0.)
    overrun = below + above
    invalid = np.flatnonzero(overrun > 0.)
    if not invalid.size:
        model.check_joints(q)
        return overrun, []
    maximum = float(np.max(overrun))
    if maximum > MAX_RECOVERY_LIMIT_OVERRUN_RAD:
        raise RuntimeError(
            f"当前关节超过模型限位 {math.degrees(maximum):.3f}°，大于恢复模式 "
            f"{math.degrees(MAX_RECOVERY_LIMIT_OVERRUN_RAD):.1f}° 上限；未发送 MoveJ"
        )
    for i in invalid:
        # The target is already checked inside the normal limits.  Keep this
        # explicit direction test so future target/limit changes cannot turn
        # recovery mode into permission to move farther outwards.
        if ((q[i] < model.lower[i] and target[i] <= q[i]) or
                (q[i] > model.upper[i] and target[i] >= q[i])):
            raise RuntimeError("恢复目标没有使越限关节单调退回安全区；未发送 MoveJ")
    return overrun, invalid.tolist()


def check_recovery_anchor(model, q, position, quaternion, invalid):
    """Check FK/SDK agreement without rejecting an allowed recovery start."""
    if not invalid:
        return model.check_anchor(q, position, quaternion)
    from replay_tron2_tracker import quaternion_distance_deg
    p, _, _ = model._forward(q)
    actual_quaternion = model.data.xquat[model.body].copy()
    error = float(np.linalg.norm(p - position))
    angle = float(quaternion_distance_deg(actual_quaternion, quaternion))
    if error > .002 or angle > .5:
        raise RuntimeError(
            f"本机 IK 模型与实机反馈不一致：{error*1000:.2f} mm / {angle:.2f}°；"
            "越限恢复未发送 MoveJ"
        )
    return error, angle


def plan_trial(q, pose):
    """只读/离线预检，保持右臂及头部坐标不变，不求解或发送 ServoJ。"""
    from tron2_left_ik import LeftArmIK

    q = np.asarray(q, dtype=float)
    if q.shape != (16,) or not np.isfinite(q).all():
        raise ValueError("恢复试验需要完整有效的 16 关节只读反馈")
    model = LeftArmIK()
    target = q.copy()
    target[:7] = fixed.FIXED_LEFT_Q
    start_overrun, recovery_joints = validate_recovery_start(model, q, target)
    check_recovery_anchor(model, q, pose.left_position, pose.left_quaternion,
                          recovery_joints)
    model.check_anchor(target, fixed.FIXED_LEFT_POSITION, fixed.FIXED_LEFT_QUATERNION)

    # 原厂夹爪不是实际左 O6；右侧工具已拆除。仅在内存中的离线模型
    # 排除这两组错误工具几何，不修改模型源文件或真实机器人负载。
    m, d, mj = model.model, model.data, model.mj
    left_geoms = set()
    for geom in range(m.ngeom):
        body = m.body(int(m.geom_bodyid[geom])).name
        if body.startswith("grasper_"):
            m.geom_contype[geom] = 0
            m.geom_conaffinity[geom] = 0
        elif body.endswith("_L_Link") and (m.geom_contype[geom] or m.geom_conaffinity[geom]):
            left_geoms.add(geom)
    delta = target[:7] - q[:7]
    segments = max(1, int(math.ceil(float(np.max(np.abs(delta))) / MAX_JOINT_STEP_RAD)))
    waypoints = np.tile(q, (segments, 1))
    waypoints[:, :7] += np.linspace(1 / segments, 1, segments)[:, None] * delta
    waypoints[-1, :7] = fixed.FIXED_LEFT_Q
    # Every command target must be inside the ordinary limits.  If equal
    # segmentation would leave the first recovery target slightly outside,
    # move just that joint directly inside on the first (still <=5deg) step.
    for i in recovery_joints:
        if q[i] < model.lower[i]:
            waypoints[0, i] = max(waypoints[0, i], model.lower[i] + 1e-4)
        else:
            waypoints[0, i] = min(waypoints[0, i], model.upper[i] - 1e-4)
    for waypoint in waypoints:
        model.check_joints(waypoint)
    steps = np.diff(np.vstack((q, waypoints)), axis=0)[:, :7]
    maximum_step = float(np.max(np.abs(steps)))
    if maximum_step > MAX_JOINT_STEP_RAD + 1e-9:
        raise RuntimeError("越限恢复首段超过 5°；未发送 MoveJ")

    # Check the actual piecewise MoveJ targets, including the adjusted first
    # recovery step, instead of checking only the direct start-to-end chord.
    path = np.vstack((q, waypoints))
    samples_per_segment = max(2, int(math.ceil(1001 / segments)))
    previous_overrun = start_overrun.copy()
    collision_samples = 0
    for segment in range(segments):
        for beta in np.linspace(0, 1, samples_per_segment,
                                endpoint=(segment == segments - 1)):
            if segment and beta == 0:
                continue
            sample = path[segment] + beta * (path[segment + 1] - path[segment])
            below = np.maximum(model.lower - sample, 0.)
            above = np.maximum(sample - model.upper, 0.)
            sample_overrun = below + above
            progress = (segment + beta) / segments
            if np.any(sample_overrun > previous_overrun + 1e-9):
                raise RuntimeError(
                    f"恢复试验限位偏差未单调减小：路径 {progress:.1%}；未发送 MoveJ"
                )
            previous_overrun = sample_overrun
            if not np.any(sample_overrun):
                model.check_joints(sample)
            d.qpos[model.qadr] = sample
            mj.mj_forward(m, d)
            collision_samples += 1
            for contact in d.contact:
                if (contact.dist < 0 and
                        (int(contact.geom1) in left_geoms or int(contact.geom2) in left_geoms)):
                    names = [m.body(int(m.geom_bodyid[g])).name
                             for g in (contact.geom1, contact.geom2)]
                    raise RuntimeError(
                        f"恢复试验模型碰撞：路径 {progress:.1%}，{names}；未发送 MoveJ"
                    )
    return waypoints, {
        "segments": segments,
        "maximum_planned_joint_step_deg": math.degrees(maximum_step),
        "collision_samples": collision_samples,
        "model": str(model.model_path),
        "missing_geometry": ["actual_O6", "mount", "table", "wires"],
        "actual_firmware_interpolation_verified": False,
        "recovery_start_limit_overrun_deg": math.degrees(float(np.max(start_overrun))),
        "recovery_start_joint_indices": recovery_joints,
        "recovery_rule": "pre-existing <=2deg overrun must decrease monotonically",
    }


def run_trial(client, q, dq, pose, args, trace=None):
    """需现场输入 RESTORE；不会修改正式路径验证标志或调用回放。"""
    fixed.require_validated_left_start()
    if client.accid != fixed.FIXED_ROBOT_ID:
        raise RuntimeError("机器人身份不符合保存起点，拒绝恢复试验")
    q = np.asarray(q, dtype=float).copy()
    dq = np.asarray(dq, dtype=float)
    if (q.shape != (16,) or not np.isfinite(q).all()
            or dq.shape != (16,) or not np.isfinite(dq).all()
            or float(np.max(np.abs(dq))) > 0.05):
        raise RuntimeError("机器人尚未静止或速度反馈无效，不开始恢复试验")
    if (not math.isfinite(args.trial_step_duration)
            or not 2 <= args.trial_step_duration <= 10
            or not 5 <= args.countdown <= 30
            or not 0.005 <= args.joint_tolerance <= 0.1):
        raise ValueError("恢复试验时长、倒计时或关节容差无效")
    if fixed.already_at_fixed_start(q, pose, args):
        print("左臂已经在准备起点；不发送试验 MoveJ，也不修改路径验证标志。")
        return
    waypoints, report = plan_trial(q, pose)
    tolerance = min(args.joint_tolerance, TRIAL_JOINT_TOLERANCE_RAD)
    print(f"独立恢复试验：仅规划左臂，{len(waypoints)} 段 MoveJ，"
          f"每段计划关节变化最多 {report['maximum_planned_joint_step_deg']:.2f}°，"
          f"每段 {args.trial_step_duration:g} 秒。")
    if report["recovery_start_limit_overrun_deg"] > 0:
        print(f"越限恢复模式：起点已有 {report['recovery_start_limit_overrun_deg']:.3f}° "
              "模型限位偏差；第一段目标已回到正常限位内，后续目标不得再次越限。")
    print(f"试验到位关节容差：{tolerance:.4f} rad；不会发送 ServoP、ServoJ、头部或 O6 轨迹。")
    print("模型未包含实际 O6、转接件、桌面和线缆，固件插值/接管响应也未经此试验验证。")
    print("请现场确认整个手臂及 O6 扫掠范围已清空，实体急停在手边。"
          "Ctrl+C 和软件中止不能撤回已接受的 MoveJ。")
    if trace is not None:
        trace.record("restore_trial_plan", **report, current_q=q.tolist(),
                     waypoint_q=waypoints.tolist(), step_duration_s=args.trial_step_duration,
                     joint_tolerance_rad=tolerance, path_approved=False,
                     target_label=fixed.FIXED_LEFT_START_LABEL,
                     target_measured=fixed.FIXED_LEFT_START_MEASURED,
                     height_offset_m=fixed.FIXED_LEFT_HEIGHT_OFFSET_M,
                     position_offset_m=fixed.FIXED_LEFT_POSITION_OFFSET_M.tolist(),
                     profile=str(fixed.FIXED_LEFT_START_PROFILE))
    try:
        answer = input("只测试恢复动作、不启动回放，输入 RESTORE 继续: ").strip()
    except EOFError as exc:
        raise RuntimeError("未收到现场 RESTORE 确认，未发送 MoveJ") from exc
    if answer != "RESTORE":
        raise RuntimeError("cancelled by user")
    fixed.countdown(args.countdown)

    # 左臂变化会使已确认路径失效。保持臂在等待期间少量改变且已停稳时，
    # 以最新姿态重新检查整条路径，避免发送等待前的右臂保持目标。
    fresh_q, fresh_dq = client.joint_snapshot(label="restore_trial_pre_start")
    fresh_q, fresh_dq = np.asarray(fresh_q), np.asarray(fresh_dq)
    if (fresh_q.shape != (16,) or fresh_dq.shape != (16,)
            or not np.isfinite(fresh_q).all() or not np.isfinite(fresh_dq).all()
            or float(np.max(np.abs(fresh_dq))) > 0.05):
        raise RuntimeError("确认后机器人未停稳或反馈无效；未发送 MoveJ，请重新只读预检")
    left_change = float(np.max(np.abs(fresh_q[:7] - q[:7])))
    right_change = float(np.max(np.abs(fresh_q[7:14] - q[7:14])))
    head_change = float(np.max(np.abs(fresh_q[14:16] - q[14:16])))
    if trace is not None:
        trace.record("restore_trial_start_check", left_change_rad=left_change,
                     right_change_rad=right_change, head_change_rad=head_change,
                     maximum_speed_rad_s=float(np.max(np.abs(fresh_dq))),
                     left_change_limit_rad=START_CHANGE_TOLERANCE_RAD)
    if left_change > START_CHANGE_TOLERANCE_RAD:
        raise RuntimeError(
            f"确认后机器人姿态已变化：左臂 {left_change:.5f} rad "
            f"({math.degrees(left_change):.3f}°) 超过 "
            f"{START_CHANGE_TOLERANCE_RAD:.5f} rad；未发送 MoveJ，请重新只读预检"
        )
    if head_change > 0.03:
        raise RuntimeError("确认后头部出现非计划偏移；未发送 MoveJ，请重新只读预检")

    # 编码器小变化也可能影响首段实际步长；重新规划使用完整的新快照。
    if float(np.max(np.abs(fresh_q - q))) > 1e-8:
        fresh_pose = client.current_servo_pose(label="restore_trial_pre_start")
        fixed.check_right_hold(fresh_q, fresh_pose, q[7:14], pose, args)
        fresh_waypoints, fresh_report = plan_trial(fresh_q, fresh_pose)
        if trace is not None:
            trace.record("restore_trial_replan", **fresh_report,
                         reason="refresh_start_and_held_targets_before_first_movej",
                         previous_q=q.tolist(), current_q=fresh_q.tolist(),
                         right_hold_q=fresh_q[7:14].tolist(),
                         waypoint_q=fresh_waypoints.tolist(), path_approved=False)
        if right_change > START_CHANGE_TOLERANCE_RAD:
            print(f"等待期间右臂最大变化 {math.degrees(right_change):.3f}°，现已静止；"
                  "已按最新右臂位置重新检查左臂恢复路径，不回拉旧右臂位置。")
        else:
            print("已按确认后的最新关节反馈重新检查左臂恢复路径。")
        q, pose, waypoints = fresh_q.copy(), fresh_pose, fresh_waypoints

    previous = q.copy()
    for index, waypoint in enumerate(waypoints, 1):
        before_q, before_dq = client.joint_snapshot(label=f"restore_trial_before_{index}")
        before_q, before_dq = np.asarray(before_q), np.asarray(before_dq)
        if (before_q.shape != (16,) or before_dq.shape != (16,)
                or not np.isfinite(before_q).all() or not np.isfinite(before_dq).all()
                or float(np.max(np.abs(before_dq))) > 0.05):
            raise RuntimeError("分段恢复前反馈无效或未停稳，不发送下一段 MoveJ")
        if float(np.max(np.abs(before_q[:7] - previous[:7]))) > tolerance:
            raise RuntimeError("左臂偏离上一段已验收位置，不发送下一段 MoveJ")
        if index == 1:
            # 重新规划后若起点继续变化，不能发送已过期的首段/右臂保持目标。
            change = float(np.max(np.abs(before_q - q)))
            if change > START_CHANGE_TOLERANCE_RAD:
                raise RuntimeError(
                    f"首段发送前机器人再次变化 {change:.5f} rad；"
                    "未发送 MoveJ，请重新只读预检"
                )
        try:
            before_pose = client.current_servo_pose(label=f"restore_trial_before_{index}")
        except TimeoutError as exc:
            raise TimeoutError(
                f"恢复第 {index}/{len(waypoints)} 段发送前，腕位姿查询超时；"
                f"本段及后续 MoveJ 尚未发送。{exc}"
            ) from exc
        fixed.check_right_hold(before_q, before_pose, q[7:14], pose, args)
        if float(np.max(np.abs(before_q[14:16] - q[14:16]))) > 0.03:
            raise RuntimeError("头部出现非计划偏移，不发送下一段 MoveJ")

        # 协议严格只有左7+右7；右7保持本次实测起点，头部不进入目标。
        target = waypoint[:14].copy()
        print(f"恢复试验 {index}/{len(waypoints)}：仅左臂计划变化。", flush=True)
        if trace is not None:
            trace.record("restore_trial_movej", segment=index, target_q=target.tolist(),
                         time_s=args.trial_step_duration)
        reply = client.request("request_movej", {
            "time": args.trial_step_duration, "joint": target.tolist(),
        }, "response_movej", timeout=args.trial_step_duration + 5)
        if reply.get("data", {}).get("result") != "success":
            raise RuntimeError(f"恢复试验第 {index} 段 MoveJ 未被接受：{reply.get('data')}")
        # 单次读取超时可在截止窗口内重试，但不会重发 MoveJ。
        wait_for_target(client, target, slice(0, 7), args.trial_step_duration + 5, tolerance)
        after_q, after_dq = client.joint_snapshot(label=f"restore_trial_after_{index}")
        after_q, after_dq = np.asarray(after_q), np.asarray(after_dq)
        try:
            after_pose = client.current_servo_pose(label=f"restore_trial_after_{index}")
        except TimeoutError as exc:
            raise TimeoutError(
                f"恢复第 {index}/{len(waypoints)} 段 MoveJ 已发送，但腕位姿验收查询超时；"
                f"未发送下一段。{exc}"
            ) from exc
        if (after_q.shape != (16,) or after_dq.shape != (16,)
                or not np.isfinite(after_q).all() or not np.isfinite(after_dq).all()
                or float(np.max(np.abs(after_dq))) > 0.05
                or float(np.max(np.abs(after_q[:7] - target[:7]))) > tolerance):
            raise RuntimeError("分段恢复到位状态未通过验收，不发送下一段 MoveJ")
        fixed.check_right_hold(after_q, after_pose, q[7:14], pose, args)
        if float(np.max(np.abs(after_q[14:16] - q[14:16]))) > 0.03:
            raise RuntimeError("头部出现非计划偏移，不发送下一段 MoveJ")
        if trace is not None:
            trace.record("restore_trial_segment_complete", segment=index,
                         actual_q=after_q.tolist(),
                         left_position=after_pose.left_position.tolist(),
                         left_quaternion=after_pose.left_quaternion.tolist())
        previous = waypoint.copy()
        print(f"恢复试验 {index}/{len(waypoints)}：到位检查通过。", flush=True)

    fixed.check_fixed_pose(after_pose, args.position_tolerance, args.angle_tolerance)
    if trace is not None:
        trace.record("restore_trial_endpoint_reached", path_approved=False,
                     automatic_restore_enabled=False)
    print("左臂分段恢复试验到位；到位结果不等于整条路径验收，"
          "正式自动恢复标志未修改。请先复核现场观察和日志。")
