#!/usr/bin/env python3
"""独立左腕标定：默认只读规划；--execute 才执行小段 MoveJ、采样并离线求解。

不依赖网页。Tracker 必须与左腕刚性固定；O6/原厂夹爪已拆除。
只变化左腕末端三关节，其余关节保持本次起点；不发送 ServoP/ServoJ。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

OFFSETS_DEG = np.array([
    [0, 0, 0], [12, 0, 0], [-12, 0, 0], [0, 12, 0], [0, -8, 0],
    [0, 0, 12], [0, 0, -12], [12, 12, 0], [-12, 12, 0],
    [12, 0, 12], [-12, 0, -12], [0, 12, 12], [0, 12, -12],
    [8, 8, -8], [-8, -6, 8], [0, 0, 0],
], dtype=float)


def build_plan(model, start):
    """检查关节限位、模型内自碰撞，分成不超过 5° 的关节段。"""
    start = np.asarray(start, float)
    model.check_joints(start)
    targets = np.tile(start, (len(OFFSETS_DEG), 1))
    m, d, mj = model.model, model.data, model.mj
    # 用户确认两只 O6 已拆，模型原厂夹爪也不代表当前 Tracker 支架。
    for g in range(m.ngeom):
        if m.body(int(m.geom_bodyid[g])).name.startswith("grasper_"):
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
    stages, minimum, previous, positions = [], math.inf, start.copy(), []
    # MoveJ 文档腕 pitch/roll 范围比通用模型更窄，取两者交集。
    lower, upper = model.lower.copy(), model.upper.copy()
    lower[[5, 6, 12, 13]] = np.maximum(lower[[5, 6, 12, 13]], -.7505)
    upper[[5, 6, 12, 13]] = np.minimum(upper[[5, 6, 12, 13]], .7505)
    if np.any(start[:14] < lower[:14]) or np.any(start[:14] > upper[:14]):
        raise ValueError("当前姿态超出 MoveJ 文档范围，未发送动作")
    offsets = OFFSETS_DEG.copy()
    # 靠近一侧限位时减少该方向的采样角度，不裁剪实际下发关节角。
    for axis in range(3):
        j = 4+axis
        room_negative = np.rad2deg(start[j]-lower[j])-5
        room_positive = np.rad2deg(upper[j]-start[j])-5
        if min(room_negative, room_positive) < 0:
            raise ValueError("当前腕关节距限位不足5°，请先调整准备姿态")
        offsets[:,axis] *= np.where(offsets[:,axis]<0,
                                   min(1., room_negative/12), min(1., room_positive/12))
    targets[:,4:7] += np.deg2rad(offsets)
    for target in targets:
        model.check_joints(target)
        if np.any(target[:14] < lower[:14]) or np.any(target[:14] > upper[:14]):
            raise ValueError("当前姿态或采样偏转超出 MoveJ 文档范围，未发送动作")
        margin = float(np.rad2deg(np.minimum(target[4:7]-lower[4:7], upper[4:7]-target[4:7])).min())
        if margin < 5-1e-8:
            raise ValueError("采样目标距腕关节限位不足 5°；请调整起始姿态后重新规划")
        minimum = min(minimum, margin)
        positions.append(model.pose(target)[0].tolist())
        for alpha in np.linspace(0, 1, 61):
            q = previous * (1-alpha) + target * alpha
            d.qpos[model.qadr] = q
            mj.mj_forward(m, d)
            for contact in d.contact:
                names = [m.body(int(m.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)]
                if contact.dist < 0 and any(n.endswith("_L_Link") for n in names):
                    raise ValueError(f"模型内自碰撞：{names}；未发送动作")
        n = int(math.ceil(float(np.max(abs(target - previous))) / math.radians(5)))
        stages.append([previous + (target - previous) * (j+1) / n for j in range(n)])
        previous = target.copy()
    return targets, stages, {"start_q_rad": start.tolist(), "wrist_offsets_deg": offsets.tolist(),
                            "target_q_rad": targets.tolist(), "tcp_positions_m": positions,
                            "minimum_wrist_joint_margin_deg": minimum,
                            "maximum_step_deg": 5, "model_self_collision_passed": True,
                            "missing_geometry": ["Tracker支架", "线缆", "桌面及环境"]}


def current_joints(client):
    data = client.request("request_get_joint_state", {}, "response_get_joint_state")["data"]
    q, dq = np.asarray(data.get("q", []), float), np.asarray(data.get("dq", []), float)
    if q.shape != (16,) or dq.shape != (16,) or not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise RuntimeError("缺少有效的16维实测关节/速度反馈")
    return q, dq


def require_tracker(session, diagnostic_statuses=()):
    s = session.status()
    t, r = s.get("tracker"), s.get("robot")
    if (not t or not r or t["age_ms"] > 250 or r["age_ms"] > 250
            or t["tracking_status"] not in ({2} | set(diagnostic_statuses)) or s["error"]):
        raise RuntimeError(f"Tracker/关节反馈未就绪：Tracker状态={t.get('tracking_status') if t else None}，"
                           f"帧龄={round(t['age_ms'],1) if t else None}ms，"
                           f"关节帧龄={round(r['age_ms'],1) if r else None}ms，错误={s['error']}")
    return t["mac"]


def capture_group(session, label, diagnostic_statuses=()):
    session.sample(0, 2., label)
    session.capture_thread.join(timeout=5)
    if session.capture_thread.is_alive() or session.capture_error:
        raise RuntimeError("静止姿态采样未完成")
    group = session.groups[-1]
    diagnostic_only = (set(group['tracking_statuses']) <= ({2} | set(diagnostic_statuses))
                       and group['reasons'] and all(r.startswith('含非 OK 跟踪状态') for r in group['reasons']))
    if not group["eligible_for_calibration"] and not diagnostic_only:
        raise RuntimeError("样本不稳定：" + "; ".join(group["reasons"]))
    return group


def execute_stages(client, session, start, targets, stages, log, duration=2., diagnostic_statuses=(), allowed_joints=(4,5,6), sample_indices=None):
    """逐段确认实测关节；MoveJ 超时不会重发，错误后不自动返回。"""
    previous = start.copy()
    identity = require_tracker(session, diagnostic_statuses)
    for i, (target, segments) in enumerate(zip(targets, stages)):
        for goal in segments:
            if require_tracker(session, diagnostic_statuses) != identity:
                raise RuntimeError("Tracker 身份变化，停止后续动作")
            q, dq = current_joints(client)
            if np.max(abs(q - previous)) > .025 or np.max(abs(dq)) > .05:
                raise RuntimeError("关节偏离上一段或仍在运动，停止后续 MoveJ")
            log("movej_request", group=i+1, joint=goal[:14].tolist(), duration=duration)
            began = time.monotonic()
            # 官方 MoveJ 为左7+右7；不下发任何头部目标。
            reply = client.request("request_movej", {"time": duration, "joint": goal[:14].tolist()}, "response_movej")
            log("movej_reply", response=reply)
            stable = 0
            while True:
                require_tracker(session, diagnostic_statuses)
                q, dq = current_joints(client)
                error = float(np.max(abs(q-goal)))
                log("joint_feedback", group=i+1, q=q.tolist(), max_error_rad=error)
                if (np.any(q[:7] < np.minimum(previous[:7], goal[:7])-.05)
                        or np.any(q[:7] > np.maximum(previous[:7], goal[:7])+.05)):
                    raise RuntimeError("左臂超出当前分段的计划关节范围，停止后续动作")
                if np.max(abs(q[7:] - start[7:])) > .035:
                    raise RuntimeError("右臂/头部实测偏移超过2°，停止后续动作")
                fixed_left=[j for j in range(7) if j not in allowed_joints]
                if fixed_left and np.max(abs(q[fixed_left] - start[fixed_left])) > .035:
                    raise RuntimeError("左臂未规划关节偏移超过2°，停止后续动作")
                if time.monotonic()-began >= duration and error <= .015 and np.max(abs(dq)) <= .03:
                    stable += 1
                    if stable >= 3:
                        break
                else:
                    stable = 0
                if time.monotonic()-began > duration+6:
                    raise TimeoutError(f"MoveJ 未停稳，关节误差 {error:.4f} rad；没有重发")
                time.sleep(.1)
            previous = goal.copy()
        if sample_indices is not None and i not in sample_indices:
            log("transit_completed", target_index=i)
            print(f"返回路径已完成目标 {i+1}/{len(targets)}", flush=True)
            continue
        time.sleep(.3)
        require_tracker(session, diagnostic_statuses)
        label=f"左臂关节{list(allowed_joints)}偏转 {np.rad2deg(target[list(allowed_joints)]-start[list(allowed_joints)]).round(3).tolist()} deg"
        g = capture_group(session, label, diagnostic_statuses)
        log("sample_completed", target_index=i, sample_id=g["id"] if "id" in g else len(session.groups),
            eligible=g["eligible_for_calibration"], tracking_statuses=g["tracking_statuses"])
        print(f"已保存 {i+1}/{len(targets)} 组，{g['pair_count']} 对；"
              f"Tracker波动 {g['tracker_statistics']['position_radius_mm']:.2f} mm / "
              f"{g['tracker_statistics']['rotation_radius_deg']:.2f}°", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-ip", default="10.192.1.2")
    p.add_argument("--tracker-index", type=int, default=0, choices=range(5))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--passive-hid", required=True, help="现有采集正在使用的 /dev/hidrawN；仅只读，不初始化")
    p.add_argument("--tracker-mac", required=True, help="只接收该 Tracker 的完整位姿报告")
    p.add_argument("--tracker-quaternion", choices=['legacy-wzyx','raw-xyzw'],default='legacy-wzyx',
                   help="求解时显式选择四元数解释；原始采样不改写")
    p.add_argument("--diagnostic-status", type=int, choices=[18,20], action='append', default=[],
                   help="保留未知原始状态做诊断采样，结果不能标为正式通过")
    p.add_argument("--countdown", type=int, default=5, choices=range(5, 31))
    args = p.parse_args()
    from capture_tron2_tracker_calibration import HardwareSource, CaptureSession, acquire_capture_lock
    from tron2_left_ik import LeftArmIK
    from tools.tron2_head_pose import HeadClient
    from dexumi_wrist_solver import solve_dexumi
    root = Path(__file__).resolve().parent
    handle, client, session = None, None, None
    run = root / "tracker_calibrations" / ("auto_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    audit = (run / "motion.jsonl").open("x")
    def log(event, **fields):
        audit.write(json.dumps({"event": event, "time": time.time(), **fields}, ensure_ascii=False)+"\n")
        audit.flush()
    try:
        handle = acquire_capture_lock()
        client = HeadClient(args.robot_ip)
        if client.accid != "DACH_TRON2A_253":
            raise RuntimeError("机器人身份不符，未发送动作")
        start, dq = current_joints(client)
        if np.max(abs(dq)) > .05:
            raise RuntimeError("机器人正在运动，请停稳后重试")
        targets, stages, report = build_plan(LeftArmIK(), start)
        (run / "plan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"计划：16组左腕姿态，{sum(map(len,stages))}段 MoveJ；每段2秒，不超过5°。", flush=True)
        print(f"最小腕关节余量：{report['minimum_wrist_joint_margin_deg']:.2f}°；计划文件：{run/'plan.json'}", flush=True)
        if not args.execute:
            print("只读规划结束，没有连接 Tracker 或发送运动。")
            return 0
        status = client.request("request_drag_teach_manage", {"cmd":"status", "arg":""}, "response_drag_teach_manage")
        log("initial_mode", response=status)
        if "state=ST_DEVED;" not in status["data"].get("message", ""):
            raise RuntimeError("自动采样需要 High-Level Dev；当前模式不符，没有发送动作")
        source = HardwareSource(args.robot_ip, args.tracker_index, passive_hid=args.passive_hid,
                                tracker_mac=args.tracker_mac)
        session = CaptureSession(source, run, lambda _: None,
                                 {"mock":False, "tracker_index":args.tracker_index,
                                  "robot_ip":args.robot_ip, "map_label":"当前 Tracker 地图",
                                  "diagnostic_statuses":args.diagnostic_status,
                                  "mount_label":"左腕 Tracker+支架，O6已拆；自动腕部采样"})
        session.start()
        deadline = time.monotonic()+20
        while True:
            try:
                require_tracker(session, args.diagnostic_status)
                break
            except RuntimeError:
                if time.monotonic()>deadline:
                    raise
                time.sleep(.1)
        print(f"已连接 Tracker 和实测关节；样本文件：{session.path}", flush=True)
        print("将开始左腕小范围动作。实体急停在手边；Ctrl+C不能撤回已接受的MoveJ。", flush=True)
        for i in range(args.countdown,0,-1):
            print(f"{i}秒后开始", flush=True)
            time.sleep(1)
        fresh, dq = current_joints(client)
        if np.max(abs(fresh-start))>.005 or np.max(abs(dq))>.05:
            raise RuntimeError("准备期间姿态发生变化，未发送动作，请重新规划")
        status = client.request("request_drag_teach_manage", {"cmd":"status", "arg":""}, "response_drag_teach_manage")
        if "state=ST_DEVED;" not in status["data"].get("message", ""):
            raise RuntimeError("倒计时后模式已改变，未发送动作")
        execute_stages(client, session, start, targets, stages, log, diagnostic_statuses=args.diagnostic_status)
        path = session.path
        session.close();session=None
        result = solve_dexumi(path, validation_ids={4,8,12,16}, diagnostic_statuses=args.diagnostic_status,
                              quaternion_convention=args.tracker_quaternion)
        print(f"结果目录：{result['output_dir']}", flush=True)
        return 0 if result["status"]=="passed" else 2
    except (Exception, KeyboardInterrupt) as exc:
        log("aborted", error=str(exc) or "KeyboardInterrupt")
        print(f"已停止后续动作：{exc}；没有自动返回或重发 MoveJ。", flush=True)
        return 1
    finally:
        if session is not None:
            session.close()
        if client is not None:
            client.socket.close()
        if handle is not None:
            handle.close()
        audit.close()


if __name__ == "__main__":
    raise SystemExit(main())
