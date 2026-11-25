#!/usr/bin/env python3
"""TRON2 单臂末端负载参数：默认只读，不切换模式或发送运动指令。

原始每臂四参数为 [m, mx, my, mz]：质量单位 kg，后三项是质量一阶矩
kg*m，不能把重心坐标（m）直接填入。若已测量重心 c，应先计算 m*c。
一阶矩须在厂商负载接口规定的参考系中提供；公开接口页未明示其原点，
不能把 SDK 虚拟 TCP 当作负载原点，应核对厂商安装坐标定义。
仅 --execute --arm SIDE --mass-kg M --first-moment-kg-m MX MY MZ 才写入。
写入只尝试一次，随后回读双臂核对；失败时不重发、不自动回滚。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from numbers import Real
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from websocket import WebSocketException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture_tron2_tracker_calibration import acquire_capture_lock
from tools.tron2_head_pose import HeadClient


ROBOT_ID = "DACH_TRON2A_253"
PAYLOAD_FIELDS = ["mass_kg", "first_moment_x_kg_m", "first_moment_y_kg_m", "first_moment_z_kg_m"]
MAX_STATIONARY_SPEED_RAD_S = 0.05


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="10.192.1.2")
    parser.add_argument("--arm", choices=("left", "right"), help="写入时必须显式指定单臂")
    parser.add_argument("--mass-kg", type=float, help="实测附件总质量，kg；必须为正数")
    parser.add_argument("--first-moment-kg-m", type=float, nargs=3, metavar=("MX", "MY", "MZ"),
                        help="实测质量一阶矩 m*c，单位 kg*m；不是重心坐标 m")
    parser.add_argument("--execute", action="store_true", help="完整四参数已显式给出时，单次写入指定臂")
    parser.add_argument("--output", type=Path, help="独立 JSON 记录；不覆盖已存在的证据文件")
    return parser.parse_args(argv)


def _numeric_vector(value, count, label):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{label} 必须是 {count} 个有限数值")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in value):
        raise ValueError(f"{label} 必须是 {count} 个有限数值")
    try:
        result = [float(v) for v in value]
    except OverflowError as exc:
        raise ValueError(f"{label} 必须是 {count} 个有限数值") from exc
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{label} 必须是 {count} 个有限数值")
    return result


def _json_record(value):
    """畸形设备回复中的 NaN/Inf 以明确字符串留证，不生成无效 JSON。"""
    if isinstance(value, dict):
        return {k: _json_record(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_record(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return f"nonfinite:{value!r}"
    return value


def requested_parameters(args):
    """返回显式给出的原始四参数；不从历史负载或质量猜测一阶矩。"""
    supplied = args.mass_kg is not None or args.first_moment_kg_m is not None
    if args.execute or supplied:
        if args.arm is None or args.mass_kg is None or args.first_moment_kg_m is None:
            raise ValueError("写入或预览目标必须完整指定 --arm、--mass-kg 和 --first-moment-kg-m MX MY MZ")
        target = _numeric_vector([args.mass_kg, *args.first_moment_kg_m], 4, "负载目标")
        if target[0] <= 0:
            raise ValueError("--mass-kg 必须为有限正质量，不能猜测负载")
        return target
    return None


def parse_payload_reply(reply):
    if not isinstance(reply, dict) or not isinstance(reply.get("data"), dict):
        raise ValueError("负载回复缺少有效 data 对象")
    data = reply["data"]
    if data.get("result") != "success":
        raise RuntimeError(f"负载查询未成功：{data.get('result')!r}")
    raw = _numeric_vector(data.get("data"), 8, "双臂原始负载参数")
    if raw[0] <= 0 or raw[4] <= 0:
        raise ValueError("双臂负载质量必须为正数；未发送参数写入")
    return raw


def read_stationary_joints(client):
    reply = client.request("request_get_joint_state", {}, "response_get_joint_state")
    if not isinstance(reply, dict):
        raise ValueError("关节回复必须是对象，未确认静止状态")
    data = reply.get("data", {})
    if not isinstance(data, dict) or data.get("result") != "success":
        raise RuntimeError("关节反馈查询未成功，未确认静止状态")
    q = _numeric_vector(data.get("q"), 16, "关节位置反馈")
    dq = _numeric_vector(data.get("dq"), 16, "关节速度反馈")
    maximum = max(map(abs, dq))
    if maximum > MAX_STATIONARY_SPEED_RAD_S:
        raise RuntimeError(f"机器人尚未静止：最大关节速度 {maximum:.5f} rad/s；不确认本次写入成功")
    return {"q_rad": q, "dq_rad_s": dq}


def read_payload(client, report, label):
    reply = client.request("request_get_payload_param", {}, "response_get_payload_param")
    report[f"{label}_reply"] = reply
    raw = parse_payload_reply(reply)
    report[f"{label}_payload_raw_kg_kgm"] = raw
    return raw


def _same_parameters(actual, expected):
    # SET/GET 经过固件 float32，允许其表示误差，不把明显改变视为相同。
    return all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8) for a, b in zip(actual, expected))


def verify_payload_readback(before, after, arm, wanted):
    offset = 0 if arm == "left" else 4
    other_offset = 4 - offset
    if not _same_parameters(after[other_offset:other_offset + 4], before[other_offset:other_offset + 4]):
        raise RuntimeError("负载回读显示对侧参数发生变化；不认定写入成功，不自动回滚")
    if not _same_parameters(after[offset:offset + 4], wanted):
        raise RuntimeError("指定臂负载回读与目标不符；不认定写入成功，不重发写入")


def print_payload(raw):
    print("参数顺序：[质量 kg，一阶矩 MX/MY/MZ kg*m]；后三项不是重心坐标 m。")
    print(f"left 原始负载: {raw[:4]}")
    print(f"right 原始负载: {raw[4:]}")


def main(argv=None):
    args = parse_args(argv)
    client = lock = output = None
    report = {
        "schema": "tron2-payload-inspection-v1",
        "captured_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "robot_ip": args.robot_ip, "execute_requested": args.execute,
        "raw_parameter_fields_per_arm": PAYLOAD_FIELDS,
        "raw_payload_order": "left[m,mx,my,mz],right[m,mx,my,mz]",
        "first_moment_definition": "mass_kg * center_of_mass_m; units kg*m, not center_of_mass_m",
        "set_attempts": 0, "motion_or_mode_requests_sent": False,
        "status": "not_started",
    }
    try:
        wanted = requested_parameters(args)
        report["selected_arm"] = args.arm
        report["requested_raw_kg_kgm"] = wanted
        if args.output is not None:
            destination = args.output.expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            output = destination.open("x", encoding="utf-8")
        if args.execute:
            lock = acquire_capture_lock()
        client = HeadClient(args.robot_ip)
        report["robot_id"] = client.accid
        if client.accid != ROBOT_ID:
            raise RuntimeError(f"连接的是 {client.accid}，本工具要求 {ROBOT_ID}")
        before = read_payload(client, report, "before")
        print_payload(before)
        if wanted is not None:
            print(f"{args.arm} 目标原始四参数: {wanted}（kg，kg*m，kg*m，kg*m）")
        if not args.execute:
            report["status"] = "preview_only" if wanted is not None else "read_only"
            print("目标预览完成。" if wanted is not None else "只读查询完成。")
            return 0

        report["joint_state_before"] = read_stationary_joints(client)
        payload = {"arm": 0 if args.arm == "left" else 1, "data": wanted.copy()}
        report["set_request_data"] = payload
        report["set_attempts"] = 1
        set_error = None
        try:
            reply = client.request("request_set_payload_param", payload, "response_set_payload_param")
            report["set_reply"] = reply
            if not isinstance(reply, dict):
                raise ValueError("负载写入回复不是有效对象")
            data = reply.get("data", {})
            if (not isinstance(data, dict) or data.get("result") != "success"
                    or ("arm" in data and data["arm"] != payload["arm"])):
                raise RuntimeError("负载写入回复未确认指定臂成功")
        except (OSError, ValueError, RuntimeError, TimeoutError, WebSocketException) as exc:
            set_error = exc
            report["set_error"] = str(exc)

        # 即使 SET 回复丢失，也只尝试 GET 记录现值；绝不重发 SET 或自动回滚。
        try:
            after = read_payload(client, report, "after")
        except (OSError, ValueError, RuntimeError, TimeoutError, WebSocketException) as exc:
            report["readback_error"] = str(exc)
            if set_error is not None:
                raise RuntimeError(f"负载写入未确认：{set_error}；回读也失败：{exc}；未重发写入") from set_error
            raise
        if set_error is not None:
            raise RuntimeError(f"负载写入未确认：{set_error}；已保存只读回读，未重发写入") from set_error
        verify_payload_readback(before, after, args.arm, wanted)
        report["joint_state_after"] = read_stationary_joints(client)
        report["status"] = "set_verified"
        print_payload(after)
        print(f"{args.arm} 负载单次写入并回读通过，对侧参数保持。")
        return 0
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt; set acceptance may be unknown")
        print("已中断；没有重发负载写入或自动回滚。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, TimeoutError, WebSocketException) as exc:
        report.update(status="error", error=str(exc))
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.socket.close()
        if lock is not None:
            lock.close()
        if output is not None:
            try:
                json.dump(_json_record(report), output, ensure_ascii=False, indent=2, allow_nan=False)
                output.write("\n")
            finally:
                output.close()


if __name__ == "__main__":
    raise SystemExit(main())
