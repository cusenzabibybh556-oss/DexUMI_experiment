#!/usr/bin/env python3
"""记录固定姿态下的 ServoP 接管、2cm 抬升、返回；不写负载或控制参数。

默认使用 --snapshot 文件只做离线规划。--execute 读取新状态、共用回放锁，
五秒倒计时后发送左右腕目标（只有左腕Z变化）。遥测线程不阻塞发送周期。
异常只停止后续指令，不自动返回；软件停止不是实体急停。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture_tron2_tracker_calibration import acquire_capture_lock
from replay_tron2_tracker import (
    DualArmPose, TraceRecorder, Tron2JsonClient, check_workspace,
    parse_dual_arm_pose, pose_for_trace, quaternion_distance_deg, servop_data,
)
from tron2_left_ik import LeftArmIK
from tron2_large_calibration_plan import _check_path, _disable_removed_graspers
from tools.tron2_motor_observer import MotorObserver

PHASES = ('initial_hold', 'raise', 'upper_hold', 'return', 'return_hold')
PHASE_SECONDS = 3.0
HEIGHT_M = .02
RATE_HZ = 300.0
POSE_TIMEOUT_S = .5
POSITION_GUARD_M = .025
ANGLE_GUARD_DEG = 3.0
HELD_POSITION_GUARD_M = .01

# 保持SDK对象存活至 CLI 的 os._exit，避免已知原生析构崩溃。
_ACTIVE_CLIENTS = []


def make_targets(anchor, rate=RATE_HZ):
    """五段连续轨迹，端点零速度；原姿态及右腕始终不变。"""
    times = np.linspace(0., len(PHASES)*PHASE_SECONDS,
                        int(len(PHASES)*PHASE_SECONDS*rate)+1)
    left = np.tile(np.r_[anchor.left_position, anchor.left_quaternion], (len(times), 1))
    right = np.tile(np.r_[anchor.right_position, anchor.right_quaternion], (len(times), 1))
    phases = []
    for i, t in enumerate(times):
        stage = min(int(t/PHASE_SECONDS), len(PHASES)-1)
        alpha = min(1., max(0., (t-stage*PHASE_SECONDS)/PHASE_SECONDS))
        smooth = alpha*alpha*(3.-2.*alpha)
        dz = HEIGHT_M*smooth if stage == 1 else HEIGHT_M if stage == 2 else HEIGHT_M*(1.-smooth) if stage == 3 else 0.
        left[i, 2] += dz
        phases.append(PHASES[stage])
    return times, left, right, phases


def strict_model():
    model = LeftArmIK()
    lo = np.deg2rad([-180, -15, -210, -150, -100, -43, -43,
                    -180, -170, -85, -150, -80, -43, -43, -45, -90])
    hi = np.deg2rad([149, 170, 85, 30, 80, 43, 43,
                    149, 15, 210, 30, 100, 43, 43, 60, 90])
    model.lower = np.maximum(model.lower, lo)
    model.upper = np.minimum(model.upper, hi)
    return model


def prepare(snapshot):
    frames = snapshot['samples']
    if len(frames) < 3:
        raise ValueError('至少需要三组最新静止反馈')
    model = strict_model()
    qs = np.asarray([s['joint_reply']['data']['q'] for s in frames], dtype=float)
    dqs = np.asarray([s['joint_reply']['data']['dq'] for s in frames], dtype=float)
    if qs.shape != (len(frames), 16) or dqs.shape != qs.shape or not np.isfinite(qs).all() or not np.isfinite(dqs).all():
        raise ValueError('无效关节快照')
    if np.max(abs(dqs)) > .05 or np.max(np.ptp(qs, axis=0)) > .003:
        raise RuntimeError('关节未停稳，拒绝发送')
    poses = [parse_dual_arm_pose(s['pose_reply']) for s in frames]
    anchor = poses[-1]
    for q, pose in zip(qs, poses):
        model.check_anchor(q, pose.left_position, pose.left_quaternion)
        for side in ('left', 'right'):
            if (np.linalg.norm(getattr(pose, side+'_position')-getattr(anchor, side+'_position')) > .001
                    or quaternion_distance_deg(getattr(pose, side+'_quaternion'), getattr(anchor, side+'_quaternion')) > .2):
                raise RuntimeError('双腕在基线采样中未停稳')
    ts, left, right, phases = make_targets(anchor)
    check_workspace('left', left[:, :3])
    removed = _disable_removed_graspers(model)
    joint_plan, metrics = model.plan_tracking(qs[-1], left[:, :3], left[:, 3:], ts)
    for a, b in zip(joint_plan[::10], joint_plan[10::10]):
        err = _check_path(model, a, b, model.lower, model.upper)
        if err:
            raise RuntimeError('模型路径检查失败：'+err)
    margins = np.minimum(joint_plan[:, :7]-model.lower[:7], model.upper[:7]-joint_plan[:, :7])
    if margins.min() < math.radians(5.):
        raise RuntimeError('诊断路径关节余量小于5°')
    return model, anchor, ts, left, right, phases, {
        'height_m': HEIGHT_M, 'rate_hz': RATE_HZ, 'phase_duration_s': PHASE_SECONDS,
        'phases': list(PHASES), 'duration_s': float(ts[-1]), 'frames': len(ts),
        'left_anchor': left[0].tolist(), 'right_hold': right[0].tolist(),
        'minimum_joint_margin_deg': float(np.rad2deg(margins.min())),
        'offline_ik': metrics, 'removed_model_geometry': removed,
        'missing_collision_geometry': ['O6', 'mount', 'table', 'wires'],
        'payload_writes': 0, 'o6_targets': 0, 'head_targets': 0,
    }


def guard(anchor, target, pose):
    delta = pose.left_position-target[:3]
    angle = quaternion_distance_deg(pose.left_quaternion, target[3:])
    right_drift = np.linalg.norm(pose.right_position-anchor.right_position)
    right_angle = quaternion_distance_deg(pose.right_quaternion, anchor.right_quaternion)
    if np.linalg.norm(delta) > POSITION_GUARD_M or angle > ANGLE_GUARD_DEG:
        raise RuntimeError(f'左腕误差超限：{np.linalg.norm(delta)*1000:.1f}mm/{angle:.2f}°')
    if right_drift > HELD_POSITION_GUARD_M or right_angle > ANGLE_GUARD_DEG:
        raise RuntimeError(f'右腕保持偏移超限：{right_drift*1000:.1f}mm/{right_angle:.2f}°')


def require_same_anchor(anchor, latest):
    """规划及等待期间起点须保持；不把运动中的新读数重新当作起点。"""
    for side in ('left', 'right'):
        if (np.linalg.norm(getattr(latest, side+'_position')-getattr(anchor, side+'_position')) > .001
                or quaternion_distance_deg(getattr(latest, side+'_quaternion'), getattr(anchor, side+'_quaternion')) > .2):
            raise RuntimeError('首帧前基线发生变化，未发送运动')


def validate_payload_reply(reply):
    """核验只读回复，不推测未知负载，也不执行写入。"""
    data = reply.get('data', {})
    values = np.asarray(data.get('data', []), dtype=float)
    if (data.get('result') != 'success' or values.shape != (8,)
            or not np.isfinite(values).all() or values[0] < 0 or values[4] < 0):
        raise RuntimeError('负载只读回复无效，未开始诊断')
    return values


def require_expected_payload(snapshot, expected):
    """单因素复测时，在每次基线读取后核对指定的完整左右负载配置。"""
    if expected is None:
        return
    expected = np.asarray(expected, dtype=float)
    actual = validate_payload_reply(snapshot['payload_reply'])
    if (expected.shape != (8,) or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, rtol=1e-6, atol=1e-8)):
        raise RuntimeError('负载配置与本轮预期不符，未启动诊断运动')


class Telemetry:
    def __init__(self, client, trace):
        self.client, self.trace = client, trace
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.latest = None
        self.latest_joint = None
        self.error = None
        self.query_timing = {}

    def query(self, request, response, event):
        # 只重试无副作用查询；单个回复丢失不应阻塞整个遥测线程。
        # 三次短查询合计最多约0.3秒，主发送环的0.5秒新鲜度门限不变。
        for attempt in range(1, 4):
            sent = time.monotonic_ns()
            try:
                reply = self.client.request(request, {}, response, timeout=.1)
            except TimeoutError as exc:
                self.trace.record('telemetry_query_retry', request=request,
                                  attempt=attempt, message=str(exc))
                if attempt == 3:
                    raise
                continue
            received = time.monotonic_ns()
            self.query_timing[event] = (sent, received, attempt)
            self.trace.record(event, request_guid=reply.get('guid'), sent_monotonic_ns=sent,
                              received_monotonic_ns=received, reply=reply, attempts=attempt)
            return received, reply

    def run(self):
        index = 0
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                at, reply = self.query('request_get_servo_pose', 'response_get_servo_pose', 'diagnostic_pose')
                self.latest = (at, parse_dual_arm_pose(reply))
                if index % 2 == 0:
                    at, reply = self.query('request_get_joint_state', 'response_get_joint_state', 'diagnostic_joint')
                    self.latest_joint = (at, reply['data'])
                    self.query('request_get_move_pose', 'response_get_move_pose', 'diagnostic_move_pose')
                index += 1
                self.stop.wait(max(0., .05-(time.monotonic()-started)))
        except Exception as exc:
            self.error = str(exc)
            self.trace.record('telemetry_error', message=self.error)

    def require_fresh(self):
        if self.error:
            raise RuntimeError('遥测失败：'+self.error)
        now = time.monotonic_ns()
        if self.latest is None or (now-self.latest[0])/1e9 > POSE_TIMEOUT_S:
            raise RuntimeError('腕位姿反馈失联，停止后续指令')
        if self.latest_joint is None or (now-self.latest_joint[0])/1e9 > POSE_TIMEOUT_S:
            raise RuntimeError('关节反馈失联，停止后续指令')
        return self.latest[1], self.latest_joint[1]


def baseline(client):
    mode = client.request('request_drag_teach_manage', {'cmd':'status', 'arg':''}, 'response_drag_teach_manage')
    if 'state=ST_DEVED;' not in mode['data'].get('message', ''):
        raise RuntimeError('需要High-Level Dev，不自动切模式')
    payload = client.request('request_get_payload_param', {}, 'response_get_payload_param')
    validate_payload_reply(payload)
    samples = []
    for _ in range(5):
        joint = client.request('request_get_joint_state', {}, 'response_get_joint_state')
        pose = client.request('request_get_servo_pose', {}, 'response_get_servo_pose')
        samples.append({'joint_reply': joint, 'pose_reply': pose})
        time.sleep(.1)
    move = client.request('request_get_move_pose', {}, 'response_get_move_pose')
    return {'robot_id': client.accid, 'mode_reply': mode, 'payload_reply': payload,
            'samples': samples, 'move_pose_reply': move}


def readonly_configuration(client):
    """保留当前配置回复；只读接口不支持时记录错误，不推测或写入默认值。"""
    report = {}
    for title in ('compliant_enable', 'compliant_param'):
        try:
            report[title] = client.request('request_get_'+title, {},
                                           'response_get_'+title, timeout=1.)
        except Exception as exc:
            report[title] = {'read_error': str(exc)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--transport', choices=('sdk', 'raw-ws'), default='sdk',
                        help='诊断传输入口；两者均发送相同ServoP数据，不切换运动接口')
    parser.add_argument('--expected-payload', nargs=8, type=float,
                        help='可选：首次及倒计时后基线必须匹配的左右负载原始八参数')
    parser.add_argument('--snapshot', type=Path, help='仅离线规划使用的原始基线JSON')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.execute and args.snapshot is None:
        parser.error('默认仅规划，需提供--snapshot；真机执行需显式--execute')
    args.output.mkdir(parents=True, exist_ok=False)
    trace = TraceRecorder(args.output/'command_state.jsonl')
    client = telemetry = lock = observer = None
    summary = {'captured_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'motion_requested': args.execute, 'transport': args.transport,
               'result': 'not_started'}
    try:
        if args.execute:
            lock = acquire_capture_lock()
            if args.transport == 'raw-ws':
                from tools.tron2_raw_ws_client import RawWSClient
                client = RawWSClient('10.192.1.2', trace)
            else:
                client = Tron2JsonClient('10.192.1.2', trace)
                observer = MotorObserver(trace)
                observer.start(client.robot)
            _ACTIVE_CLIENTS.append(client)
            client.wait_for_identity()
            if client.accid != 'DACH_TRON2A_253':
                raise RuntimeError('机器人身份不符')
            snapshot = baseline(client)
            require_expected_payload(snapshot, args.expected_payload)
            snapshot['configuration_readonly'] = readonly_configuration(client)
            trace.record('diagnostic_configuration', configuration=snapshot['configuration_readonly'])
        else:
            snapshot = json.loads(args.snapshot.read_text())
        (args.output/'baseline.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n')
        model, anchor, ts, left, right, phases, report = prepare(snapshot)
        (args.output/'plan.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        np.savez_compressed(args.output/'planned_targets.npz', times_s=ts, left=left, right=right, phases=phases)
        summary['plan'] = report
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if not args.execute:
            summary['result'] = 'planned_only'
            return 0
        for remain in range(5, 0, -1):
            print(f"{remain}秒后开始左臂高度诊断，最大相对抬升{report['height_m']*100:g}cm；实体急停在手边。", flush=True)
            time.sleep(1)
        fresh = baseline(client)
        require_expected_payload(fresh, args.expected_payload)
        fresh_anchor = parse_dual_arm_pose(fresh['samples'][-1]['pose_reply'])
        require_same_anchor(anchor, fresh_anchor)
        prepare(fresh)
        trace.record('diagnostic_baseline', snapshot=fresh, **pose_for_trace(anchor))
        telemetry = Telemetry(client, trace)
        telemetry.thread.start()
        deadline = time.monotonic()+1.
        while telemetry.latest is None or telemetry.latest_joint is None:
            if telemetry.error or time.monotonic() >= deadline:
                raise RuntimeError('首帧前未收到完整遥测，未发送运动')
            time.sleep(.005)
        require_same_anchor(anchor, telemetry.latest[1])
        start = time.monotonic()
        trace.record('diagnostic_start', start_monotonic_ns=int(start*1e9), **report)
        current_phase = None
        last_joint_stamp = None
        for index, due in enumerate(ts):
            remaining = start+due-time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            if time.monotonic()-(start+due) > .1:
                raise RuntimeError('发送周期滞后超过100ms，不追赶发送后续轨迹')
            actual, joint = telemetry.require_fresh()
            guard(anchor, left[index], actual)
            if last_joint_stamp != telemetry.latest_joint[0]:
                model.check_joints(np.asarray(joint['q'],dtype=float))
                if np.max(abs(np.asarray(joint['q'])[7:14]-np.asarray(fresh['samples'][-1]['joint_reply']['data']['q'])[7:14])) > .03:
                    raise RuntimeError('右臂关节发生非计划偏移，停止后续发送')
                if np.max(abs(np.asarray(joint['q'])[14:16]-np.asarray(snapshot['samples'][-1]['joint_reply']['data']['q'])[14:16])) > .03:
                    raise RuntimeError('头部发生非计划偏移，停止后续发送')
                last_joint_stamp = telemetry.latest_joint[0]
            if phases[index] != current_phase:
                current_phase = phases[index]
                trace.record('diagnostic_phase', phase=current_phase, scheduled_time_s=float(due))
                print('诊断阶段：'+current_phase, flush=True)
            failure = client.failure()
            if failure:
                raise RuntimeError('控制器拒绝ServoP：'+failure)
            trace.record('diagnostic_command', index=index, phase=current_phase,
                         due_monotonic_ns=int((start+due)*1e9), left=left[index].tolist(), right=right[index].tolist())
            client.send('request_servop', servop_data(left[index], right[index]))
        trace.record('diagnostic_sending_stopped')
        # 正常返回原目标后继续只读观察；异常分支不再发送任何恢复指令。
        time.sleep(1.)
        summary['result'] = 'completed'
        print('诊断指令发送完成；已返回原目标，随后只读观察1秒。', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(result='aborted', error=str(exc) or 'KeyboardInterrupt')
        trace.record('diagnostic_abort', message=summary['error'])
        print('已停止后续发送：'+summary['error']+'；这不是硬件急停，不自动返回。',file=sys.stderr,flush=True)
        return 2
    finally:
        if telemetry is not None:
            telemetry.stop.set()
            telemetry.thread.join(timeout=1.)
        if client is not None:
            for request,response,event in (
                ('request_get_servo_pose','response_get_servo_pose','diagnostic_final_pose'),
                ('request_get_joint_state','response_get_joint_state','diagnostic_final_joint'),
                ('request_get_payload_param','response_get_payload_param','diagnostic_final_payload')):
                try:
                    trace.record(event, reply=client.request(request,{},response,timeout=.5))
                except Exception as exc:
                    trace.record(event+'_error', message=str(exc))
        if observer is not None:
            observer.stop()
            summary['motor_observer'] = observer.summary()
        if client is not None and args.transport == 'raw-ws':
            try:
                client.close()
            except Exception as exc:
                summary['transport_close_error'] = str(exc)
        summary['note'] = 'returned target does not guarantee physical return; inspect recorded state'
        (args.output/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        if lock is not None:
            lock.close()
        trace.close()


if __name__ == '__main__':
    import os
    code = main()
    if '--execute' in sys.argv:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(code)
    raise SystemExit(code)
