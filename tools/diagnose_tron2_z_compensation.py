#!/usr/bin/env python3
"""独立验证左腕高度位置外环；不是力控制或已识别的重力前馈。

默认只做离线包络预检。真机使用单一ServoP入口，不写负载、刚度或柔顺参数。
先保持原物理参考，最多修正5mm验证方向，再缓慢扩大到显式给出的上限。
上下运动期间冻结修正，平台期间用新反馈学习。结束保留最后修正，不突然清零。
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from replay_tron2_tracker import (TraceRecorder, Tron2JsonClient, parse_dual_arm_pose,
                                  pose_for_trace, servop_data, check_workspace)
from capture_tron2_tracker_calibration import acquire_capture_lock
from tools.diagnose_tron2_servop import (Telemetry, baseline, prepare, guard,
                                        require_same_anchor, require_expected_payload,
                                        readonly_configuration, _ACTIVE_CLIENTS)
from tools.tron2_motor_observer import MotorObserver
from tron2_large_calibration_plan import _check_path
from tron2_z_compensation import Config, ZCompensationController


@dataclass(frozen=True)
class PosePacket:
    guid: str
    sent_ns: int
    received_ns: int
    pose: object
    source_timestamp_ms: float
    source_progress: bool


class CompensationTelemetry(Telemetry):
    """沿用既有遥测守护，另存用于外环的完整、同一回复数据包。"""
    def __init__(self, client, trace):
        super().__init__(client, trace)
        self.packet = None
        self.source_clocks = {}

    def run(self):
        """只查询控制必需的腕位姿和关节，避免辅助MovePose查询阻塞看门狗。"""
        index = 0
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                at, reply = self.query('request_get_servo_pose',
                                       'response_get_servo_pose', 'diagnostic_pose')
                self.latest = (at, parse_dual_arm_pose(reply))
                if index % 2 == 0:
                    at, reply = self.query('request_get_joint_state',
                                           'response_get_joint_state', 'diagnostic_joint')
                    self.latest_joint = (at, reply['data'])
                index += 1
                self.stop.wait(max(0., .05-(time.monotonic()-started)))
        except Exception as exc:
            self.error = str(exc)
            self.trace.record('telemetry_error', message=self.error)

    def query(self, request, response, event):
        query_started = time.monotonic_ns()
        received, reply = super().query(request, response, event)
        # Pair feedback to the successful request attempt. A timed-out read plus
        # a healthy retry is recovery latency, not the successful packet's RTT.
        sent, successful_received, attempts = self.query_timing.get(
            event, (query_started, received, 1))
        if successful_received != received:
            raise RuntimeError('遥测查询计时记录不一致')
        if attempts > 1:
            self.trace.record('telemetry_query_recovered',request=request,
                attempts=attempts,total_recovery_ms=(received-query_started)/1e6,
                successful_rtt_ms=(received-sent)/1e6)
        progress = True
        if event in ('diagnostic_pose', 'diagnostic_joint'):
            stamp = reply.get('data', {}).get('timestamp')
            if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not np.isfinite(stamp):
                raise RuntimeError('底层反馈缺少有限时间戳，停止外环')
            previous = self.source_clocks.get(event)
            if previous is not None:
                if stamp < previous[0]:
                    raise RuntimeError('机器人反馈时间戳倒退，停止外环')
                if stamp == previous[0]:
                    progress = False
                    if received-previous[1] > 500_000_000:
                        raise RuntimeError('底层反馈时间戳超过0.5秒未推进')
            if progress:
                self.source_clocks[event] = (stamp, received)
        if event == 'diagnostic_pose':
            self.packet = PosePacket(str(reply['guid']), sent, received,
                                     parse_dual_arm_pose(reply), stamp, progress)
        return received, reply


def prepare_envelope(snapshot, max_offset, lift):
    model, anchor, _, _, _, _, original = prepare(snapshot)
    q = np.asarray(snapshot['samples'][-1]['joint_reply']['data']['q'], dtype=float)
    t = np.linspace(0., 10., 301)
    positions = np.tile(anchor.left_position, (len(t), 1))
    positions[:, 2] += (max_offset+lift)*t/t[-1]
    rotations = np.tile(anchor.left_quaternion, (len(t), 1))
    check_workspace('left', positions)
    joints, metrics = model.plan_tracking(q, positions, rotations, t)
    for first, second in zip(joints[:-1], joints[1:]):
        error = _check_path(model, first, second, model.lower, model.upper)
        if error:
            raise RuntimeError('补偿包络模型检查失败：'+error)
    margins = np.minimum(joints[:, :7]-model.lower[:7], model.upper[:7]-joints[:, :7])
    if margins.min() < np.deg2rad(5.):
        raise RuntimeError('补偿包络最小关节余量小于5°')
    return model, anchor, {'max_offset_m': max_offset, 'nominal_lift_m': lift,
        'maximum_command_raise_m': max_offset+lift,
        'minimum_model_joint_margin_deg': float(np.rad2deg(margins.min())),
        'envelope_ik': metrics, 'original_diagnostic_plan': original,
        'reference_policy': 'fixed pre-start measured wrist pose, never reanchored after sag',
        'compensation_type': 'bounded Z position feedback, not torque/gravity identification',
        'missing_collision_geometry': ['actual O6', 'adapter', 'table', 'wires']}


class Trial:
    def __init__(self, client, telemetry, trace, model, anchor, initial_q, controller,
                 *, health_check=None):
        self.client, self.telemetry, self.trace = client, telemetry, trace
        self.model, self.anchor = model, anchor
        self.initial_q, self.controller = initial_q, controller
        self.latest_guid = None
        self.last_joint = None
        self.samples = []
        self.count = 0
        self.health_check = health_check

    def segment(self, phase, seconds, nominal, adaptive, *, finish_when_settled=False):
        start = time.monotonic_ns()
        if not adaptive:
            self.controller.freeze(start)
        self.trace.record('z_comp_phase', phase=phase, nominal_duration_s=seconds,
                          correction_enabled=adaptive, active_limit_m=self.controller.active_limit)
        recent = deque()
        stage_samples = []
        index = 0
        while True:
            due = start+int(index/300.*1e9)
            now = time.monotonic_ns()
            if now < due:
                time.sleep((due-now)/1e9)
            if self.health_check is not None:
                self.health_check()
            self.telemetry.require_fresh()
            joint_stamp, joint = self.telemetry.latest_joint
            packet = self.telemetry.packet
            if packet is None:
                raise RuntimeError('缺少完整的腕部反馈数据包')
            actual = packet.pose
            now = time.monotonic_ns()  # 必须晚于所取packet的接收时间。
            elapsed = (now-start)/1e9
            if elapsed > seconds:
                break
            if now-due > 100_000_000:
                raise RuntimeError('发送滞后超过100ms，不追赶轨迹')
            reference = np.r_[self.anchor.left_position, self.anchor.left_quaternion].copy()
            reference[2] += nominal(min(seconds, max(0., elapsed)))
            if packet is not None and packet.guid != self.latest_guid:
                self.latest_guid = packet.guid
                query_time = ((packet.sent_ns+packet.received_ns)//2-start)/1e9
                reference_at_query = self.anchor.left_position[2]+nominal(max(0., min(seconds, query_time)))
                accepted = self.controller.observe(packet.guid, packet.sent_ns, packet.received_ns,
                                                   now, float(reference_at_query),
                                                   float(packet.pose.left_position[2]),
                                                   adaptive and packet.source_progress)
                quality_ok = (now-packet.received_ns <= 150_000_000
                              and packet.received_ns-packet.sent_ns <= 100_000_000
                              and packet.source_progress)
                row = {'phase': phase, 'received_ns': packet.received_ns,
                       'nominal_z': float(reference_at_query),
                       'actual_z': float(packet.pose.left_position[2]),
                       'offset_m': self.controller.applied_offset,
                       'error_m': float(reference_at_query-packet.pose.left_position[2]),
                       'integrator_accepted': accepted, 'quality_ok': quality_ok}
                self.samples.append(row); stage_samples.append(row); recent.append(row)
                while recent and packet.received_ns-recent[0]['received_ns'] > 1_100_000_000:
                    recent.popleft()
                self.trace.record('z_comp_feedback', **row, request_guid=packet.guid,
                                  sent_monotonic_ns=packet.sent_ns,
                                  robot_timestamp_ms=packet.source_timestamp_ms,
                                  **pose_for_trace(packet.pose), controller_status=self.controller.status)
            offset = self.controller.advance(now)
            issued = reference.copy(); issued[2] += offset
            check_workspace('left', issued[None, :3])
            # 同时对真实物理参考和补偿后的内部目标验收，不能用修正掩盖失控。
            guard(self.anchor, reference, actual)
            guard(self.anchor, issued, actual)
            if np.linalg.norm(actual.left_position[:2]-reference[:2]) > .005:
                raise RuntimeError('横向反馈误差超过5mm，停止Z补偿试验')
            if self.last_joint != joint_stamp:
                q = np.asarray(joint['q'], dtype=float)
                self.model.check_joints(q)
                if np.minimum(q[:7]-self.model.lower[:7], self.model.upper[:7]-q[:7]).min() < np.deg2rad(3.):
                    raise RuntimeError('实测关节余量小于3°，停止补偿试验')
                if np.max(abs(q[7:14]-self.initial_q[7:14])) > .03 or np.max(abs(q[14:]-self.initial_q[14:])) > .03:
                    raise RuntimeError('右臂或头部出现非计划关节偏移')
                self.last_joint = joint_stamp
            failure = self.client.failure()
            if failure:
                raise RuntimeError('控制器拒绝ServoP：'+failure)
            right = np.r_[self.anchor.right_position, self.anchor.right_quaternion]
            self.trace.record('z_comp_command', index=self.count, phase=phase,
                              nominal_left=reference.tolist(), issued_left=issued.tolist(),
                              offset_m=offset, target_offset_m=self.controller.target_offset,
                              offset_applied_monotonic_ns=now,
                              due_monotonic_ns=due)
            self.client.send('request_servop', servop_data(issued, right))
            self.count += 1
            index += 1
            if (finish_when_settled and elapsed >= 2. and len(recent) >= 10
                    and recent[-1]['received_ns']-recent[0]['received_ns'] >= 900_000_000
                    and all(r['quality_ok'] and r['integrator_accepted'] for r in recent)
                    and max(abs(r['error_m']) for r in recent) <= .001):
                self.trace.record('z_comp_settled', phase=phase, elapsed_s=elapsed,
                                  final_offset_m=offset)
                return stage_samples, True
        return stage_samples, False


def mean_tail(samples, field):
    if len(samples) < 10:
        raise RuntimeError('阶段有效反馈不足')
    last = samples[-1]['received_ns']
    valid = [row for row in samples if last-row['received_ns'] <= 1_000_000_000
             and row['quality_ok']]
    if len(valid) < 10:
        raise RuntimeError('阶段末有效反馈不足，不认定方向/误差')
    return float(np.mean([row[field] for row in valid]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-correction-mm', type=float, default=5.)
    parser.add_argument('--lift-mm', type=float, default=0.)
    parser.add_argument('--countdown', type=int, default=5)
    args = parser.parse_args(argv)
    if not (5. <= args.max_correction_mm <= 20. and 0. <= args.lift_mm <= 20. and args.countdown >= 5):
        parser.error('上限须5–20mm，上抬须0–20mm，倒计时至少5秒')
    if args.max_correction_mm == 5. and args.lift_mm:
        parser.error('5mm模式仅验证方向；上下运动需显式给出更大的修正上限')
    if not args.execute and args.snapshot is None:
        parser.error('离线预检需--snapshot；真机需显式--execute')
    args.output.mkdir(parents=True, exist_ok=False)
    trace = TraceRecorder(args.output/'command_state.jsonl')
    client = telemetry = observer = lock = controller = expected_payload = None
    summary = {'captured_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'result': 'not_started', 'motion_requested': args.execute,
               'nominal_reference_reanchored': False, 'parameter_writes': 0}
    try:
        if args.execute:
            lock = acquire_capture_lock()
            client = Tron2JsonClient('10.192.1.2', trace)
            _ACTIVE_CLIENTS.append(client)
            observer = MotorObserver(trace); observer.start(client.robot)
            client.wait_for_identity()
            if client.accid != 'DACH_TRON2A_253':
                raise RuntimeError('机器人身份不符')
            snapshot = baseline(client)
            snapshot['configuration_readonly'] = readonly_configuration(client)
        else:
            snapshot = json.loads(args.snapshot.read_text())
        expected_payload = [.396, 0., 0., 0.] + snapshot['payload_reply']['data']['data'][4:]
        require_expected_payload(snapshot, expected_payload)
        (args.output/'baseline.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n')
        model, anchor, report = prepare_envelope(snapshot, args.max_correction_mm/1000., args.lift_mm/1000.)
        summary['plan'] = report
        (args.output/'plan.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if not args.execute:
            summary['result'] = 'planned_only'
            return 0
        for remain in range(args.countdown, 0, -1):
            print(f'{remain}秒后开始独立Z补偿试验；不执行Tracker回放。', flush=True)
            time.sleep(1)
        fresh = baseline(client)
        require_expected_payload(fresh, expected_payload)
        require_same_anchor(anchor, parse_dual_arm_pose(fresh['samples'][-1]['pose_reply']))
        prepare_envelope(fresh, args.max_correction_mm/1000., args.lift_mm/1000.)
        controller = ZCompensationController(Config(max_offset_m=args.max_correction_mm/1000.))
        summary['controller_config'] = asdict(controller.config)
        telemetry = CompensationTelemetry(client, trace); telemetry.thread.start()
        deadline = time.monotonic()+1.
        while telemetry.latest is None or telemetry.latest_joint is None or telemetry.packet is None:
            if telemetry.error or time.monotonic() > deadline:
                raise RuntimeError('首帧前缺少完整遥测')
            time.sleep(.005)
        require_same_anchor(anchor, telemetry.latest[1])
        trace.record('z_comp_start', **pose_for_trace(anchor), **report)
        trial = Trial(client, telemetry, trace, model, anchor,
                      np.asarray(fresh['samples'][-1]['joint_reply']['data']['q']), controller)
        initial, _ = trial.segment('baseline_hold', 3., lambda t: 0., False)
        direction, _ = trial.segment('direction_5mm', 10., lambda t: 0., True)
        applied = mean_tail(direction, 'offset_m')
        if applied < .0045:
            raise RuntimeError('5mm方向校验未达到足够修正量，不扩大范围')
        gain = (mean_tail(direction, 'actual_z')-mean_tail(initial, 'actual_z'))/applied
        summary['direction_trial_gain'] = gain
        trace.record('z_comp_direction_validation', gain=gain, applied_offset_m=applied)
        if not .5 <= gain <= 1.5:
            raise RuntimeError('5mm校验的实测方向/增益异常，不扩大修正范围')
        if args.max_correction_mm == 5.:
            summary['result'] = 'direction_trial_completed'
            summary['converged'] = False
        else:
            controller.set_active_limit(args.max_correction_mm/1000.)
            learned, settled = trial.segment('learn_hold', 25., lambda t: 0., True,
                                              finish_when_settled=True)
            if not settled:
                raise RuntimeError('有界修正未使高度误差稳定到1mm以内，不启动上下轨迹')
            summary['learned_offset_m'] = controller.applied_offset
            summary['learned_height_error_m'] = mean_tail(learned, 'error_m')
            if args.lift_mm:
                lift = args.lift_mm/1000.
                trial.segment('raise', 3., lambda t: lift*(t/3.)**2*(3.-2.*t/3.), False)
                trial.segment('upper_hold', 5., lambda t: lift, True)
                trial.segment('return', 3., lambda t: lift*(1.-(t/3.)**2*(3.-2.*t/3.)), False)
            final, final_settled = trial.segment('final_hold', 8., lambda t: 0., True,
                                                  finish_when_settled=True)
            summary['final_height_error_m'] = mean_tail(final, 'error_m')
            summary['converged'] = final_settled
            if not final_settled:
                raise RuntimeError('最终高度误差未稳定到1mm以内')
            summary['result'] = 'completed'
        summary['published_commands'] = trial.count
        summary['final_offset_m'] = controller.applied_offset
        trace.record('z_comp_sending_stopped', final_offset_m=controller.applied_offset)
        time.sleep(1.)
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(result='aborted', error=str(exc) or 'KeyboardInterrupt')
        trace.record('z_comp_abort', message=summary['error'])
        print('停止后续指令，不清零补偿或自动返回：'+summary['error'], flush=True)
    finally:
        if telemetry is not None:
            telemetry.stop.set(); telemetry.thread.join(timeout=1.)
        if client is not None:
            for kind, title in (('pose', 'servo_pose'), ('payload', 'payload_param')):
                try:
                    reply = client.request('request_get_'+title, {}, 'response_get_'+title, timeout=.5)
                    trace.record('z_comp_final_'+kind, reply=reply)
                    if kind == 'payload':
                        require_expected_payload({'payload_reply': reply}, expected_payload)
                except Exception as exc:
                    summary['final_'+kind+'_error'] = str(exc)
                    if kind == 'payload' and summary['result'] in ('completed', 'direction_trial_completed'):
                        summary['result'] = 'unverified_final_configuration'
        if observer is not None:
            observer.stop(); summary['motor_observer'] = observer.summary()
        if controller is not None:
            summary['retained_offset_m'] = controller.applied_offset
        summary['note'] = 'Z position feedback only; initial sag and orientation error are not claimed eliminated'
        (args.output/'result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        if lock is not None:
            lock.close()
        trace.close()
    return 0 if summary['result'] in ('completed', 'direction_trial_completed', 'planned_only') else 2


if __name__ == '__main__':
    import os
    code = main()
    if '--execute' in sys.argv:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(code)
    raise SystemExit(code)
