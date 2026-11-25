"""正式回放的显式 Z 位置外环分支；不发送力矩，不改变录制轨迹语义。"""

from collections import deque
import math
import time

import numpy as np

from replay_tron2_tracker import (PoseSample, ReplayInterrupted, FeedbackGuard,
    compose_pose, check_workspace, quaternion_distance_deg,
    servop_data, verify_observed_motion, SERVOP_ACTION_CONTRACT, pose_for_trace,
    check_anchor_stable)
from tools.diagnose_tron2_servop import strict_model
from tools.diagnose_tron2_z_compensation import CompensationTelemetry
from tron2_z_compensation import Config, ZCompensationController
from tools.tron2_raw_ws_client import RawWSClient


class NullTrace:
    def record(self, *args, **kwargs):
        pass


def nominal_quiet_flags(times, positions, quaternions):
    """只在完整6D名义目标足够静止时学习；运动时保留已验证的冻结策略。"""
    times = np.asarray(times)
    positions = np.asarray(positions)
    quaternions = np.asarray(quaternions)
    left = np.maximum(0, np.searchsorted(times, times-.05)-1)
    right = np.minimum(len(times)-1, np.searchsorted(times, times+.05))
    dt = np.diff(times)
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1)/dt
    angle_speed = np.array([quaternion_distance_deg(a, b)
                            for a, b in zip(quaternions[:-1], quaternions[1:])])/dt
    # 检查窗口中每段速度，不能让短促往返的净位移抵消后被误判为静止。
    return np.array([np.max(speed[a:b]) <= .005 and np.max(angle_speed[a:b]) <= 3.
                     for a, b in zip(left, right)], dtype=bool)


class ZReplaySession:
    """直接回放：修正从0开始，只在录制轨迹的静态片段学习。"""

    def __init__(self, client, anchor, args):
        self.client, self.anchor, self.args = client, anchor, args
        self.trace = client.trace or NullTrace()
        # 仅保留在线关节限位检查，不进行整段IK或碰撞包络规划。
        self.model = strict_model()
        config = Config(max_offset_m=args.z_max_correction_mm/1000.)
        self.controller = ZCompensationController(config, active_limit_m=config.max_offset_m)
        self.telemetry = None
        self.feedback_client = None
        self.guard = FeedbackGuard(anchor, 'left', feedback_timeout_s=args.feedback_timeout)
        self.samples = []
        self.history = deque(maxlen=2000)
        self.latest_guid = self.latest_joint = None
        self.initial_q = None
        self.adaptive = False
        self.last_stream_ns = None

    def start(self):
        """取得首组新鲜反馈，不发送保持指令、不等待高度收敛。"""
        # 不改变ServoP发送入口；关键反馈独立读取，避免排在SDK发送消息队列中。
        self.feedback_client = RawWSClient(getattr(self.args, 'robot_ip', '10.192.1.2'), self.trace)
        self.feedback_client.wait_for_identity()
        if self.feedback_client.accid != self.client.accid:
            raise RuntimeError('独立反馈连接与控制连接的机器人身份不一致')
        is_pose = getattr(self.args, 'pose_compensation', False)
        self.trace.record('pose_replay_feedback_transport' if is_pose else 'z_replay_feedback_transport', control='existing_sdk_client',
                          feedback='independent_raw_websocket')
        self.telemetry = CompensationTelemetry(self.feedback_client, self.trace)
        self.telemetry.thread.start()
        deadline = time.monotonic()+self.args.feedback_timeout
        while self.telemetry.packet is None or self.telemetry.latest_joint is None:
            if self.telemetry.error or time.monotonic() >= deadline:
                raise RuntimeError('直接回放首帧前未收到完整遥测')
            time.sleep(.005)
        self.telemetry.require_fresh()
        packet = self.telemetry.packet
        _, joint = self.telemetry.latest_joint
        q, dq = np.asarray(joint['q'], dtype=float), np.asarray(joint['dq'], dtype=float)
        if (q.shape != (16,) or dq.shape != (16,)
                or not np.isfinite(q).all() or not np.isfinite(dq).all()
                or np.max(abs(dq)) > .05):
            raise RuntimeError('直接回放首帧前关节反馈无效或尚未停稳')
        self.model.check_joints(q)
        check_anchor_stable(self.anchor, packet.pose)
        self.initial_q = q.copy()
        sample = PoseSample(packet.received_ns/1e9, packet.pose)
        self.samples.append(sample)
        self.guard.observe(sample, self.anchor.left_position, self.anchor.left_quaternion,
                           'direct_start')
        self.trace.record('pose_replay_direct_start' if is_pose else 'z_replay_direct_start', initial_offset_m=0.,
                          warmup=False, envelope_preflight=False, model_reference_preflight=is_pose, phase='feedback_only',
                          settle_handover_requested=getattr(self.args, 'settle_handover', False))

    def settle_handover(self, *, timeout_s=30., stable_s=1., z_tolerance_m=.001,
                        health_check=None):
        """在轨迹计时前恢复名义Z；不是力矩连续切换，不能阻止首瞬下沉。"""
        if any(not math.isfinite(v) or v <= 0 for v in (timeout_s, stable_s, z_tolerance_m)):
            raise ValueError('接管恢复超时、稳定时长和Z容差必须为正有限数值')
        if self.telemetry is None:
            raise RuntimeError('接管恢复前须先启动关键反馈')
        started = time.monotonic_ns()
        stable_since = last_valid = None
        seen_guid = None
        index = 0
        self.trace.record('z_handover_start', timeout_s=timeout_s, stable_s=stable_s,
                          z_tolerance_m=z_tolerance_m, torque_continuity_repaired=False)
        while True:
            due = started+int(index/self.args.rate*1e9)
            now = time.monotonic_ns()
            if now < due:
                time.sleep((due-now)/1e9)
            if health_check is not None:
                health_check()
            now = time.monotonic_ns()
            if now-started >= timeout_s*1e9:
                self.trace.record('z_handover_timeout', offset_m=self.controller.applied_offset)
                raise RuntimeError('ServoP接管后Z恢复未在限定时间内稳定；未开始Tracker/O6轨迹')
            if now-due > 100_000_000:
                raise RuntimeError('接管恢复发送滞后超过100ms，不追赶指令')
            self.stream(self.anchor.left_position, self.anchor.left_quaternion,
                        'settle_handover', adaptive=True)
            now = time.monotonic_ns()
            packet = self.telemetry.packet
            if last_valid is not None and now-last_valid > self.controller.config.feedback_max_gap_s*1e9:
                stable_since = None
            if packet.guid != seen_guid and packet.guid == self.latest_guid:
                seen_guid = packet.guid
                quality_ok = (packet.source_progress
                              and now-packet.received_ns <= self.controller.config.feedback_max_age_s*1e9
                              and packet.received_ns-packet.sent_ns <= self.controller.config.max_rtt_s*1e9)
                error = float(self.anchor.left_position[2]-packet.pose.left_position[2])
                gap_ok = (last_valid is None or packet.received_ns-last_valid
                          <= self.controller.config.feedback_max_gap_s*1e9)
                if not quality_ok or not gap_ok or abs(error) > z_tolerance_m:
                    stable_since = None
                if quality_ok:
                    last_valid = packet.received_ns
                    if abs(error) <= z_tolerance_m:
                        if stable_since is None:
                            stable_since = packet.received_ns
                        elif packet.received_ns-stable_since >= stable_s*1e9:
                            self.controller.freeze(now)
                            self.adaptive = False
                            self.trace.record('z_handover_ready', z_error_m=error,
                                              elapsed_s=(now-started)/1e9,
                                              retained_offset_m=self.controller.applied_offset,
                                              torque_continuity_repaired=False)
                            return
            index += 1

    def keep_holding_until(self, start_ns, *, health_check=None):
        """等待共享时钟时不断流；保留同一个补偿控制器及已输出偏置。"""
        while time.monotonic_ns() < start_ns:
            if health_check is not None:
                health_check()
            self.stream(self.anchor.left_position, self.anchor.left_quaternion,
                        'waiting_start', adaptive=False)
            remaining = start_ns-time.monotonic_ns()
            if remaining > 0:
                time.sleep(min(1/self.args.rate, remaining/1e9))

    def stream(self, nominal_position, nominal_quaternion, phase, adaptive=False):
        self.telemetry.require_fresh()
        packet = self.telemetry.packet
        joint_stamp, joint = self.telemetry.latest_joint
        now = time.monotonic_ns()
        if self.last_stream_ns is not None and now-self.last_stream_ns > 100_000_000:
            raise RuntimeError('补偿回放发送间隔超过100ms，停止后续轨迹')
        if self.adaptive and not adaptive:
            self.controller.freeze(now)
        self.adaptive = adaptive
        self.history.append((now, float(nominal_position[2]), adaptive))
        if packet.guid != self.latest_guid:
            self.latest_guid = packet.guid
            midpoint = (packet.sent_ns+packet.received_ns)//2
            times = np.array([r[0] for r in self.history], dtype=np.int64)
            at = max(0, int(np.searchsorted(times, midpoint, side='right')-1))
            ref_z = (self.history[at][1] if getattr(self.args, 'interpolation', 'linear') == 'none'
                     else float(np.interp(midpoint, times, [r[1] for r in self.history])))
            learn = adaptive and self.history[at][2] and packet.source_progress
            accepted = self.controller.observe(packet.guid, packet.sent_ns, packet.received_ns,
                                                now, ref_z, float(packet.pose.left_position[2]), learn)
            self.samples.append(PoseSample(packet.received_ns/1e9, packet.pose))
            self.guard.observe(self.samples[-1], nominal_position, nominal_quaternion, phase)
            self.trace.record('z_replay_feedback', phase=phase, request_guid=packet.guid,
                              nominal_z_m=ref_z, actual_z_m=float(packet.pose.left_position[2]),
                              offset_m=self.controller.applied_offset,
                              adaptation_enabled=learn, integrator_accepted=accepted,
                              sent_monotonic_ns=packet.sent_ns,
                              received_monotonic_ns=packet.received_ns,
                              robot_timestamp_ms=packet.source_timestamp_ms)
        self.guard.require_fresh(now/1e9)
        actual = packet.pose
        if np.linalg.norm(actual.left_position-nominal_position) > .05:
            raise RuntimeError('补偿回放腕部相对原物理目标误差超过50mm')
        if np.linalg.norm(actual.right_position-self.anchor.right_position) > .01:
            raise RuntimeError('补偿回放右腕相对保持起点偏移超过10mm')
        if self.latest_joint != joint_stamp:
            q = np.asarray(joint['q'], dtype=float)
            self.model.check_joints(q)
            if (np.max(abs(q[7:14]-self.initial_q[7:14])) > .03
                    or np.max(abs(q[14:]-self.initial_q[14:])) > .03):
                raise RuntimeError('补偿回放右臂或头部出现非计划偏移')
            self.latest_joint = joint_stamp
        offset = self.controller.advance(now)
        issued = np.asarray(nominal_position).copy(); issued[2] += offset
        check_workspace('left', issued[None, :])
        failure = self.client.failure()
        if failure:
            raise RuntimeError('ServoP拒绝补偿轨迹：'+failure)
        left = np.r_[issued, nominal_quaternion]
        right = np.r_[self.anchor.right_position, self.anchor.right_quaternion]
        self.trace.record('z_replay_command', phase=phase,
                          nominal_left_pos=np.r_[nominal_position, nominal_quaternion].tolist(),
                          issued_left_pos=left.tolist(), offset_m=offset,
                          offset_applied_monotonic_ns=now, adaptation_enabled=adaptive)
        self.client.send('request_servop', servop_data(left, right))
        self.last_stream_ns = now
        if time.monotonic_ns()-now > 100_000_000:
            raise RuntimeError('补偿指令发送阻塞超过100ms，停止后续轨迹')

    def final_pose(self):
        self.telemetry.require_fresh()
        reader = self.feedback_client or self.client
        pose = reader.current_servo_pose(label='final')
        q, dq = reader.joint_snapshot(label='post_z_replay')
        self.model.check_joints(q)
        if np.max(abs(dq[:14])) > .05:
            raise RuntimeError('补偿回放停发后关节仍在运动，未通过末帧验收')
        return pose

    def close(self):
        if self.telemetry is not None:
            self.telemetry.stop.set(); self.telemetry.thread.join(timeout=1.)
        if self.feedback_client is not None:
            self.feedback_client.close()
            self.feedback_client = None
        self.trace.record('z_replay_closed', retained_offset_m=self.controller.applied_offset,
                          correction_cleared=False)


def execute_z_compensated_replay(client, anchor, side, replay_times, delta_positions,
                                delta_quaternions, args, *, confirmation_word='DIRECT',
                                hand_replay=None):
    if side != 'left' or getattr(args, 'arm_control', 'servop') != 'servop':
        raise ValueError('Z补偿当前只支持左臂ServoP')
    positions = anchor.left_position[None, :]+delta_positions
    rotations = np.array([compose_pose(anchor.left_position, anchor.left_quaternion,
                                      p, r)[1] for p, r in zip(delta_positions, delta_quaternions)])
    adaptation = getattr(args, 'z_adaptation', 'stationary')
    if adaptation not in ('stationary', 'continuous'):
        raise ValueError('Z学习策略必须为stationary或continuous')
    quiet = (np.ones(len(replay_times), dtype=bool) if adaptation == 'continuous'
             else nominal_quiet_flags(replay_times, positions, rotations))
    session = None
    trace = client.trace or NullTrace()
    hand_owned = False
    start_ns = None
    settle = getattr(args, 'settle_handover', False)
    try:
        session = ZReplaySession(client, anchor, args)
        mode_label = ('先恢复接管后的名义Z，再同步回放Tracker/O6（位置补偿，不修复底层力矩切换）'
                      if settle else '直接回放Tracker/O6轨迹（无包络预检、无预热）')
        answer = input(f'将{mode_label}，输入 {confirmation_word} 继续: ')
        if answer.strip() != confirmation_word:
            raise RuntimeError('cancelled by user')
        if hand_replay is not None:
            hand_owned = True
            hand_replay.acquire()
        for remaining in range(args.countdown, 0, -1):
            if hand_replay is not None:
                hand_replay.require_healthy()
            print(f'{remaining}s后直接回放；实体急停在手边。', flush=True)
            time.sleep(1)
        session.start()
        if settle:
            print('开始ServoP接管Z恢复；首次瞬间下沉仍可能发生，轨迹时钟尚未启动。', flush=True)
            session.settle_handover(health_check=(hand_replay.require_healthy
                                                if hand_replay is not None else None))
        if hand_replay is not None:
            hand_replay.require_healthy()
        # 恢复成功后才安排手/臂共同起点；已进入ServoP时等待期间保持不断流。
        start_ns = time.monotonic_ns()+250_000_000
        if hand_replay is not None:
            hand_replay.start_at(start_ns)
        if time.monotonic_ns() >= start_ns:
            raise RuntimeError('共同起点安排超时，不追赶O6或机械臂轨迹')
        if settle:
            session.keep_holding_until(start_ns, health_check=(hand_replay.require_healthy
                                                             if hand_replay is not None else None))
        trace.record('replay_contract', **SERVOP_ACTION_CONTRACT, moving_arm='left', held_arm='right',
                     tracker_interpolation=getattr(args, 'interpolation', 'linear'),
                     z_compensation=True, z_max_correction_m=args.z_max_correction_mm/1000.,
                     z_adaptation=adaptation,
                     warmup=settle, settle_handover=settle, torque_continuity_repaired=False,
                     envelope_preflight=False,
                     tracker_mapping=getattr(args, 'tracker_mapping_info', None),
                     nominal_target_stream='z_replay_command.nominal_left_pos',
                     issued_target_stream='servop_command.left_pos',
                     with_o6_hand=hand_replay is not None, **pose_for_trace(anchor))
        trace.record('replay_start', side='left', start_monotonic_ns=start_ns,
                     arm_control='servop', z_compensation=True)
        print(('接管Z已恢复，沿用同一补偿量同步回放Tracker/O6。'
               if settle else '开始直接回放Tracker/O6，Z修正从0开始。'), flush=True)
        if getattr(args, 'web_control', False):
            print('ROBOT_REPLAY_EVENT {"phase":"replaying"}', flush=True)
        for i, due in enumerate(replay_times):
            target_ns = start_ns+int(float(due)*1e9)
            remaining = target_ns-time.monotonic_ns()
            if remaining > 0:
                time.sleep(remaining/1e9)
            if time.monotonic_ns()-target_ns > 100_000_000:
                raise RuntimeError('回放落后共同时间轴超过100ms，不追赶轨迹')
            if hand_replay is not None:
                hand_replay.require_healthy()
            session.stream(positions[i], rotations[i], 'trajectory', adaptive=bool(quiet[i]))
        # 末帧静态目标继续闭环2秒；手指数据仍使用原共同时间轴，已结束后保持末帧。
        final_hold_start = time.monotonic()
        for i in range(int(2.*args.rate)):
            due = final_hold_start+i/args.rate
            remaining = due-time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            if time.monotonic()-due > .1:
                raise RuntimeError('末帧保持发送落后超过100ms，不追赶指令')
            if hand_replay is not None:
                hand_replay.require_healthy()
            session.stream(positions[-1], rotations[-1], 'final_hold', adaptive=True)
        trace.record('z_replay_sending_stopped')
        time.sleep(.5)
        final_pose = session.final_pose()
        z_error = float(final_pose.left_position[2]-positions[-1, 2])
        trace.record('z_replay_final_height', nominal_z_m=float(positions[-1, 2]),
                     actual_z_m=float(final_pose.left_position[2]),
                     actual_minus_nominal_m=z_error, within_1mm=abs(z_error) <= .001)
        print(f'停发后末帧Z误差（实际－原目标）：{z_error*1000:+.2f}mm；'
              '完整轨迹验收与高度精度分别记录。', flush=True)
        samples = [p for p in session.samples if p.received_at >= start_ns/1e9]
        return verify_observed_motion(anchor, side, delta_positions, delta_quaternions, samples, final_pose)
    except BaseException as exc:
        # 先阻止已接管的手部sidecar继续自主回放，再关闭遥测或读取任何异常快照。
        if hand_owned and hand_replay is not None:
            try:
                hand_replay.close()
            except Exception as cleanup:
                trace.record('z_replay_hand_stop_error', message=str(cleanup))
        trace.record('control_abort', phase='z_compensated_replay', message=str(exc))
        if isinstance(exc, KeyboardInterrupt):
            raise ReplayInterrupted('Z补偿回放已中断，保留最后修正；软件停止不是急停') from exc
        raise
    finally:
        if session is not None:
            session.close()
