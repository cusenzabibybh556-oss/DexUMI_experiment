"""Experimental full-pose ServoP reference deployment; explicit opt-in only.

SDK remains the sole motion sender; raw websocket is feedback only. Local IK
does not force the firmware's branch. Hardware timing/gains remain unvalidated.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from replay_tron2_tracker import (PoseSample,ReplayInterrupted,compose_pose,check_workspace,
    quaternion_distance_deg,servop_data,SERVOP_ACTION_CONTRACT,pose_for_trace,verify_observed_motion)
from tron2_pose_reference import (VERSION,DEFAULT_PROFILE,load_profile,PoseTrajectory,
    PoseReferenceModel,preflight,limit_correction,clamp_correction)
from tron2_pose_compensation import PoseConfig,PoseCorrection
from tron2_z_replay import ZReplaySession,NullTrace


RIGHT_ARM_PASSIVE_DRIFT_LIMIT_RAD = .12
HEAD_PASSIVE_DRIFT_LIMIT_RAD = .03


def check_passive_joint_drift(q, initial_q):
    """Allow right-arm null-space settling while keeping the head guarded."""
    q = np.asarray(q, dtype=float)
    initial_q = np.asarray(initial_q, dtype=float)
    if q.shape != (16,) or initial_q.shape != (16,):
        raise RuntimeError('被动关节保护需要完整16关节反馈')
    right_drift = float(np.max(np.abs(q[7:14] - initial_q[7:14])))
    if right_drift > RIGHT_ARM_PASSIVE_DRIFT_LIMIT_RAD:
        raise RuntimeError(
            f'右臂冗余关节偏移{right_drift:.4f}rad超过'
            f'{RIGHT_ARM_PASSIVE_DRIFT_LIMIT_RAD:.2f}rad'
        )
    head_drift = float(np.max(np.abs(q[14:16] - initial_q[14:16])))
    if head_drift > HEAD_PASSIVE_DRIFT_LIMIT_RAD:
        raise RuntimeError(
            f'头部出现非计划关节偏移{head_drift:.4f}rad超过'
            f'{HEAD_PASSIVE_DRIFT_LIMIT_RAD:.2f}rad'
        )


def add_pose_arguments(parser):
    parser.add_argument('--pose-compensation',action='store_true',help='显式启用最新实验全位姿模型；与Z补偿互斥')
    parser.add_argument('--pose-profile',type=Path,default=DEFAULT_PROFILE)
    parser.add_argument('--pose-preview-ms',type=float,default=None,help='模型预瞄假设；默认124ms，未完成硬件标定')
    parser.add_argument('--pose-feedback-age-ms',type=float,default=None,help='内部位姿年龄假设；默认0，按查询中点配对')
    parser.add_argument('--pose-report',type=Path,help='保存完整模型参考预检；不覆盖已有报告')


def validate_pose_options(args):
    enabled=getattr(args,'pose_compensation',False)
    overrides=(getattr(args,'pose_preview_ms',None),getattr(args,'pose_feedback_age_ms',None),getattr(args,'pose_report',None))
    if not enabled:
        if any(v is not None for v in overrides) or Path(getattr(args,'pose_profile',DEFAULT_PROFILE))!=DEFAULT_PROFILE:
            raise ValueError('pose参数须显式配合 --pose-compensation')
        return
    if getattr(args,'z_compensation',False) or getattr(args,'settle_handover',False) or getattr(args,'z_adaptation','stationary')!='stationary':
        raise ValueError('全位姿模式与旧Z补偿/settle-handover互斥；新模式自动进行完整位姿接管恢复')
    if getattr(args,'arm_control','servop')!='servop' or args.rate!=300 or getattr(args,'interpolation','linear')!='linear' or getattr(args,'plan_ik',False):
        raise ValueError('全位姿实验只支持左臂300Hz linear ServoP；预检由新模型完成')
    if not 0<args.speed<=.5 or not 0<args.scale<=1:
        raise ValueError('当前全位姿真机实验仅接受0<speed<=0.5、0<scale<=1，不放行已知高速/放大失败范围')
    profile=load_profile(args.pose_profile)
    if args.pose_preview_ms is None:args.pose_preview_ms=profile['preview_s']*1000
    if args.pose_feedback_age_ms is None:args.pose_feedback_age_ms=profile['assumed_feedback_age_s']*1000
    if not math.isfinite(args.pose_preview_ms) or not 0<=args.pose_preview_ms<=200:
        raise ValueError('pose-preview-ms必须在0..200ms')
    if not math.isfinite(args.pose_feedback_age_ms) or not 0<=args.pose_feedback_age_ms<=100:
        raise ValueError('pose-feedback-age-ms必须在0..100ms')


def prepare_pose_replay(anchor,q_start,times,deltas,quaternions,args,trace=None):
    validate_pose_options(args)
    profile=load_profile(args.pose_profile)
    positions=np.asarray(anchor.left_position)+np.asarray(deltas)
    rotations=np.array([compose_pose(anchor.left_position,anchor.left_quaternion,p,r)[1] for p,r in zip(deltas,quaternions)])
    trajectory=PoseTrajectory(times,positions,rotations)
    report=preflight(trajectory,q_start,profile,args.pose_preview_ms/1000)
    report.update(assumed_feedback_age_ms=args.pose_feedback_age_ms,
        feedback_clock_contract=profile['feedback_contract'],model_gains_hardware_validated=False,
        handover_ramp_s=profile['handover_model_ramp_s'])
    args._pose_trajectory=trajectory;args._pose_profile=profile;args._pose_q_start=np.asarray(q_start).copy()
    if getattr(args,'pose_report',None) is not None:
        path=Path(args.pose_report);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    if trace is not None:trace.record('pose_reference_preflight',**report)
    print(f'控制器版本：{VERSION}；前馈＋预瞄＋XYZ/朝向残差反馈。')
    print(f'全位姿预检：{report["trajectory_frames"]}帧；最大模型平移修正{report["maximum_model_shift_mm"]:.2f}mm，'
          f'旋转修正{report["maximum_model_rotation_deg"]:.2f}°。')
    print(f'预瞄{args.pose_preview_ms:g}ms / 反馈年龄假设{args.pose_feedback_age_ms:g}ms；均未完成真机时序标定。')
    return report


class PoseReplaySession(ZReplaySession):
    def __init__(self,client,anchor,args):
        super().__init__(client,anchor,args)
        self.profile=args._pose_profile;self.trajectory=args._pose_trajectory
        freshness_s=float(getattr(args,'pose_input_freshness_ms',150.))/1000
        self.pose_controller=PoseCorrection(PoseConfig(
            max_feedback_age_s=freshness_s,
            max_feedback_gap_s=freshness_s,
            max_rtt_s=freshness_s,
        ));self.engine=None
        self.start_ns=None;self.takeover_ns=None;self.last_correction_p=np.zeros(3);self.last_correction_r=np.eye(3)
        self.last_issued=None

    def start(self):
        from tools.diagnose_tron2_servop import require_expected_payload
        payload=self.client.request('request_get_payload_param',{},'response_get_payload_param')
        require_expected_payload({'payload_reply':payload},[.396,0,0,0,.001,0,0,0])
        self.trace.record('pose_reference_payload_verified',reply=payload,writes=0)
        super().start()
        if np.max(abs(self.initial_q-self.args._pose_q_start))>.01:
            raise RuntimeError('全位姿预检后关节起点改变，未开始ServoP')
        self.engine=PoseReferenceModel(self.profile,self.initial_q)
        self.engine.check_anchor(self.initial_q,self.anchor.left_position,Rotation.from_quat(self.anchor.left_quaternion[[1,2,3,0]]).as_matrix())
        self.trace.record('pose_replay_ready',controller_version=VERSION,profile_sha256=self.profile['profile_sha256'],
            pose_feedback_config=asdict(self.pose_controller.config),preview_ms=self.args.pose_preview_ms,
            assumed_feedback_age_ms=self.args.pose_feedback_age_ms,hardware_timing_calibrated=False)

    def reference_at_ns(self,stamp):
        if self.start_ns is None:
            return self.trajectory.target(0)
        return self.trajectory.target((stamp-self.start_ns)/1e9)

    def stream(self,phase,reference_time_s=None,adaptive=True):
        compute_start=time.perf_counter();self.telemetry.require_fresh()
        packet=self.telemetry.packet;joint_stamp,joint=self.telemetry.latest_joint
        now=time.monotonic_ns()
        if self.last_stream_ns is not None and now-self.last_stream_ns>100_000_000:
            raise RuntimeError('全位姿ServoP发送间隔超过100ms')
        if self.takeover_ns is None:self.takeover_ns=now
        if reference_time_s is None:
            reference_time_s=(now-self.start_ns)/1e9 if self.start_ns is not None else 0.
        nominal_p,nominal_r=self.trajectory.target(reference_time_s)
        if packet.guid!=self.latest_guid:
            self.latest_guid=packet.guid
            age=round(self.args.pose_feedback_age_ms*1e6)
            paired=(packet.sent_ns+packet.received_ns)//2-age
            rp,rr=self.reference_at_ns(paired)
            accepted=False
            if adaptive and packet.source_progress:
                accepted=self.pose_controller.observe(packet.guid,packet.sent_ns-age,packet.received_ns-age,now,
                    rp,rr,packet.pose.left_position,Rotation.from_quat(packet.pose.left_quaternion[[1,2,3,0]]).as_matrix())
            else:self.pose_controller.freeze()
            self.samples.append(PoseSample(packet.received_ns/1e9,packet.pose))
            self.guard.observe(self.samples[-1],nominal_p,Rotation.from_matrix(nominal_r).as_quat()[[3,0,1,2]],phase)
            self.trace.record('pose_replay_feedback',phase=phase,request_guid=packet.guid,
                nominal_left_pos=np.r_[rp,Rotation.from_matrix(rr).as_quat()[[3,0,1,2]]].tolist(),
                actual_left_pos=np.r_[packet.pose.left_position,packet.pose.left_quaternion].tolist(),
                feedback_pair_ns=paired,sent_monotonic_ns=packet.sent_ns,received_monotonic_ns=packet.received_ns,
                assumed_feedback_age_ms=self.args.pose_feedback_age_ms,integrator_accepted=accepted,
                robot_timestamp_ms=packet.source_timestamp_ms)
        self.guard.require_fresh(now/1e9)
        if np.linalg.norm(packet.pose.left_position-nominal_p)>.05:raise RuntimeError('全位姿相对原目标误差超过50mm')
        if quaternion_distance_deg(packet.pose.left_quaternion,Rotation.from_matrix(nominal_r).as_quat()[[3,0,1,2]])>15:
            raise RuntimeError('全位姿实际朝向误差超过15deg')
        if np.linalg.norm(packet.pose.right_position-self.anchor.right_position)>.01:raise RuntimeError('全位姿右腕保持偏移超过10mm')
        q=np.asarray(joint['q']);self.model.check_joints(q)
        check_passive_joint_drift(q,self.initial_q)
        residual_p,residual_r=self.pose_controller.advance(now)
        preview=self.args.pose_preview_ms/1000 if self.start_ns is not None else 0.
        preview_p,preview_r=self.trajectory.target(reference_time_s+preview)
        ramp=np.clip((now-self.takeover_ns)/1e9/self.profile['handover_model_ramp_s'],0,1)
        ramp=float(ramp*ramp*(3-2*ramp))
        dt=(now-self.last_stream_ns)/1e9 if self.last_stream_ns is not None else 1/300
        snapshot=getattr(type(self.engine),'snapshot_state',None)
        model_state=snapshot(self.engine) if snapshot is not None else None
        issued_p,issued_r,model=self.engine.step(preview_p+residual_p,residual_r@preview_r,dt,weight=ramp)
        correction_p=issued_p-preview_p;correction_r=issued_r@preview_r.T
        correction_p,correction_r,total_limited=clamp_correction(
            correction_p,correction_r,self.profile['maximum_total_translation_m'],
            self.profile['maximum_total_rotation_deg'])
        if total_limited:
            issued_p=preview_p+correction_p;issued_r=correction_r@preview_r
            issued_p,issued_r,fraction=self.engine.constrain_issued(issued_p,issued_r)
            model['total_correction_ik_fraction']=float(fraction)
            correction_p=issued_p-preview_p;correction_r=issued_r@preview_r.T
        model['total_correction_limited']=total_limited
        correction_p,correction_r,limited=limit_correction(self.last_correction_p,self.last_correction_r,
            correction_p,correction_r,dt,self.profile['maximum_correction_rate_m_s'],self.profile['maximum_correction_rate_deg_s'])
        if limited:
            issued_p=preview_p+correction_p;issued_r=correction_r@preview_r
            issued_p,issued_r,fraction=self.engine.constrain_issued(issued_p,issued_r)
            model['slew_ik_fraction']=float(fraction)
            correction_p=issued_p-preview_p;correction_r=issued_r@preview_r.T
        check_workspace('left',issued_p[None,:])
        failure=self.client.failure()
        if failure:raise RuntimeError('ServoP拒绝全位姿轨迹：'+failure)
        elapsed=time.perf_counter()-compute_start
        if elapsed>.02:
            if model_state is not None:self.engine.restore_state(model_state)
            held=(self.last_issued.copy() if self.last_issued is not None else
                  np.r_[self.anchor.left_position,self.anchor.left_quaternion])
            self.trace.record('pose_replay_compute_overrun',phase=phase,calculation_ms=elapsed*1000,
                action='hold_last_valid_target',held_left_pos=held.tolist())
            self.client.send('request_servop',servop_data(
                held,np.r_[self.anchor.right_position,self.anchor.right_quaternion]))
            self.last_stream_ns=time.monotonic_ns()
            return
        issued=np.r_[issued_p,Rotation.from_matrix(issued_r).as_quat()[[3,0,1,2]]]
        self.trace.record('pose_replay_command',controller_version=VERSION,phase=phase,reference_time_s=reference_time_s,
            nominal_left_pos=np.r_[nominal_p,Rotation.from_matrix(nominal_r).as_quat()[[3,0,1,2]]].tolist(),
            issued_left_pos=issued.tolist(),correction_xyz_m=correction_p.tolist(),
            correction_rotation_vector_rad=Rotation.from_matrix(correction_r).as_rotvec().tolist(),
            residual_xyz_m=residual_p.tolist(),residual_rotation_vector_rad=Rotation.from_matrix(residual_r).as_rotvec().tolist(),
            model=model,ramp=ramp,correction_rate_limited=limited,calculation_ms=elapsed*1000,offset_applied_monotonic_ns=now)
        self.client.send('request_servop',servop_data(issued,np.r_[self.anchor.right_position,self.anchor.right_quaternion]))
        self.last_stream_ns=now;self.last_correction_p=correction_p;self.last_correction_r=correction_r;self.last_issued=issued
        if time.monotonic_ns()-now>100_000_000:raise RuntimeError('全位姿SDK发送阻塞超过100ms')

    def settle_pose(self,health_check=None):
        started=time.monotonic_ns();stable=None;seen=None;last_valid=None;index=0
        self.trace.record('pose_handover_start',ramp_s=self.profile['handover_model_ramp_s'],position_tolerance_m=.003,angle_tolerance_deg=1.,timeout_s=45.)
        while True:
            due=started+round(index/300*1e9);remaining=due-time.monotonic_ns()
            if remaining>0:time.sleep(remaining/1e9)
            if time.monotonic_ns()-due>100_000_000:raise RuntimeError('全位姿接管排程超时')
            if health_check:health_check()
            self.stream('settle_pose')
            now=time.monotonic_ns();packet=self.telemetry.packet
            if now-started>45_000_000_000:raise RuntimeError('45秒内全位姿未稳定，未启动Tracker/O6轨迹')
            if packet.guid!=seen:
                seen=packet.guid
                c=self.pose_controller.config
                valid=(packet.source_progress
                       and now-packet.received_ns<=c.max_feedback_age_s*1e9
                       and packet.received_ns-packet.sent_ns<=c.max_rtt_s*1e9)
                within=(np.linalg.norm(packet.pose.left_position-self.anchor.left_position)<=.003 and
                        quaternion_distance_deg(packet.pose.left_quaternion,self.anchor.left_quaternion)<=1.)
                gap_ok=(last_valid is None or packet.received_ns-last_valid
                        <=c.max_feedback_gap_s*1e9)
                if not gap_ok:stable=None
                if valid:last_valid=packet.received_ns
                if valid and within and gap_ok and now-started>=self.profile['handover_model_ramp_s']*1e9:
                    stable=stable or packet.received_ns
                    if packet.received_ns-stable>=1_000_000_000:
                        self.pose_controller.freeze();self.trace.record('pose_handover_ready',elapsed_s=(now-started)/1e9)
                        return
                else:stable=None
            index+=1

    def hold_until(self,stamp,health_check=None):
        while time.monotonic_ns()<stamp:
            if health_check:health_check()
            self.stream('waiting_start',adaptive=False)
            remaining=stamp-time.monotonic_ns()
            if remaining>0:time.sleep(min(1/300,remaining/1e9))

    def close(self):
        if self.telemetry is not None:
            self.telemetry.stop.set();self.telemetry.thread.join(timeout=1.)
        if self.feedback_client is not None:self.feedback_client.close();self.feedback_client=None
        self.trace.record('pose_replay_closed',controller_version=VERSION,
            retained_correction_xyz_m=self.last_correction_p.tolist(),correction_cleared=False)


def execute_pose_compensated_replay(client,anchor,side,times,positions,quaternions,args,*,confirmation_word='REPLAY',hand_replay=None):
    validate_pose_options(args)
    if side!='left':raise ValueError('全位姿实验只支持左臂')
    trace=client.trace or NullTrace();session=None;hand_owned=False
    try:
        if not hasattr(args,'_pose_trajectory'):
            q,dq=client.joint_snapshot(label='pose_preflight')
            if np.max(abs(dq))>.05:raise RuntimeError('全位姿预检前关节未停稳')
            prepare_pose_replay(anchor,q,times,positions,quaternions,args,trace)
        session=PoseReplaySession(client,anchor,args)
        answer=input(f'实验控制器 {VERSION}；使用未硬件标定的时序/动力学假设。输入 {confirmation_word} 继续: ')
        if answer.strip()!=confirmation_word:raise RuntimeError('cancelled by user')
        if hand_replay is not None:hand_owned=True;hand_replay.acquire()
        for value in range(args.countdown,0,-1):
            if hand_replay is not None:hand_replay.require_healthy()
            print(f'{value}s后开始实验全位姿接管；实体急停在旁。',flush=True);time.sleep(1)
        session.start()
        health=hand_replay.require_healthy if hand_replay is not None else None
        session.settle_pose(health)
        start=time.monotonic_ns()+250_000_000;session.start_ns=start
        if hand_replay is not None:hand_replay.start_at(start)
        if time.monotonic_ns()>=start:raise RuntimeError('全位姿手臂共同起点安排超时')
        session.hold_until(start,health)
        trace.record('replay_contract',**SERVOP_ACTION_CONTRACT,controller_version=VERSION,
            pose_compensation=True,z_compensation=False,preview_ms=args.pose_preview_ms,
            assumed_feedback_age_ms=args.pose_feedback_age_ms,timing_hardware_calibrated=False,
            model_gains_hardware_validated=False,profile_sha256=args._pose_profile['profile_sha256'],
            nominal_target_stream='pose_replay_command.nominal_left_pos',issued_target_stream='servop_command.left_pos',
            with_o6_hand=hand_replay is not None,**pose_for_trace(anchor))
        trace.record('replay_start',side=side,start_monotonic_ns=start,controller_version=VERSION,pose_compensation=True)
        for due in times:
            stamp=start+round(float(due)*1e9);remaining=stamp-time.monotonic_ns()
            if remaining>0:time.sleep(remaining/1e9)
            if time.monotonic_ns()-stamp>100_000_000:raise RuntimeError('全位姿轨迹排程落后超过100ms')
            if health:health()
            session.stream('trajectory',float(due))
        tail=time.monotonic_ns()
        for i in range(600):
            due=tail+round(i/300*1e9);remaining=due-time.monotonic_ns()
            if remaining>0:time.sleep(remaining/1e9)
            if time.monotonic_ns()-due>100_000_000:raise RuntimeError('全位姿末帧保持排程超时')
            if health:health()
            session.stream('final_hold',float(times[-1]))
        trace.record('pose_replay_sending_stopped');time.sleep(.5)
        pose=session.final_pose();target=args._pose_trajectory.positions[-1]
        trace.record('pose_replay_final_error',actual_minus_nominal_xyz_m=(pose.left_position-target).tolist(),
            rotation_error_deg=quaternion_distance_deg(pose.left_quaternion,
                Rotation.from_matrix(args._pose_trajectory.target(times[-1])[1]).as_quat()[[3,0,1,2]]))
        return verify_observed_motion(anchor,side,positions,quaternions,
            [p for p in session.samples if p.received_at>=start/1e9],pose)
    except BaseException as exc:
        if hand_owned and hand_replay is not None:
            try:hand_replay.close()
            except Exception as cleanup:trace.record('pose_hand_stop_error',message=str(cleanup))
        trace.record('control_abort',phase='pose_compensated_replay',controller_version=VERSION,message=str(exc))
        if isinstance(exc,KeyboardInterrupt):raise ReplayInterrupted('全位姿回放中断；保留最后参考，软件停止不是急停') from exc
        raise
    finally:
        if session is not None:session.close()
