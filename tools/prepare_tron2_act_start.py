#!/usr/bin/env python3
"""Prepare the config-defined ACT left-arm start from the original fixed start.

This is an isolated MoveJ experiment.  It never starts ACT, ServoP, camera, or
O6 control.  The current experiment only accepts the reviewed 20 mm downward
translation with the original wrist orientation and requires the robot to be
at the original fixed start before moving.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from prepare_tron2_arms import move_stage
from replay_tron2_fixed_start import (
    FIXED_LEFT_POSITION,FIXED_LEFT_Q,FIXED_LEFT_QUATERNION,FIXED_ROBOT_ID,
    acquire_replay_lock,countdown,
)
from replay_tron2_tracker import (
    DEFAULT_ROBOT_IP,TraceRecorder,Tron2JsonClient,default_trace_path,
    quaternion_distance_deg,
)
from tools.deploy_tron2_act import act_start,load_runtime


DEFAULT_CONFIG=ROOT/'configs/tron2_act_deploy_chunk125_last_zminus20.json'
TARGET_OFFSET=np.array([0.,0.,-.020])


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--duration',type=float,default=3.)
    parser.add_argument('--countdown',type=int,default=5)
    parser.add_argument('--robot-ip',default=DEFAULT_ROBOT_IP)
    return parser.parse_args(argv)


def check_target(config):
    """Offline kinematic and direct MoveJ-path review; sends no commands."""
    from tron2_left_ik import LeftArmIK
    required=('initial_left_q_rad','initial_left_position_m','initial_left_quaternion_wxyz')
    if not all(field in config for field in required):
        raise ValueError('This tool requires an explicit ACT start in --config')
    target_q,target_p,target_u=act_start(config)
    expected=FIXED_LEFT_POSITION+TARGET_OFFSET
    if (not np.allclose(target_p,expected,atol=1e-9) or
            quaternion_distance_deg(target_u,FIXED_LEFT_QUATERNION)>1e-6):
        raise ValueError('This reviewed tool only accepts original start Z minus 20 mm with unchanged orientation')
    model=LeftArmIK()
    start=np.r_[FIXED_LEFT_Q,np.zeros(9)]
    target=np.r_[target_q,np.zeros(9)]
    model.check_anchor(start,FIXED_LEFT_POSITION,FIXED_LEFT_QUATERNION)
    model.check_anchor(target,target_p,target_u)
    delta=target_q-FIXED_LEFT_Q
    maximum=float(np.max(np.abs(delta)))
    if maximum>math.radians(5)+1e-9:
        raise RuntimeError('Reviewed lower-start MoveJ exceeds 5 degrees on one joint')
    margin=float(np.min(np.minimum(target-model.lower,model.upper-target)))
    return target_q,target_p,target_u,{
        'offset_m':TARGET_OFFSET.tolist(),
        'maximum_joint_change_deg':math.degrees(maximum),
        'minimum_target_joint_margin_deg':math.degrees(margin),
        'model':str(model.model_path),
        'missing_geometry':['actual_O6','mount','table','wires'],
    }


def review_runtime_path(current,target_left):
    """Check the exact straight joint interpolation sent to MoveJ."""
    from tron2_left_ik import LeftArmIK
    model=LeftArmIK();target=np.asarray(current,float).copy();target[:7]=target_left
    model.check_joints(current);model.check_joints(target)
    m,d,mj=model.model,model.data,model.mj
    left_geoms=set()
    for geom in range(m.ngeom):
        body=m.body(int(m.geom_bodyid[geom])).name
        if body.startswith('grasper_'):
            m.geom_contype[geom]=0;m.geom_conaffinity[geom]=0
        elif body.endswith('_L_Link') and (m.geom_contype[geom] or m.geom_conaffinity[geom]):
            left_geoms.add(geom)
    for i,beta in enumerate(np.linspace(0.,1.,1001)):
        sample=current+beta*(target-current);model.check_joints(sample)
        d.qpos[model.qadr]=sample;mj.mj_forward(m,d)
        for contact in d.contact:
            if contact.dist<0 and (int(contact.geom1) in left_geoms or int(contact.geom2) in left_geoms):
                names=[m.body(int(m.geom_bodyid[g])).name for g in (contact.geom1,contact.geom2)]
                raise RuntimeError(f'Lower-start model collision at {i/10:.1f}%: {names}')
    return target


def pose_errors(pose,target_p,target_u):
    return (float(np.linalg.norm(pose.left_position-target_p)),
            float(quaternion_distance_deg(pose.left_quaternion,target_u)))


def require_original_start(q,pose):
    joint=float(np.max(np.abs(q[:7]-FIXED_LEFT_Q)))
    position,angle=pose_errors(pose,FIXED_LEFT_POSITION,FIXED_LEFT_QUATERNION)
    if joint>.04 or position>.02 or angle>3.:
        raise RuntimeError(
            f'Left arm must first be restored to the original fixed start; '
            f'joint={joint:.4f}rad, pose={position*1000:.1f}mm/{angle:.2f}deg')


def main(argv=None):
    args=parse_args(argv)
    if not math.isfinite(args.duration) or not 2<=args.duration<=10:
        raise ValueError('--duration must be within 2..10 seconds')
    if not 5<=args.countdown<=30:
        raise ValueError('--countdown must be within 5..30 seconds')
    config=load_runtime(args.config);target_q,target_p,target_u,report=check_target(config)
    print('ACT experimental start (left wrist):',target_p.tolist())
    print('Left joint target (rad):',target_q.tolist())
    print(f"Offline IK valid; maximum joint change {report['maximum_joint_change_deg']:.2f}deg; "
          f"minimum target margin {report['minimum_target_joint_margin_deg']:.2f}deg.")
    print('Original fixed start remains unchanged; O6/right arm/head are not commanded by this tool.')
    if not args.execute:
        print('Offline review only; no robot connection or motion command.')
        return 0

    lock=trace=client=None
    try:
        lock=acquire_replay_lock();trace=TraceRecorder(default_trace_path())
        trace.record('act_lower_start_plan',config=str(args.config.resolve()),**report,
                     target_q=target_q.tolist(),target_position=target_p.tolist(),
                     target_quaternion_wxyz=target_u.tolist(),duration_s=args.duration)
        print('Trace:',trace.path)
        client=Tron2JsonClient(args.robot_ip,trace=trace);client.wait_for_identity()
        if client.accid!=FIXED_ROBOT_ID:raise RuntimeError('Unexpected robot identity: '+client.accid)
        q,dq=client.joint_snapshot('before_act_lower_start');pose=client.current_servo_pose('before_act_lower_start')
        if np.max(np.abs(dq))>.05:raise RuntimeError('Robot is not stationary')
        target_position_error,target_angle_error=pose_errors(pose,target_p,target_u)
        if (np.max(np.abs(q[:7]-target_q))<=.015 and target_position_error<=.005
                and target_angle_error<=1.):
            print('Left arm is already at the Z-minus-20mm ACT start; no MoveJ sent.')
            return 0
        require_original_start(q,pose)
        review_runtime_path(q,target_q)
        print('Only the left arm will move 20 mm downward from the original fixed start.')
        print('The local model omits the actual O6, adapter, table, and wires; clear the physical sweep area.')
        try:answer=input('Input LOWER to continue: ').strip()
        except EOFError as exc:raise RuntimeError('No LOWER confirmation; no MoveJ sent') from exc
        if answer!='LOWER':raise RuntimeError('cancelled by user')
        countdown(args.countdown)
        fresh_q,fresh_dq=client.joint_snapshot('act_lower_start_pre_move')
        fresh_pose=client.current_servo_pose('act_lower_start_pre_move')
        if np.max(np.abs(fresh_dq))>.05:raise RuntimeError('Robot moved during countdown')
        require_original_start(fresh_q,fresh_pose)
        if np.max(np.abs(fresh_q-q))>.005:
            raise RuntimeError('Robot joint state changed during confirmation; rerun preparation')
        review_runtime_path(fresh_q,target_q)
        target14=fresh_q[:14].copy();target14[:7]=target_q
        move_stage(client,'ACT Z-minus-20mm start','left',target14,slice(0,7),args.duration,.015)
        final_q,final_dq=client.joint_snapshot('after_act_lower_start')
        final_pose=client.current_servo_pose('after_act_lower_start')
        position_error,angle_error=pose_errors(final_pose,target_p,target_u)
        joint_error=float(np.max(np.abs(final_q[:7]-target_q)))
        if (joint_error>.015 or position_error>.005 or angle_error>1. or
                np.max(np.abs(final_dq))>.05):
            raise RuntimeError(
                f'Lower start verification failed: joint={joint_error:.4f}rad, '
                f'pose={position_error*1000:.1f}mm/{angle_error:.2f}deg')
        trace.record('act_lower_start_verified',joint_error_rad=joint_error,
                     position_error_m=position_error,angle_error_deg=angle_error)
        print(f'ACT Z-minus-20mm start verified: {position_error*1000:.1f}mm / {angle_error:.2f}deg.')
        return 0
    finally:
        if client is not None:
            try:client.close()
            except Exception:pass
        if trace is not None:trace.close()
        if lock is not None:lock.close()


if __name__=='__main__':
    code=0
    try:code=main()
    except (OSError,ValueError,RuntimeError,TimeoutError) as exc:
        print('error:',exc,file=sys.stderr);code=2
    if '--execute' in sys.argv:
        sys.stdout.flush();sys.stderr.flush();os._exit(code)
    raise SystemExit(code)
