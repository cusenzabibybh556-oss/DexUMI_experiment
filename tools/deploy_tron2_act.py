#!/usr/bin/env python3
"""ACT real-robot entry point. Default offline; --shadow is read-only; --execute sends ServoP/O6."""
import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def require_initial_o6_state(config, measured):
    import numpy as np
    expected=np.asarray(config['initial_o6_qpos'],dtype=float)
    measured=np.asarray(measured,dtype=float)
    tolerance=float(config['initial_o6_tolerance'])
    if expected.shape!=(6,) or measured.shape!=(6,) or not np.isfinite(measured).all():
        raise RuntimeError('Invalid O6 initial-state feedback')
    error=np.abs(measured-expected)
    if np.max(error)>tolerance:
        raise RuntimeError(
            'O6 is not at ACT initial state; run tools/reset_o6_act_start.py first. '
            f'measured={np.rint(measured).astype(int).tolist()}, '
            f'expected={np.rint(expected).astype(int).tolist()}, '
            f'max_error={float(np.max(error)):.1f}'
        )
    return error


def load_runtime(path):
    c=json.loads(Path(path).read_text())
    import numpy as np
    for k,limit in [('max_displacement_m',.35),('max_angle_deg',90.),('startup_ramp_s',10.)]:
        if isinstance(c[k],bool) or not np.isfinite(c[k]) or not 0<c[k]<=limit:raise ValueError('Invalid '+k)
    if c['max_duration_s'] is not None and (isinstance(c['max_duration_s'],bool)
            or not np.isfinite(c['max_duration_s']) or c['max_duration_s']<=0):
        raise ValueError('Invalid max_duration_s')
    if (isinstance(c['execution_speed'],bool) or not np.isfinite(c['execution_speed'])
            or not .1<=c['execution_speed']<=1.):
        raise ValueError('Invalid execution_speed')
    if (isinstance(c['io_freshness_ms'],bool) or not np.isfinite(c['io_freshness_ms'])
            or not 100<=c['io_freshness_ms']<=1000):
        raise ValueError('Invalid io_freshness_ms')
    if not np.isfinite(c['temporal_decay']) or c['temporal_decay']<0:raise ValueError('Invalid temporal_decay')
    if c['camera_clock'] not in {'receive','header'}:raise ValueError('Unknown camera clock')
    for field in ('virtual_look_down_deg', 'virtual_look_right_deg'):
        degrees = c.get(field, 0.)
        if not isinstance(degrees, (int, float)) or isinstance(degrees, bool) or not np.isfinite(degrees):
            raise ValueError(f'{field} must be a finite angle in degrees')
        if degrees and not c['align_camera_intrinsics']:
            raise ValueError(f'{field} requires align_camera_intrinsics')
    if not 0<=c['pose_preview_ms']<=200 or not 0<=c['pose_feedback_age_ms']<=100:
        raise ValueError('Invalid pose timing assumptions')
    for field in ('initial_o6_target','initial_o6_qpos'):
        values=np.asarray(c[field],dtype=float)
        if values.shape!=(6,) or not np.isfinite(values).all() or np.any(values<0) or np.any(values>255):
            raise ValueError('Invalid '+field)
    if not np.isfinite(c['initial_o6_tolerance']) or not 0<c['initial_o6_tolerance']<=20:
        raise ValueError('Invalid initial_o6_tolerance')
    start_fields=('initial_left_q_rad','initial_left_position_m','initial_left_quaternion_wxyz')
    present=[field in c for field in start_fields]
    if any(present) and not all(present):
        raise ValueError('Configured ACT start requires q, position, and quaternion together')
    if all(present):
        q=np.asarray(c['initial_left_q_rad'],dtype=float)
        p=np.asarray(c['initial_left_position_m'],dtype=float)
        u=np.asarray(c['initial_left_quaternion_wxyz'],dtype=float)
        if q.shape!=(7,) or p.shape!=(3,) or u.shape!=(4,) or not all(
                np.isfinite(values).all() for values in (q,p,u)):
            raise ValueError('Invalid configured ACT left-arm start')
        if not np.isclose(np.linalg.norm(u),1.,atol=1e-5):
            raise ValueError('Configured ACT start quaternion must be normalized WXYZ')
    from o6_replay_bridge import validate_o6_can
    validate_o6_can(c['o6_can'])
    return c


def act_start(c):
    """Return the configured ACT anchor, falling back to the original fixed start."""
    import numpy as np
    from replay_tron2_fixed_start import FIXED_LEFT_Q,FIXED_LEFT_POSITION,FIXED_LEFT_QUATERNION
    fields=('initial_left_q_rad','initial_left_position_m','initial_left_quaternion_wxyz')
    if all(field in c for field in fields):
        return (np.asarray(c[fields[0]],dtype=float).copy(),
                np.asarray(c[fields[1]],dtype=float).copy(),
                np.asarray(c[fields[2]],dtype=float).copy())
    return FIXED_LEFT_Q.copy(),FIXED_LEFT_POSITION.copy(),FIXED_LEFT_QUATERNION.copy()


def reference_point(c):
    from tron2_act.deploy_core import ReferencePoint
    if c['reference_point']=='sdk-proxy':return ReferencePoint.identity()
    if c['reference_point']!='mounted-tracker' or not c['mount_transform_file']:
        raise ValueError('mounted-tracker requires an explicit verified T_ET file')
    mount=json.loads((ROOT/c['mount_transform_file']).read_text())
    return ReferencePoint.from_matrix(mount['T_ET'])


def offline(c,args,out):
    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation
    from tron2_act.data import ACTDataset,IMAGE_KEY
    from tron2_act.live_policy import LivePolicy
    from tron2_act.deploy_core import SpatialChunk
    model=LivePolicy(c,ROOT)
    dataset=ACTDataset(args.prepared,'test',normalized=False,images=False)
    chosen=[i for i,(path,row) in enumerate(dataset.items) if path.endswith(f'{args.offline_episode:06d}.hdf5')][:args.frames]
    if not chosen:raise ValueError('Offline episode is not in the held-out split')
    _,start_position,start_quaternion=act_start(c)
    r0=Rotation.from_quat(start_quaternion[[1,2,3,0]]).as_matrix()
    predictions=[];latency=[]
    from replay_tron2_tracker import TraceRecorder
    trace=TraceRecorder(out/'offline.jsonl')
    try:
        for index in chosen:
            path,row=dataset.items[index];sample=dataset[index]
            with h5py.File(path,'r') as f:rgb=f[IMAGE_KEY][row]
            action,metrics,foreground=model.predict(rgb,sample['state'].numpy(),already_aligned=True)
            stamp=round(dataset.episode(path).image_times[row]/model.config.time_scale*1e9)
            chunk=SpatialChunk(stamp,start_position,r0,action,reference_point(c),
                               execution_speed=c['execution_speed'])
            p,r,hand=chunk.target(stamp+chunk.step)
            trace.record('offline_prediction',episode=args.offline_episode,row=row,action=action.tolist(),
                         first_sdk_position=p.tolist(),first_o6_action=hand.tolist(),**metrics)
            predictions.append(action);latency.append(metrics['inference_ms'])
        np.savez_compressed(out/'predictions.npz',action=predictions)
        report={'mode':'offline','hardware_connections':0,'hardware_motion_commands':0,'frames':len(predictions),
                'inference_p95_ms':float(np.percentile(latency,95)),
                'note':'Recorded measured observations; fixed-start decoding fixture; no physics simulation or hardware execution'}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report
    finally:trace.close()


def live(c,args,out):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from replay_tron2_tracker import TraceRecorder,Tron2JsonClient,check_anchor_stable
    from tools.tron2_raw_ws_client import RawWSClient
    from tools.diagnose_tron2_z_compensation import CompensationTelemetry
    from replay_tron2_fixed_start import acquire_replay_lock
    from o6_replay_bridge import O6ReplayBridge
    from tron2_act.live_runtime import ROSInputs,Observations,make_pose_session
    from tron2_act.live_policy import policy_worker,vision_monitor
    from tron2_act.deploy_core import ChunkPool,LiveReference,PendingObservation
    trace=TraceRecorder(out/'trace.jsonl');client=telemetry=io=bridge=session=worker=monitor=lock=None
    error=None;interrupted=False;cleanup_sigint=None
    queries=sent_hands=missed_ticks=0;rejected_frames=hold_events=0
    last_sequence=None;last_query_ns=0;last_result_ns=None;pending_camera=None;policy_holding=False
    freshness_ns=round(c['io_freshness_ms']*1e6)
    reference=reference_point(c);pool=ChunkPool(reference,c['temporal_decay'],c['execution_speed'],c['io_freshness_ms'])
    context=mp.get_context('spawn');requests=context.Queue(maxsize=1);responses=context.Queue(maxsize=4)
    vision_frames=context.Queue(maxsize=2) if args.monitor_vision else None
    try:
        if args.execute:lock=acquire_replay_lock()
        if vision_frames is not None:
            monitor=context.Process(target=vision_monitor,args=(vision_frames,str(out/'vision_record'),True),daemon=True);monitor.start()
        worker=context.Process(target=policy_worker,args=(requests,responses,c,str(ROOT),str(out/'policy_previews'),vision_frames),daemon=True);worker.start()
        ready=responses.get(timeout=60.)
        if ready['event']!='ready':raise RuntimeError(ready.get('error','Policy did not initialize'))
        io=ROSInputs(c,trace)
        if not args.existing_o6_bridge:
            bridge=O6ReplayBridge(c['o6_can'],out/'o6_bridge.log',trace);bridge.start(io)
        io.start()
        if not io.ready.wait(10.):raise RuntimeError(io.error or 'ROS/O6 identity startup timed out')
        client=Tron2JsonClient(c['robot_ip'],trace) if args.execute else RawWSClient(c['robot_ip'],trace)
        client.wait_for_identity()
        if client.accid!=c['robot_id']:raise RuntimeError('Unexpected robot identity: '+client.accid)
        start_q,start_position,start_quaternion=act_start(c)
        q,dq=client.joint_snapshot('act_start');anchor=client.current_servo_pose('act_start')
        if np.max(abs(dq))>.05:raise RuntimeError('Robot is not stationary')
        if (np.max(abs(q[:7]-start_q))>.04 or np.linalg.norm(anchor.left_position-start_position)>.02 or
            np.rad2deg(Rotation.from_quat(anchor.left_quaternion[[1,2,3,0]]).inv().__mul__(Rotation.from_quat(start_quaternion[[1,2,3,0]])).magnitude())>3):
            raise RuntimeError('Left arm is not at the ACT start configured by --config; run the matching start preparation first')
        trajectory=LiveReference(anchor.left_position,Rotation.from_quat(anchor.left_quaternion[[1,2,3,0]]).as_matrix(),pool,
                                 c['max_displacement_m'],c['max_angle_deg'],c['startup_ramp_s'])
        observations=Observations(reference,c['camera_clock'],c['pose_feedback_age_ms'],c['io_freshness_ms'])
        if args.execute:
            session=make_pose_session(client,anchor,q,trajectory,c,trace);session.start();telemetry=session.telemetry
        else:
            telemetry=CompensationTelemetry(client,trace);telemetry.thread.start()
            deadline=time.monotonic()+5.
            while telemetry.packet is None or telemetry.latest_joint is None:
                if telemetry.error or time.monotonic()>deadline:raise RuntimeError(telemetry.error or 'Initial feedback timed out')
                time.sleep(.005)
        def pump():
            telemetry.require_fresh();observations.pump(telemetry,io)
            if io.error:raise RuntimeError(io.error)
        def require_live_inputs():
            pump()
            current=time.monotonic_ns()
            io.healthy(current)
            if observations.last_robot_ns is None or current-observations.last_robot_ns>freshness_ns:
                raise RuntimeError('Measured robot feedback expired')
            if observations.last_hand_ns is None or current-observations.last_hand_ns>freshness_ns:
                raise RuntimeError('Measured O6 feedback expired')
        def require_robot_inputs():
            # ServoP takeover/settling needs fresh robot feedback, but it does
            # not consume camera or O6 samples.  Those streams are revalidated
            # by prime() immediately before the ACT clock starts.
            pump();current=time.monotonic_ns();io.healthy(current,require_camera=False)
            if observations.last_robot_ns is None or current-observations.last_robot_ns>freshness_ns:
                raise RuntimeError('Measured robot feedback expired')
        initial_deadline=time.monotonic()+5.;initial_hand=None;initial_error=None
        seen_initial_stamp=None;initial_stable=0;initial_failure=None
        while time.monotonic()<initial_deadline and initial_stable<3:
            pump()
            if observations.last_hand_ns is not None and observations.last_hand_ns!=seen_initial_stamp:
                seen_initial_stamp=observations.last_hand_ns
                initial_hand=np.asarray(observations.hands.rows[-1][1],dtype=float)
                try:
                    initial_error=require_initial_o6_state(c,initial_hand)
                    initial_failure=None;initial_stable+=1
                except RuntimeError as exc:
                    initial_failure=exc;initial_stable=0
            time.sleep(.01)
        if initial_stable<3:
            if initial_failure is not None:raise initial_failure
            raise RuntimeError('No stable measured O6 initial state')
        trace.record('act_initial_o6_verified',q_pos=initial_hand.tolist(),
                     expected=c['initial_o6_qpos'],maximum_error=float(np.max(initial_error)))
        # The existing ServoP takeover/settle path is used without MoveJ, gain,
        # payload writes or CAN ownership in this process.
        def prime(holding=False):
            nonlocal last_sequence,last_result_ns,pending_camera
            cutoff=time.monotonic_ns();deadline=time.monotonic()+10.
            pending_camera=None
            awaiting=False
            while True:
                pump()
                camera=pending_camera or io.camera
                if not awaiting and camera and camera['sequence']!=last_sequence:
                    try:
                        pending_camera=camera
                        packet=observations.packet(camera,time.monotonic_ns(),startup_cutoff=cutoff)
                        if packet is None:
                            last_sequence=camera['sequence']
                            pending_camera=None
                        else:
                            requests.put_nowait(packet);last_sequence=camera['sequence'];pending_camera=None
                            awaiting=True
                    except queue.Full:
                        pass
                    except PendingObservation as exc:
                        # Keep this exact image pending: the next measured robot
                        # sample may provide the right interpolation endpoint.
                        trace.record('act_startup_observation_wait',reason=str(exc),sequence=camera['sequence'])
                    except RuntimeError as exc:
                        trace.record('act_startup_observation_wait',reason=str(exc),sequence=camera['sequence'])
                        last_sequence=camera['sequence'];pending_camera=None
                try:
                    result=responses.get_nowait()
                    awaiting=False
                    if result['event']=='error':raise RuntimeError(result['error'])
                    if result['event']=='observation_rejected':
                        trace.record('act_startup_observation_rejected',**result)
                        result=None
                    completed=time.monotonic_ns()
                    if result is not None and result['stamp_ns']>=cutoff:
                        if completed-result['stamp_ns']>freshness_ns:
                            trace.record('act_startup_prediction_discarded',age_ms=(completed-result['stamp_ns'])/1e6)
                        else:
                            pool.add(result['stamp_ns'],result['p'],result['r'],result['action'],completed)
                            last_result_ns=completed
                            trace.record('act_initial_prediction',observation_ns=result['stamp_ns'],
                                         mask_pixels=result['mask_pixels'],mask_centroids=result['mask_centroids'],
                                         mask_bboxes=result['mask_bboxes'],motion_started=holding)
                            return
                except queue.Empty:pass
                if time.monotonic()>deadline:raise RuntimeError('No aligned live prediction within startup deadline')
                if holding:session.stream('act_priming')
                time.sleep(1/300)
        # Verify actual camera/foreground/proprioception before any ServoP target.
        prime(holding=False)
        if session:
            countdown=time.monotonic()+5.
            print('Camera/model/O6 feedback ready; ServoP takeover starts in 5 seconds.',flush=True)
            while time.monotonic()<countdown:require_robot_inputs();time.sleep(.01)
            require_robot_inputs()
            latest_pose,joints=telemetry.require_fresh()
            check_anchor_stable(anchor,latest_pose)
            live_q=np.asarray(joints['q']);live_dq=np.asarray(joints['dq'])
            if (live_q.shape!=(16,) or live_dq.shape!=(16,) or not np.isfinite(live_q).all()
                    or not np.isfinite(live_dq).all() or np.max(abs(live_q-q))>.01 or np.max(abs(live_dq))>.05):
                raise RuntimeError('Robot moved during ACT countdown')
            session.settle_pose(health_check=require_robot_inputs)
            io.send({'command':'acquire'})
            deadline=time.monotonic()+5.
            while not io.acquired.is_set():
                require_robot_inputs();session.stream('act_acquiring_hand')
                if time.monotonic()>deadline:raise RuntimeError('O6 ownership timed out')
                time.sleep(1/300)
            pool.chunks.clear();pool.last_observation_ns=None
            prime(holding=True)
        epoch=time.monotonic_ns();trajectory.epoch_ns=epoch
        if session:session.start_ns=epoch
        index=0;last_hand_due=epoch;first_hand_sent=None;last_hand_values=None
        while True:
            due=epoch+round(index/300*1e9);now=time.monotonic_ns()
            if now<due:time.sleep((due-now)/1e9)
            now=time.monotonic_ns()
            if c['max_duration_s'] is not None and now-epoch>=c['max_duration_s']*1e9:break
            if now-due>100_000_000:raise RuntimeError('ServoP scheduler overrun; no command catch-up')
            pump();io.healthy(now,need_applied=first_hand_sent is not None and now-first_hand_sent>250_000_000,
                              require_camera=False)
            if observations.last_robot_ns is None or now-observations.last_robot_ns>freshness_ns:
                raise RuntimeError('Measured robot feedback expired')
            if observations.last_hand_ns is None or now-observations.last_hand_ns>freshness_ns:
                raise RuntimeError('Measured O6 feedback expired')
            camera=pending_camera or io.camera
            if io.camera_fresh(now) and camera['sequence']!=last_sequence and now-last_query_ns>=50_000_000:
                try:
                    pending_camera=camera
                    packet=observations.packet(camera,time.monotonic_ns());requests.put_nowait(packet)
                    last_sequence=camera['sequence'];last_query_ns=now;pending_camera=None
                except queue.Full:
                    pass
                except PendingObservation as exc:
                    # Do not consume a camera frame merely because feedback has
                    # not bracketed its timestamp yet; retry after the next pump.
                    trace.record('act_observation_alignment_wait',reason=str(exc),sequence=camera['sequence'])
                except RuntimeError as exc:
                    rejected_frames+=1
                    trace.record('act_observation_alignment_rejected',reason=str(exc),sequence=camera['sequence'])
                    last_sequence=camera['sequence'];pending_camera=None
            try:
                result=responses.get_nowait()
                if result['event']=='error':raise RuntimeError(result['error'])
                if result['event']=='observation_rejected':
                    rejected_frames+=1
                    trace.record('act_observation_rejected',**result)
                    result=None
                if result is not None:
                    try:
                        pool.add(result['stamp_ns'],result['p'],result['r'],result['action'],now)
                    except RuntimeError as exc:
                        # A delayed/out-of-order inference result is equivalent
                        # to a missed observation.  Drop it and keep the last
                        # action instead of aborting the rollout.
                        rejected_frames+=1
                        trace.record('act_prediction_dropped',reason=str(exc),
                                     observation_ns=result['stamp_ns'],sequence=result['sequence'])
                        result=None
                if result is None:
                    raise queue.Empty
                last_result_ns=now;queries+=1
                if policy_holding:
                    policy_holding=False
                    trace.record('act_policy_resumed',observation_ns=result['stamp_ns'])
                trace.record('act_raw_prediction',observation_ns=result['stamp_ns'],completed_ns=result['completed_ns'],
                             sequence=result['sequence'],anchor_p=np.asarray(result['p']).tolist(),anchor_r=np.asarray(result['r']).tolist(),
                             state=np.asarray(result['state']).tolist(),action=result['action'].tolist(),
                             input_age_ms=(now-result['stamp_ns'])/1e6,inference_ms=result['inference_ms'],
                             mask_pixels=result['mask_pixels'],mask_centroids=result['mask_centroids'],
                             mask_bboxes=result['mask_bboxes'])
            except queue.Empty:pass
            if last_result_ns is None or now-last_result_ns>freshness_ns:
                if not policy_holding:
                    policy_holding=True;hold_events+=1
                    trace.record('act_policy_hold_started',reason='camera/prediction unavailable',
                                 last_prediction_age_ms=None if last_result_ns is None else (now-last_result_ns)/1e6)
            if session:session.stream('act_control')
            elif index%15==0:
                p,r=trajectory.target((now-epoch)/1e9)
                trace.record('act_shadow_target',sdk_position=p.tolist(),sdk_rotation=r.tolist(),motion_commands=0)
            if now-last_hand_due>=50_000_000:
                try:
                    _,_,hand=pool.target(now,hand=True)
                except PendingObservation:
                    if last_hand_values is None:
                        trace.record('act_hand_waiting_for_first_due_target',due_ns=now)
                        hand=None
                    else:
                        hand=np.asarray(last_hand_values,dtype=float)
                        trace.record('act_hand_holding_last_target',due_ns=now,values=last_hand_values)
                if hand is not None:
                    command={'command':'target','sequence':sent_hands,'due_ns':now,'values':np.rint(hand).tolist()}
                    last_hand_values=command['values']
                    if args.execute:
                        io.send(command)
                        if first_hand_sent is None:first_hand_sent=now
                    trace.record('act_hand_scheduled',execute=args.execute,**command)
                    sent_hands+=1;last_hand_due=now
            next_index=max(index+1,int((time.monotonic_ns()-epoch)*300//1_000_000_000)+1)
            if next_index>index+1:
                missed_ticks+=next_index-index-1
                trace.record('act_scheduler_skipped',periods=next_index-index-1)
            index=next_index
        if session:
            trajectory.frozen=trajectory.target((time.monotonic_ns()-epoch)/1e9)
            until=time.monotonic()+1.
            while time.monotonic()<until:
                pump();session.stream('act_final_hold')
                now=time.monotonic_ns()
                if last_hand_values is not None and now-last_hand_due>=50_000_000:
                    io.send({'command':'target','sequence':sent_hands,'due_ns':now,'values':last_hand_values})
                    sent_hands+=1;last_hand_due=now
                time.sleep(1/300)
    except KeyboardInterrupt:
        interrupted=True
        # The first Ctrl+C requests an orderly stop.  Ignore additional SIGINT
        # during teardown so the O6 bridge, ROS sidecar, queues and report are
        # not left behind half-closed.
        cleanup_sigint=signal.signal(signal.SIGINT,signal.SIG_IGN)
        trace.record('act_user_stop',message='KeyboardInterrupt')
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';trace.record('act_error',error=error,traceback=traceback.format_exc())
    finally:
        if io:
            try:io.close()
            except BaseException as exc:error=error or ('ROS cleanup: '+str(exc))
        if session:
            try:session.close()
            except BaseException as exc:error=error or ('ServoP cleanup: '+str(exc))
        elif telemetry:
            telemetry.stop.set();telemetry.thread.join(timeout=2.)
        if client:
            try:
                if hasattr(client,'close'):client.close()
            except BaseException as exc:trace.record('act_cleanup_error',error=str(exc))
        if bridge:
            try:bridge.close()
            except BaseException as exc:error=error or ('Bridge cleanup: '+str(exc))
        if worker:
            try:requests.put_nowait(None)
            except queue.Full:pass
            worker.join(timeout=3.)
            if worker.is_alive():worker.terminate();worker.join(timeout=2.)
        if monitor:
            try:vision_frames.put(None,timeout=1.)
            except queue.Full:pass
            monitor.join(timeout=3.)
            if monitor.is_alive():monitor.terminate();monitor.join(timeout=2.)
        for channel in [requests,responses]:channel.cancel_join_thread();channel.close()
        if vision_frames is not None:vision_frames.cancel_join_thread();vision_frames.close()
        if lock:lock.close()
        try:trace.close()
        except BaseException as exc:error=error or ('Trace cleanup: '+str(exc))
    report={'mode':'execute' if args.execute else 'shadow','completed':error is None,'error':error,
            'policy_predictions':queries,'hand_schedule_events':sent_hands,'skipped_servo_periods':missed_ticks,
            'hardware_motion_enabled':bool(args.execute),'reference_point':c['reference_point'],
            'execution_speed':c['execution_speed'],'max_duration_s':c['max_duration_s'],
            'io_freshness_ms':c['io_freshness_ms'],
            'rejected_observations':rejected_frames,'policy_hold_events':hold_events,
            'stopped_by_user':interrupted}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if cleanup_sigint is not None:signal.signal(signal.SIGINT,cleanup_sigint)
    if error:raise RuntimeError(error)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/tron2_act_deploy.json')
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--shadow',action='store_true');modes.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--existing-o6-bridge',action='store_true')
    parser.add_argument('--monitor-vision',action='store_true',help='实时显示并录制对齐 Top 与 ACT 黑底输入')
    parser.add_argument('--o6-can');parser.add_argument('--max-duration',type=float,
        help='任务段自动停止秒数；0表示一直运行到Ctrl+C')
    parser.add_argument('--execution-speed',type=float,
        help='ACT动作重定时倍率，0.1..1.0；1.0为训练部署速度')
    parser.add_argument('--offline-episode',type=int,default=74);parser.add_argument('--frames',type=int,default=40)
    parser.add_argument('--prepared',default=str(ROOT/'artifacts/cup_task_audit_20261117/prepared_24'))
    args=parser.parse_args(argv);c=load_runtime(args.config)
    if args.shadow and not args.existing_o6_bridge:parser.error('--shadow requires --existing-o6-bridge; no hand initialization in read-only mode')
    if args.o6_can:
        from o6_replay_bridge import validate_o6_can
        c['o6_can']=validate_o6_can(args.o6_can)
    if args.max_duration is not None:
        if not math.isfinite(args.max_duration) or args.max_duration<0:
            parser.error('--max-duration must be finite and nonnegative')
        c['max_duration_s']=None if args.max_duration==0 else args.max_duration
    if args.execution_speed is not None:
        if not math.isfinite(args.execution_speed) or not .1<=args.execution_speed<=1.:
            parser.error('--execution-speed must be within 0.1..1.0')
        c['execution_speed']=args.execution_speed
    if args.frames<1:parser.error('--frames must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'runtime_config.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
    result=live(c,args,args.output) if args.shadow or args.execute else offline(c,args,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':
    code=0
    try:code=main()
    except (RuntimeError,ValueError,OSError,TimeoutError) as exc:
        print(str(exc),file=sys.stderr);code=2
    if '--execute' in sys.argv:
        sys.stdout.flush();sys.stderr.flush();os._exit(code)
    raise SystemExit(code)
