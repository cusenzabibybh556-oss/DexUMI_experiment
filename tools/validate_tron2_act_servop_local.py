#!/usr/bin/env python3
"""Validate one saved ACT chunk through the loopback vendor ServoP runtime.

This is a controller integration test using recorded-image predictions, not a
visual closed-loop or O6/contact task test. The address and simulation SN are
hard coded; there is no real-robot endpoint option.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(output):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from tools.tron2_raw_ws_client import RawWSClient
    from replay_tron2_tracker import servop_data
    from tron2_act.simulation import ActionChunk, SimServoP
    output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    control = feedback = None
    stop = threading.Event()
    thread = None
    records = []
    sent = 0
    error = None
    shared = {}
    source = ROOT/'artifacts/tron2_act/cup_stack_baseline_20261117_01/test/predictions.npz'
    try:
        control = RawWSClient('127.0.0.1')
        control.wait_for_identity()
        if control.accid != 'DACH_TRON2A_001':
            raise RuntimeError('Expected local simulation identity DACH_TRON2A_001')
        q,dq = control.joint_snapshot('local_sim_initial')
        engine = SimServoP(q)
        anchor = control.current_servo_pose()
        p0 = np.asarray(anchor.left_position)
        r0 = Rotation.from_quat(np.asarray(anchor.left_quaternion)[[1,2,3,0]]).as_matrix()
        engine.engine.check_anchor(q,p0,r0)
        right = np.r_[anchor.right_position,anchor.right_quaternion]
        action = np.load(source)['predicted'][0]
        np.save(output/'action_chunk.npy',action)
        chunk = ActionChunk(10.,p0,r0,np.full(6,255.),action)
        def target(t):
            if t < 10:
                return p0.copy(),r0.copy(),np.full(6,255.)
            return chunk.target(t)
        feedback = RawWSClient('127.0.0.1')
        feedback.wait_for_identity()
        def read_feedback():
            try:
                while not stop.is_set():
                    before = time.monotonic()
                    pose = feedback.current_servo_pose('local_sim_feedback')
                    after = time.monotonic()
                    shared['sample'] = ((before+after)/2,pose,after-before)
                    stop.wait(max(0.,.05-(after-before)))
            except BaseException as exc:
                shared['error'] = str(exc)
        thread = threading.Thread(target=read_feedback,daemon=True);thread.start()
        start = time.monotonic()
        last_sample = None
        for step in range(round(13.5*300)):
            due = start+step/300
            remaining = due-time.monotonic()
            if remaining>0:time.sleep(remaining)
            t = time.monotonic()-start
            if t-step/300>.1:
                raise RuntimeError('Sender fell behind by more than 100ms')
            if 'error' in shared:raise RuntimeError(shared['error'])
            sample = shared.get('sample')
            if sample is not None and sample[0] != last_sample:
                stamp,pose,rtt = sample
                last_sample = stamp
                if time.monotonic()-stamp>.15 or rtt>.1:
                    raise RuntimeError('Stale feedback or excessive RTT')
                measured_t = max(0.,stamp-start)
                actual_p = np.asarray(pose.left_position)
                actual_r = Rotation.from_quat(np.asarray(pose.left_quaternion)[[1,2,3,0]]).as_matrix()
                nominal_p,nominal_r,_ = target(measured_t)
                t = time.monotonic()-start
                engine.observe(measured_t,nominal_p,nominal_r,actual_p,actual_r,now=t)
                ep = float(np.linalg.norm(actual_p-nominal_p)*1000)
                er = float(np.rad2deg(Rotation.from_matrix(actual_r@nominal_r.T).magnitude()))
                records.append({'t':measured_t,'position_error_mm':ep,'rotation_error_deg':er,
                    'actual_p':actual_p.tolist(),'nominal_p':nominal_p.tolist(),'rtt_ms':rtt*1000})
            ramp = min(1.,t/5);ramp = ramp*ramp*(3-2*ramp)
            _,diagnostic = engine.step(t,target,ramp)
            issued_r = engine.engine.forward(engine.engine.qissued)[1]
            issued_p = engine.engine.forward(engine.engine.qissued)[0]
            left = np.r_[issued_p,Rotation.from_matrix(issued_r).as_quat()[[3,0,1,2]]]
            failure = control.failure()
            if failure:raise RuntimeError('ServoP notification: '+failure)
            control.send('request_servop',servop_data(left,right))
            sent += 1
    except BaseException as exc:
        error = str(exc)
        save('error.json',{'error':error,'traceback':traceback.format_exc()})
    finally:
        stop.set()
        if thread:thread.join(timeout=5)
        if feedback:feedback.close()
        if control:control.close()
    (output/'trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    active = [x for x in records if 10<=x['t']<=12.5]
    def metrics(rows):
        if not rows:return None
        e = np.array([x['position_error_mm'] for x in rows])
        return {'samples':len(rows),'position_rmse_mm':float(np.sqrt(np.mean(e*e))),
            'position_p95_mm':float(np.percentile(e,95)),
            'rotation_p95_deg':float(np.percentile([x['rotation_error_deg'] for x in rows],95))}
    report = {'completed':error is None,'stop_reason':error,'endpoint':'127.0.0.1:5000',
        'expected_sn':'DACH_TRON2A_001','hardware_connections':0,'vendor_servop_runtime':True,
        'servop_commands_sent':sent,'controller':'existing pose-reference + PoseCorrection + bounded streaming reference',
        'source':str(source),'prediction_index':0,'visual_closed_loop':False,'o6_actuated':False,
        'active_chunk':metrics(active),'all_feedback':metrics(records),
        'limitations':['Recorded-image ACT chunk; no visual policy replanning in this interface test.',
                       'Stock vendor motor simulator has no articulated O6; O6 targets are saved only.']}
    save('report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
