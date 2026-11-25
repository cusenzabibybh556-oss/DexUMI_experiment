#!/usr/bin/env python3
"""Bounded free-space 20 mm A/B test; no load/gain writes or finger targets.

Default is offline envelope planning from --snapshot. --execute enables one
300 Hz ServoP sender, existing telemetry/held-arm guards, and bounded Z feedback.
It does not repair firmware torque handover. Each invocation has its own anchor.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture_tron2_tracker_calibration import acquire_capture_lock
from replay_tron2_tracker import TraceRecorder, Tron2JsonClient
from tools.tron2_raw_ws_client import RawWSClient
from tools.tron2_motor_observer import MotorObserver
from tools.diagnose_tron2_servop import baseline, require_expected_payload, guard
from tools.diagnose_tron2_z_compensation import prepare_envelope
from tron2_z_replay import ZReplaySession

KEEP = []


def targets():
    ts = np.linspace(0, 20, 6001)
    dz = np.zeros(len(ts))
    up = ts < 6
    a = ts[up]/6
    dz[up] = .02*a*a*(3-2*a)
    dz[(ts >= 6) & (ts < 10)] = .02
    down = (ts >= 10) & (ts < 16)
    a = (ts[down]-10)/6
    dz[down] = .02*(1-a*a*(3-2*a))
    return ts, dz


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--mode', choices=('stationary', 'continuous'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.execute and args.snapshot is None:
        parser.error('Offline planning requires --snapshot')
    args.output.mkdir(parents=True, exist_ok=False)
    trace = client = reader = observer = session = lock = None
    report = dict(result='not_started', mode=args.mode, torque_continuity_repaired=False,
                  payload_writes=0, gain_writes=0, finger_targets=0, head_targets=0)
    try:
        if args.execute:
            lock = acquire_capture_lock()
            trace = TraceRecorder(args.output/'command_state.jsonl')
            reader = RawWSClient('10.192.1.2', trace)
            reader.wait_for_identity()
            if reader.accid != 'DACH_TRON2A_253':
                raise RuntimeError('Unexpected robot identity')
            snapshot = baseline(reader)
        else:
            snapshot = json.loads(args.snapshot.read_text())
        require_expected_payload(snapshot, [.396, 0, 0, 0, .001, 0, 0, 0])
        _, anchor, plan = prepare_envelope(snapshot, .02, .02)
        (args.output/'baseline.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n')
        (args.output/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
        ts, dz = targets()
        nominal = np.tile(anchor.left_position, (len(ts), 1))
        nominal[:, 2] += dz
        np.savez_compressed(args.output/'targets.npz', time_s=ts, nominal=nominal)
        report['plan'] = plan
        if not args.execute:
            report['result'] = 'planned_only'
            print(json.dumps(plan, ensure_ascii=False))
            return 0
        # SDK is the sole motion sender; websocket is used only for readback.
        client = Tron2JsonClient('10.192.1.2', trace)
        KEEP.append(client)
        client.wait_for_identity(timeout=12)
        if client.accid != reader.accid:
            raise RuntimeError('Motion/readback robot identities differ')
        observer = MotorObserver(trace, rate_hz=50)
        observer.start(client.robot)
        for remaining in range(5, 0, -1):
            print(f'{remaining}秒后验证{args.mode}：名义抬升20mm，修正上限20mm，修正限速1mm/s。', flush=True)
            time.sleep(1)
        fresh = baseline(reader)
        require_expected_payload(fresh, [.396, 0, 0, 0, .001, 0, 0, 0])
        _, fresh_anchor, _ = prepare_envelope(fresh, .02, .02)
        from tools.diagnose_tron2_servop import require_same_anchor
        require_same_anchor(anchor, fresh_anchor)
        reader.close()
        reader = None
        options = SimpleNamespace(robot_ip='10.192.1.2', feedback_timeout=.5,
                                  z_max_correction_mm=20., rate=300., interpolation='linear')
        session = ZReplaySession(client, anchor, options)
        session.start()
        session.settle_handover()
        print('接管Z恢复完成，开始20秒上下往返。', flush=True)
        start = time.monotonic_ns()+250_000_000
        session.keep_holding_until(start)
        trace.record('validation_trajectory_start', start_monotonic_ns=start,
                     mode=args.mode, correction_limit_m=.02, correction_rate_m_s=.001)
        stage = None
        for i, due in enumerate(ts):
            left = start+int(due*1e9)-time.monotonic_ns()
            if left > 0:
                time.sleep(left/1e9)
            if time.monotonic_ns()-(start+int(due*1e9)) > 100_000_000:
                raise RuntimeError('Trajectory scheduling exceeded 100 ms')
            phase = 'raise' if due < 6 else 'upper_hold' if due < 10 else 'return' if due < 16 else 'return_hold'
            if phase != stage:
                stage = phase
                print('阶段：'+phase, flush=True)
                trace.record('validation_stage', phase=phase, due_s=float(due))
            packet = session.telemetry.packet
            guard(anchor, np.r_[nominal[i], anchor.left_quaternion], packet.pose)
            adaptive = args.mode == 'continuous' or phase.endswith('hold')
            session.stream(nominal[i], anchor.left_quaternion, 'validation_'+phase, adaptive=adaptive)
        trace.record('validation_sending_stopped')
        time.sleep(.5)
        actual = session.final_pose()
        report.update(result='completed', final_z_error_mm=float((actual.left_position[2]-anchor.left_position[2])*1000),
                      retained_offset_mm=session.controller.applied_offset*1000,
                      nominal_anchor_m=anchor.left_position.tolist(), final_actual_m=actual.left_position.tolist())
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        report.update(result='aborted', error=str(exc) or 'KeyboardInterrupt')
        if trace:
            trace.record('validation_abort', message=report['error'])
        print('停止后续发送：'+report['error'], flush=True)
        return 2
    finally:
        if session:
            session.close()
        if observer:
            observer.stop()
            report['observer'] = observer.summary()
        if reader:
            reader.close()
        if trace:
            trace.close()
        if lock:
            lock.close()
        (args.output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid the SDK's known destructor issue after closing logs and readback.
    os._exit(code)
