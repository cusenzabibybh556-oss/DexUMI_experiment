#!/usr/bin/env python3
"""Move the left O6 to the configured ACT start shape and verify fresh feedback."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/tron2_act_deploy.json')
    parser.add_argument('--can',default='can0')
    parser.add_argument('--timeout',type=float,default=15.)
    parser.add_argument('--speed',type=int,default=255)
    parser.add_argument('--torque',type=int,default=40)
    args=parser.parse_args(argv)
    if not 2<=args.timeout<=30 or not 10<=args.speed<=255 or not 1<=args.torque<=255:
        parser.error('timeout/speed/torque outside safe CLI range')
    from o6_replay_bridge import existing_o6_bridges,validate_o6_can
    interface=validate_o6_can(args.can)
    existing=existing_o6_bridges()
    if existing:
        raise RuntimeError('O6 bridge already running: '+','.join(map(str,existing)))
    config=json.loads(args.config.read_text())
    target=np.asarray(config['initial_o6_target'],dtype=float)
    expected=np.asarray(config['initial_o6_qpos'],dtype=float)
    tolerance=float(config['initial_o6_tolerance'])
    if target.shape!=(6,) or expected.shape!=(6,):raise ValueError('Invalid O6 start vectors')
    sdk=ROOT/'TransHandR0725/TransHandR'
    sys.path.insert(0,str(sdk))
    from LinkerHand.linker_hand_api import LinkerHandApi
    hand=LinkerHandApi(hand_joint='O6',hand_type='left',can=interface)
    try:
        hand.set_speed([args.speed]*6);hand.set_torque([args.torque]*6)
        deadline=time.monotonic()+args.timeout;stable=0;last=None;next_report=0.
        while time.monotonic()<deadline:
            hand.finger_move(target.tolist())
            # get_state() sends a request and returns the previous asynchronous reply.
            hand.get_state();time.sleep(.04)
            last=np.asarray(hand.get_state(),dtype=float);time.sleep(.01)
            error=float(np.max(np.abs(last-expected))) if last.shape==(6,) else float('inf')
            stable=stable+1 if error<=tolerance else 0
            if time.monotonic()>=next_report:
                print(f'O6 q_pos={np.rint(last).astype(int).tolist()} max_error={error:.1f}',flush=True)
                next_report=time.monotonic()+1.
            if stable>=5:
                print('O6 ACT initial state verified:',np.rint(last).astype(int).tolist(),flush=True)
                return 0
        raise RuntimeError(
            f'O6 did not reach ACT initial state in {args.timeout:g}s; '
            f'last={None if last is None else np.rint(last).astype(int).tolist()}, '
            f'expected={np.rint(expected).astype(int).tolist()}'
        )
    finally:
        device=getattr(hand,'hand',None)
        if device is not None:
            device.running=False
            thread=getattr(device,'receive_thread',None)
            if thread is not None:thread.join(timeout=1.)
            bus=getattr(device,'bus',None)
            if bus is not None:
                try:bus.shutdown()
                except Exception:pass


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,RuntimeError,ValueError) as exc:
        print(f'error: {exc}',file=sys.stderr);raise SystemExit(2)
