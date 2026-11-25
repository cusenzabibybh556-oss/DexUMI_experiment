#!/usr/bin/env python3
"""Check the real spawned GPU worker with held-out images; no robot/ROS/CAN I/O."""
import argparse
import base64
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import cv2
    import h5py
    import numpy as np
    from tools.deploy_tron2_act import load_runtime
    from tools.preview_tron2_top_camera import read_reference
    from tron2_act.data import ACTDataset,IMAGE_KEY
    from tron2_act.live_policy import policy_worker
    config=load_runtime(ROOT/'configs/tron2_act_deploy.json')
    dataset=ACTDataset(ROOT/'artifacts/cup_task_audit_20261117/prepared_24','test',normalized=False,images=False)
    camera,_=read_reference(config['reference_hdf5'],0)
    context=mp.get_context('spawn');requests=context.Queue(1);responses=context.Queue(4)
    worker=context.Process(target=policy_worker,args=(requests,responses,config,str(ROOT),str(args.output/'previews')))
    records=[];worker.start()
    try:
        ready=responses.get(timeout=60.)
        if ready['event']!='ready':raise RuntimeError(ready)
        for sequence,index in enumerate([0,40,80,120,160]):
            path,row=dataset.items[index];sample=dataset[index]
            with h5py.File(path) as f:rgb=f[IMAGE_KEY][row]
            ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            if not ok:raise RuntimeError('JPEG fixture encoding failed')
            start=time.monotonic_ns()
            requests.put({'sequence':sequence,'stamp_ns':start,'jpeg':base64.b64encode(jpeg).decode('ascii'),
                          'info':asdict(camera),'state':sample['state'].numpy(),'p':np.zeros(3),'r':np.eye(3)})
            result=responses.get(timeout=10.)
            if result['event']!='prediction':raise RuntimeError(result)
            if result['sequence']!=sequence or result['stamp_ns']!=start:raise RuntimeError('Worker lost observation identity')
            action=np.asarray(result['action'])
            if action.shape!=(50,15) or not np.isfinite(action).all():raise RuntimeError('Invalid worker output')
            records.append({'sequence':sequence,'inference_ms':result['inference_ms'],
                            'round_trip_ms':(time.monotonic_ns()-start)/1e6,'mask_pixels':result['mask_pixels']})
        report={'hardware_connections':0,'hardware_motion_commands':0,'step':ready['step'],
                'frames':len(records),'records':records,'note':'Includes JPEG decode, reference-intrinsics mapping, foreground, ACT and IPC; recorded images only.'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        try:requests.put(None,timeout=1.)
        except Exception:pass
        worker.join(timeout=5.)
        if worker.is_alive():worker.terminate();worker.join(timeout=2.)
        for channel in [requests,responses]:channel.cancel_join_thread();channel.close()


if __name__=='__main__':main()
