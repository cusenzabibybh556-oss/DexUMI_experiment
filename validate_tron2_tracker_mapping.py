#!/usr/bin/env python3
"""小幅肩肘移动验证已求出的映射；固定矩阵，不用这些样本重新拟合。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
from auto_calibrate_tron2_tracker import current_joints, require_tracker, execute_stages
from capture_tron2_tracker_calibration import HardwareSource, CaptureSession, acquire_capture_lock
from dexumi_wrist_solver import tracker_pose_for_calibration
from solve_tron2_tracker_calibration import load_samples, PoseGroup, group_errors, statistics
from tools.tron2_head_pose import HeadClient
from tron2_left_ik import LeftArmIK


def plan(model, start):
    targets=np.tile(start,(8,1))
    for i,(joint,angle) in enumerate([(0,2),(0,-2),(3,2),(3,-2),(1,2),(1,-2)],1):
        targets[i,joint]+=np.deg2rad(angle)
    for g in range(model.model.ngeom):
        if model.model.body(int(model.model.geom_bodyid[g])).name.startswith('grasper_'):
            model.model.geom_contype[g]=model.model.geom_conaffinity[g]=0
    previous=start.copy()
    stages=[]
    for target in targets:
        model.check_joints(target)
        if np.any(abs(target[[5,6,12,13]])>.7505):
            raise ValueError('腕关节超出 MoveJ 文档范围')
        for a in np.linspace(0,1,61):
            model.data.qpos[model.qadr]=previous*(1-a)+target*a
            model.mj.mj_forward(model.model,model.data)
            for contact in model.data.contact:
                names=[model.model.body(int(model.model.geom_bodyid[g])).name for g in (contact.geom1,contact.geom2)]
                if contact.dist<0 and any(n.endswith('_L_Link') for n in names):
                    raise ValueError('模型内自碰撞：'+str(names))
        stages.append([] if np.max(abs(target-previous))<1e-9 else [target])
        previous=target
    poses=np.array([model.pose(q)[0] for q in targets])
    span=float(np.linalg.norm(poses-poses[0],axis=1).max())
    if span>.03:raise ValueError('验证目标末端偏移超过3cm')
    return targets,stages,span


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--passive-hid',required=True)
    p.add_argument('--tracker-mac',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    calibration=json.loads(args.calibration.read_text())
    if calibration['status']!='passed':raise ValueError('需要已通过几何检查的标定候选')
    run=Path(__file__).resolve().parent/'tracker_calibrations'/('validation_'+datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
    run.mkdir()
    log_file=(run/'motion.jsonl').open('x')
    def log(event,**fields):
        log_file.write(json.dumps({'event':event,'time':time.time(),**fields})+'\n');log_file.flush()
    session=client=handle=None
    try:
        handle=acquire_capture_lock()
        client=HeadClient(calibration['session']['robot_ip'])
        if client.accid!=calibration['session']['robot_id']:raise ValueError('机器人身份变化')
        start,dq=current_joints(client)
        if np.max(abs(dq))>.05:raise ValueError('机器人未停稳')
        targets,stages,span=plan(LeftArmIK(),start)
        (run/'plan.json').write_text(json.dumps({'start':start.tolist(),'targets':targets.tolist(),
            'maximum_tcp_displacement_m':span,'calibration':str(args.calibration.resolve()),
            'calibration_sha256':hashlib.sha256(args.calibration.read_bytes()).hexdigest()},indent=2))
        print(f'独立验证：8组，肩/肘单轴±2°，TCP最大偏移{span*100:.2f}cm；{run}',flush=True)
        if not args.execute:return 0
        state=client.request('request_drag_teach_manage',{'cmd':'status','arg':''},'response_drag_teach_manage')
        if 'state=ST_DEVED;' not in state['data'].get('message',''):raise ValueError('需要High-Level Dev')
        source=HardwareSource(calibration['session']['robot_ip'],0,passive_hid=args.passive_hid,tracker_mac=args.tracker_mac)
        session=CaptureSession(source,run,lambda _:None,{'tracker_index':0,'robot_ip':calibration['session']['robot_ip'],
            'mock':False,'independent_validation_of':str(args.calibration.resolve()),'map_label':'同一连接/地图',
            'mount_label':'未拆装Tracker；肩肘独立验证'})
        session.start();time.sleep(1);require_tracker(session,{18})
        for n in range(5,0,-1):print(f'{n}秒后开始独立验证',flush=True);time.sleep(1)
        fresh,dq=current_joints(client)
        if np.max(abs(fresh-start))>.005 or np.max(abs(dq))>.05:raise ValueError('准备期间姿态变化')
        state=client.request('request_drag_teach_manage',{'cmd':'status','arg':''},'response_drag_teach_manage')
        if 'state=ST_DEVED;' not in state['data'].get('message',''):raise ValueError('倒计时后模式改变')
        execute_stages(client,session,start,targets,stages,log,duration=3.,diagnostic_statuses={18},allowed_joints=(0,1,3))
        path=session.path;session.close();session=None
        meta,groups,rejected,digest=load_samples(path)
        X,Y=np.array(calibration['T_ET']),np.array(calibration['T_WB'])
        normalized=[PoseGroup(g.id,g.label,g.robot,tracker_pose_for_calibration(g.tracker,calibration['tracker_quaternion_convention']),g.statuses,g.pair_count) for g in groups]
        rows=[group_errors(g,X,Y) for g in normalized]
        metrics=statistics(rows)
        if metrics is None:raise ValueError('没有status=2的稳定验证组')
        passed=(len(groups)>=6 and metrics['wrist_position_error_mm']['rmse']<=5 and metrics['wrist_position_error_mm']['max']<=15
                and metrics['rotation_error_deg']['rmse']<=2 and metrics['rotation_error_deg']['max']<=6)
        result={'status':'passed' if passed else 'review_required','refitted':False,'calibration':str(args.calibration.resolve()),
                'samples':str(path),'input_sha256':digest,'metrics':metrics,'per_group':rows,'rejected':rejected}
        (run/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        return 0 if passed else 2
    except (Exception,KeyboardInterrupt) as e:
        log('aborted',error=str(e));print('停止后续动作：'+str(e),flush=True)
        return 1
    finally:
        if session:session.close()
        if client:client.socket.close()
        if handle:handle.close()
        log_file.close()


if __name__=='__main__':raise SystemExit(main())
