"""按DexUMI相对位姿公式和L-BFGS-B六参数目标求解，再做独立验证。

参考 project1yu/DexUMI-main/real_script/calibration/compute_ur5_iphone_offset.py。
原目标是 mean(norm([XYZ,rotvec]_pred-[XYZ,rotvec]_actual))，混合米与弧度。
平移与旋转同时优化；额外按毫米和SO(3)角度评价，不能把目标函数值当毫米误差。
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import uuid

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from solve_tron2_tracker_calibration import (
    load_samples, PoseGroup, inverse, transform, initial_estimate, group_errors, statistics,
)


def as_sixd(T):
    return np.r_[T[:3,3], Rotation.from_matrix(T[:3,:3]).as_rotvec()]


def tracker_pose_for_calibration(T, quaternion_convention='legacy-wzyx'):
    """显式转换输入约定，不改原样本。raw-xyzw 由本次实测配对验证。

    pyvut 保存 q=[r0,r3,r2,r1]；若 HID 原始 r 为 XYZW，正确 WXYZ
    应为 [r3,r0,r1,r2]=[q1,q0,q3,q2]。位置保留原始 XYZ。
    此操作不是给腕部添加一个猜测的欧拉角补偿。
    """
    if quaternion_convention == 'legacy-wzyx':
        return np.array(T, copy=True)
    if quaternion_convention != 'raw-xyzw':
        raise ValueError('不支持的四元数输入约定')
    old_xyzw=Rotation.from_matrix(T[:3,:3]).as_quat()
    old_wxyz=old_xyzw[[3,0,1,2]]
    new_wxyz=old_wxyz[[1,0,3,2]]
    return transform(Rotation.from_quat(new_wxyz[[1,2,3,0]]).as_matrix(), T[:3,3])


def wrist_from_tracker(T_WT_stored, calibration):
    """只计算目标矩阵，不连接或控制机器人；调用者需核对地图和安装身份。"""
    Q=tracker_pose_for_calibration(T_WT_stored, calibration['tracker_quaternion_convention'])
    return np.asarray(calibration['T_BW']) @ Q @ inverse(np.asarray(calibration['T_ET']))


def fit_dexumi(groups, sdk_tcp_offset_m, validation_ids):
    train=[g for g in groups if g.id not in validation_ids]
    held=[g for g in groups if g.id in validation_ids]
    if len(train)<3 or len(held)<3:
        raise ValueError('至少需要3组拟合和3组独立验证，建议16组完整采样')
    if not validation_ids <= {g.id for g in groups}:
        raise ValueError('指定的独立验证组不可用')
    # 原版UR5的tcp_offset=0：先把TRON2 SDK末端换至零偏移法兰，
    # 按原版±0.2m安装范围求解，再把安装变换换回SDK末端。
    C=transform(np.eye(3),-np.asarray(sdk_tcp_offset_m,float))  # T_SF
    flange=[PoseGroup(g.id,g.label,g.robot@C,g.tracker,g.statuses,g.pair_count) for g in train]
    X0,_,observability=initial_estimate(flange)
    P0,Q0=flange[0].robot,flange[0].tracker
    relative=[inverse(Q0)@g.tracker for g in flange]
    actual=np.array([as_sixd(g.robot) for g in flange])
    def unpack(v):
        return transform(Rotation.from_rotvec(v[3:]).as_matrix(),v[:3])
    def objective(v):
        X=unpack(v)
        predicted=np.array([as_sixd(P0@X@D@inverse(X)) for D in relative])
        return float(np.mean(np.linalg.norm(predicted-actual,axis=1)))
    # 几何初值和原版初值只按拟合组损失比较，不用留出组挑选或微调。
    original=np.array([[0,-1,0],[-1,0,0],[0,0,-1.]])
    seeds=[np.r_[np.clip(X0[:3,3],-.199,.199),Rotation.from_matrix(X0[:3,:3]).as_rotvec()],
           np.r_[np.zeros(3),Rotation.from_matrix(original).as_rotvec()]]
    bounds=[(-.2,.2)]*3+[(-np.pi,np.pi)]*3
    candidates=[minimize(objective,seed,method='L-BFGS-B',bounds=bounds,
                         options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8,'maxls':40}) for seed in seeds]
    finite=[r for r in candidates if r.success and np.isfinite(r.fun) and np.isfinite(r.x).all()]
    if not finite:
        raise ValueError('DexUMI L-BFGS-B未收敛：'+str([r.message for r in candidates]))
    best=min(finite,key=lambda r:r.fun)
    X_FT=unpack(best.x); X=C@X_FT
    Y=train[0].tracker@inverse(X)@inverse(train[0].robot)
    rows=[{**group_errors(g,X,Y),'split':'validation' if g.id in validation_ids else 'fit'} for g in groups]
    metrics={name:statistics([r for r in rows if r['split']==name]) for name in ('fit','validation')}
    bound_hit=bool(np.any(abs(best.x[:3])>.1999))
    passed=not bound_hit and all(
        metrics[s]['wrist_position_error_mm']['rmse']<=5
        and metrics[s]['wrist_position_error_mm']['max']<=15
        and metrics[s]['rotation_error_deg']['rmse']<=2
        and metrics[s]['rotation_error_deg']['max']<=6 for s in metrics)
    unconfirmed_statuses=sorted({s for g in groups for s in g.statuses if s!=2})
    return {'status':'passed' if passed and not unconfirmed_statuses else 'review_required',
            'geometric_validation_passed':bool(passed),'unconfirmed_tracking_statuses':unconfirmed_statuses,
            'method':'DexUMI relative pose / joint translation+rotation L-BFGS-B',
            'reference_code':'project1yu/DexUMI-main/real_script/calibration/compute_ur5_iphone_offset.py',
            'formula':'E_i = E_0 @ T_ET @ inv(T_0) @ T_i @ inv(T_ET)',
            'frame_conversion':'fit at zero-offset flange like original UR5, export at TRON2 SDK TCP',
            'position_unit':'m','quaternion_order':'wxyz','T_ET':X.tolist(),'T_FT':X_FT.tolist(),
            'T_WB':Y.tolist(),'T_BW':inverse(Y).tolist(),
            'optimizer_objective_mixed_m_rad':float(best.fun),'translation_bound_hit':bound_hit,
            'optimizer_message':str(best.message),'observability':observability,
            'fit_group_ids':[g.id for g in train],'validation_group_ids':sorted(validation_ids),
            'metrics':metrics,'per_group':rows,'applied_to_replay':False,
            'warnings':['误差反映两套测量一致性，不是外部真值精度；需验证支架刚性和模型零位',
                        '静止窗口代替原版动态帧的固定延迟裁剪，未猜测VIVE和TRON2硬件时延']}


def solve_dexumi(path,validation_ids=None,diagnostic_statuses=(),quaternion_convention='legacy-wzyx',analysis_group_ids=None):
    path=Path(path).resolve()
    output=path.parent/'solutions'/('dexumi_'+datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    report={'status':'failed','applied_to_replay':False,'input_file':str(path),'output_dir':str(output)}
    try:
        meta,groups,rejected,digest=load_samples(path,diagnostic_statuses=diagnostic_statuses)
        if analysis_group_ids is not None:
            selected=set(analysis_group_ids)
            rejected += [{'id':g.id,'reason':'采样计划规定仅用于返回重复性检查，不进入拟合或留出验收'}
                         for g in groups if g.id not in selected]
            groups=[g for g in groups if g.id in selected]
            report['analysis_group_ids']=sorted(selected)
        if quaternion_convention == 'raw-xyzw' and meta.get('tracker_transport',{}).get('kind') != 'passive_hidraw':
            raise ValueError('raw-xyzw 只用于已保留 HID 原始报文的旁路采样，不能猜测其它数据源的顺序')
        groups=[PoseGroup(g.id,g.label,g.robot,tracker_pose_for_calibration(g.tracker,quaternion_convention),
                          g.statuses,g.pair_count) for g in groups]
        if meta.get('mock'):
            raise ValueError('此入口不把模拟数据认定为实机标定')
        offset=meta.get('robot_kinematics',{}).get('tcp_offset_m')
        if offset is None:
            raise ValueError('缺少SDK TCP定义，不能猜测法兰转换')
        if validation_ids is None:validation_ids={4,8,12,16}
        report.update(fit_dexumi(groups,offset,set(validation_ids)))
        report.update(session=meta,input_sha256=digest,rejected_groups=rejected,
                      tracker_quaternion_convention=quaternion_convention,
                      input_quaternion_conversion=('stored_wxyz[q1,q0,q3,q2] -> corrected_wxyz'
                                                   if quaternion_convention=='raw-xyzw' else 'identity'),
                      convention_evidence='本次机器人刚性配对的拟合组对比，留出组独立检验；未修改采集器')
        (output/'calibration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
        with (output/'errors.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['id','split','label','wrist_position_error_mm','rotation_error_deg'],extrasaction='ignore')
            w.writeheader();w.writerows(report['per_group'])
    except (ValueError,OSError,KeyError,TypeError,np.linalg.LinAlgError) as exc:
        report['error']=str(exc)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    lines=['DexUMI 腕部标定结果：'+report['status']]
    if 'error' in report:lines.append(report['error'])
    if 'T_ET' in report:
        lines.append('输入四元数解释：'+report['tracker_quaternion_convention'])
        for name in ('T_ET','T_WB','T_BW'):
            lines += [name,np.array2string(np.array(report[name]),precision=9)]
        for split,m in report['metrics'].items():
            lines.append(f"{split}: 位置 RMSE={m['wrist_position_error_mm']['rmse']:.3f} mm / 最大={m['wrist_position_error_mm']['max']:.3f} mm；旋转 RMSE={m['rotation_error_deg']['rmse']:.3f}° / 最大={m['rotation_error_deg']['max']:.3f}°")
        lines.extend(report['warnings'])
        lines.append('未纳入组：'+str(report['rejected_groups']))
    lines.append('结果目录：'+str(output));text='\n'.join(lines)+'\n'
    (output/'report.txt').write_text(text);print(text,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--tracker-quaternion',choices=['legacy-wzyx','raw-xyzw'],default='legacy-wzyx')
    args=p.parse_args()
    result=solve_dexumi(args.input,quaternion_convention=args.tracker_quaternion)
    raise SystemExit(0 if result['status']=='passed' else 1)
