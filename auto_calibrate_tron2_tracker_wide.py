#!/usr/bin/env python3
"""多位置、多旋转轴的左腕标定；保持现有 Tracker 连接，默认仅只读规划。

沿用 DexUMI 零 TCP 的旋转采样思路，扩展为空间中的多个位置。
采样计划预先指定拟合与独立验证组。返回动作不加入拟合。
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import uuid

import numpy as np

from auto_calibrate_tron2_tracker import current_joints, require_tracker, execute_stages
from capture_tron2_tracker_calibration import HardwareSource, CaptureSession, acquire_capture_lock
from dexumi_wrist_solver import solve_dexumi
from tools.tron2_head_pose import HeadClient
from tron2_left_ik import LeftArmIK
from tron2_wide_calibration_plan import build_wide_plan


def solve_wide_samples(path, plan=None):
    """仅以原始质量检查选组；预留组只验证，不按残差挑组。"""
    from solve_tron2_tracker_calibration import load_samples
    _, groups, rejected, _ = load_samples(path)
    if plan is None:
        planned_held=set(range(25,33));analysis_ids=set(range(1,33))
    else:
        sampling=sorted(plan['sample_indices'])
        sample_ids={target_index:group_id for group_id,target_index in enumerate(sampling,1)}
        planned_held={sample_ids[i] for i in plan['held_out_group_indices']}
        analysis_ids=planned_held | {sample_ids[i] for i in plan['fit_group_indices']}
    held = {g.id for g in groups if g.id in planned_held}
    if len(held) < 3:
        raise ValueError(f'计划留出组只有 {len(held)} 个状态正常的完整组，需要补充独立验证；排除记录：{rejected}')
    result = solve_dexumi(path, validation_ids=held, quaternion_convention='raw-xyzw',analysis_group_ids=analysis_ids)
    if 'T_ET' in result:
        result['validation_selection'] = {
            'planned_ids': sorted(planned_held), 'eligible_ids': sorted(held),
            'rule': (plan or {}).get('held_out_rule', '独立空间中心预先留出；仅剔除采样质量/跟踪状态不合格组，不按拟合残差选择')}
        for name in ('calibration.json', 'report.json'):
            (Path(result['output_dir'])/name).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-ip', default='10.192.1.2')
    parser.add_argument('--passive-hid', required=True)
    parser.add_argument('--tracker-mac', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--sampling-range', choices=['wide','large','broad'], default='wide',
                        help='wide/large沿用原采样布局；broad从当前水平起点规划多位置、多轴整臂采样')
    parser.add_argument('--countdown', type=int, default=5, choices=range(5, 31))
    parser.add_argument('--step-seconds', type=float, default=2.0)
    # 此参数只允许保存诊断数据；求解仍严格排除非 2 状态。
    parser.add_argument('--diagnostic-status', type=int, choices=[18, 20], action='append', default=[])
    args = parser.parse_args()
    if not np.isfinite(args.step_seconds) or not 2 <= args.step_seconds <= 10:
        parser.error('每段时长必须在 2–10 秒内')
    root = Path(__file__).resolve().parent / 'tracker_calibrations'
    run = root / (args.sampling_range+'_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    handle = client = session = None
    audit = (run / 'motion.jsonl').open('x')

    def log(event, **fields):
        audit.write(json.dumps({'event': event, 'time': time.time(), **fields}, ensure_ascii=False)+'\n')
        audit.flush()

    def require_mode():
        response = client.request('request_drag_teach_manage', {'cmd': 'status', 'arg': ''}, 'response_drag_teach_manage')
        log('mode', response=response)
        if 'state=ST_DEVED;' not in response['data'].get('message', ''):
            raise RuntimeError('当前不是 High-Level Dev，未发送下一段动作')

    try:
        handle = acquire_capture_lock()
        client = HeadClient(args.robot_ip)
        if client.accid != 'DACH_TRON2A_253':
            raise RuntimeError('机器人身份不符')
        start, dq = current_joints(client)
        if np.max(abs(dq)) > .05:
            raise RuntimeError('机器人仍在运动')
        if args.sampling_range=='broad':
            from tron2_broad_calibration_plan import build_broad_plan
            targets, stages, report = build_broad_plan(LeftArmIK(), start)
        elif args.sampling_range=='large':
            from tron2_large_calibration_plan import build_large_plan
            targets, stages, report = build_large_plan(LeftArmIK(), start)
        else:
            targets, stages, report = build_wide_plan(LeftArmIK(), start)
        report.setdefault('sample_indices', list(range(32)))
        report.setdefault('held_out_rule', '第四空间中心的全部姿态预先留出；不参与优化或初值选择')
        report.update(robot_id=client.accid, step_seconds=args.step_seconds,
                      quaternion_convention='raw-xyzw', original_tracker_connection_preserved=True)
        (run/'plan.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"计划文件：{run/'plan.json'}", flush=True)
        print(f"{len(report['sample_indices'])} 组（含返回复测时另记），{len(report['fit_group_indices'])} 拟合＋{len(report['held_out_group_indices'])} 独立验证；{report['segment_count']} 段 MoveJ，每段 {args.step_seconds:g} 秒。", flush=True)
        print(f"法兰跨度 {report['flange_max_pair_distance_m']*100:.1f} cm；最大朝向差 {report['maximum_pair_orientation_deg']:.1f}°；最小腕关节余量 {report['minimum_wrist_joint_margin_deg']:.1f}°。", flush=True)
        if not args.execute:
            print('只读规划完成；没有打开 Tracker 或发送运动指令。')
            return 0
        require_mode()
        source = HardwareSource(args.robot_ip, 0, passive_hid=args.passive_hid, tracker_mac=args.tracker_mac)
        session = CaptureSession(source, run, lambda _: None, {
            'mock': False, 'tracker_index': 0, 'robot_ip': args.robot_ip,
            'map_label': '现有 Tracker 地图，连接未重启',
            'mount_label': '左腕 Tracker＋支架100g；左右 O6 已拆；多位置完整左臂协调',
            'diagnostic_statuses': args.diagnostic_status,
            'sampling_plan': str(run/'plan.json'),
        })
        session.start()
        deadline = time.monotonic()+20
        while True:
            try:
                require_tracker(session, args.diagnostic_status)
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.1)
        print(f'旁路只读采样文件：{session.path}', flush=True)
        print('即将开始左臂多位置采样；右臂和头部目标保持本次起点。Ctrl+C 不能撤回已接受的 MoveJ。', flush=True)
        for seconds in range(args.countdown, 0, -1):
            print(f'{seconds} 秒后开始', flush=True)
            time.sleep(1)
        fresh, dq = current_joints(client)
        if np.max(abs(fresh-start)) > .005 or np.max(abs(dq)) > .05:
            raise RuntimeError('规划后姿态发生变化，未发送动作')
        require_mode()
        execute_stages(client, session, start, targets, stages, log,
                       duration=args.step_seconds, diagnostic_statuses=args.diagnostic_status,
                       allowed_joints=tuple(range(7)), sample_indices=set(report['sample_indices']))
        final, dq = current_joints(client)
        log('completed', final_q_rad=final.tolist(), final_dq_rad_s=dq.tolist(),
            max_return_error_deg=float(np.rad2deg(np.max(abs(final-start)))))
        path = session.path
        session.close()
        session = None
        # 明确使用原始 HID XYZW 解释；不修改采集程序、不覆盖原始四元数。
        result = solve_wide_samples(path,report)
        print(f"求解结果：{result['status']}；{result['output_dir']}", flush=True)
        return 0 if result['status'] == 'passed' else 2
    except (Exception, KeyboardInterrupt) as exc:
        log('aborted', error=str(exc) or 'KeyboardInterrupt')
        print(f'停止后续动作：{exc}；没有自动返回或重发。记录：{run}', flush=True)
        return 1
    finally:
        if session is not None:
            session.close()
        if client is not None:
            client.socket.close()
        if handle is not None:
            handle.close()
        audit.close()


if __name__ == '__main__':
    raise SystemExit(main())
