#!/usr/bin/env python3
"""只读分析已结束的独立Z补偿日志，不连接任何硬件。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    source = args.run/'command_state.jsonl'
    result_path = args.run/'result.json'
    result = json.loads(result_path.read_text())  # 拒绝把运行中的日志作为已结束结果。
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    starts = [r for r in rows if r['event'] == 'z_comp_start']
    if not starts:
        print(json.dumps({'result': result, 'analysis': 'no motion phase started'}, ensure_ascii=False))
        return
    start = starts[0]['monotonic_ns']
    commands = [r for r in rows if r['event'] == 'z_comp_command']
    poses = [r for r in rows if r['event'] == 'diagnostic_pose']
    if not commands or not poses:
        raise ValueError('没有足够目标或反馈记录')
    ct = np.array([(r['monotonic_ns']-start)/1e9 for r in commands])
    applied_times = np.array([(r.get('offset_applied_monotonic_ns', r['monotonic_ns'])-start)/1e9
                              for r in commands])
    nominal = np.array([r['nominal_left'] for r in commands])
    issued = np.array([r['issued_left'] for r in commands])
    offsets = np.array([r['offset_m'] for r in commands])
    pt_all = np.array([((r['sent_monotonic_ns']+r['received_monotonic_ns'])/2-start)/1e9 for r in poses])
    keep = pt_all >= ct[0]
    poses = [r for r, valid in zip(poses, keep) if valid]
    pt = pt_all[keep]
    actual = np.array([r['reply']['data']['left_position'] for r in poses])
    right = np.array([r['reply']['data']['right_position'] for r in poses])
    matching = np.maximum(0, np.searchsorted(ct, pt, side='right')-1)
    error_mm = (actual[:, 2]-nominal[matching, 2])*1000.
    inner_error_mm = (actual[:, 2]-issued[matching, 2])*1000.
    right_anchor = np.asarray(starts[0]['right_position'])
    stops = [r for r in rows if r['event'] == 'z_comp_sending_stopped']
    stop = (stops[0]['monotonic_ns']-start)/1e9 if stops else ct[-1]
    phases = [r for r in rows if r['event'] == 'z_comp_phase']
    phase_times = [(r['monotonic_ns']-start)/1e9 for r in phases]
    summary = {'execution_result': result['result'], 'controller_result': result,
        'command_records': len(commands), 'feedback_samples': len(poses),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (source, result_path)},
        'maximum_offset_mm': float(offsets.max()*1000),
        'maximum_command_offset_rate_mm_s': float(np.max(abs(np.diff(offsets)/np.diff(ct)))*1000),
        'maximum_algorithm_offset_rate_mm_s': (float(np.max(abs(np.diff(offsets)/np.diff(applied_times)))*1000)
                                               if all('offset_applied_monotonic_ns' in r for r in commands) else None),
        'right_maximum_drift_mm': float(np.linalg.norm(right-right_anchor, axis=1).max()*1000),
        'phases': {}, 'limitations': [
            'SDK pose feedback, not external metrology or measured contact force.',
            'Position-only Z outer loop; not proof that firmware gravity feedforward was restored.',
            'A new nonzero offset must not be dropped abruptly after the run.',
            'Feedback is paired by host query midpoint and latest preceding command; not a shared robot-clock claim.']}
    for i, (phase, first) in enumerate(zip(phases, phase_times)):
        last = phase_times[i+1] if i+1 < len(phases) else stop
        mask = (pt >= max(first, last-1.)) & (pt <= last)
        if not mask.any():
            continue
        summary['phases'][phase['phase']] = {
            'start_s': first, 'end_s': last, 'tail_feedback_samples': int(mask.sum()),
            'nominal_z_m': float(nominal[matching[mask], 2].mean()),
            'actual_z_m': float(actual[mask, 2].mean()),
            'actual_minus_nominal_z_mm': float(error_mm[mask].mean()),
            'actual_minus_issued_z_mm': float(inner_error_mm[mask].mean()),
            'offset_mm': float(offsets[matching[mask]].mean()*1000),
            'maximum_tail_error_mm': float(abs(error_mm[mask]).max())}
    if phases:
        first_end = phase_times[1] if len(phases) > 1 else stop
        mask = pt < first_end
        if mask.any():
            summary['initial_hold_maximum_drop_mm'] = float(max(0., nominal[0,2]-actual[mask,2].min())*1000)
    post = pt > stop
    if post.any():
        summary['post_stop_actual_minus_nominal_z_mm'] = float(error_mm[post].mean())
    final_poses = [r for r in rows if r['event'] == 'z_comp_final_pose']
    if final_poses:
        summary['final_readonly_actual_minus_nominal_z_mm'] = float(
            (final_poses[-1]['reply']['data']['left_position'][2]-nominal[-1, 2])*1000)
    out = args.run/'analysis'
    out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    np.savez_compressed(out/'series.npz', command_time_s=ct, nominal=nominal,
                        issued=issued, offset_m=offsets, pose_time_s=pt,
                        actual=actual, nominal_error_mm=error_mm, inner_error_mm=inner_error_mm)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    base = nominal[0, 2]
    axes[0].plot(ct, (nominal[:,2]-base)*1000, label='Nominal physical reference')
    axes[0].plot(ct, (issued[:,2]-base)*1000, label='Issued ServoP reference')
    axes[0].plot(pt, (actual[:,2]-base)*1000, label='SDK measured pose')
    axes[0].set_ylabel('Relative Z (mm)'); axes[0].legend()
    axes[1].plot(pt, error_mm, label='Actual - nominal')
    axes[1].plot(pt, inner_error_mm, label='Actual - issued', alpha=.7)
    axes[1].axhline(0, color='k', lw=.5); axes[1].set_ylabel('Z error (mm)'); axes[1].legend()
    axes[2].plot(ct, offsets*1000); axes[2].set_ylabel('Correction (mm)'); axes[2].set_xlabel('Host relative time (s)')
    for ax in axes:
        ax.grid(True, alpha=.25)
        for t in phase_times:
            ax.axvline(t, color='grey', alpha=.3, lw=.6)
    fig.tight_layout(); fig.savefig(out/'z_compensation.png', dpi=160); plt.close(fig)
    lines = ['# 独立Z位置闭环试验', '',
             '本报告只读分析已结束日志，没有连接硬件。Z修正不是接触力控制，也不证明固件重力前馈已恢复。', '',
             f"执行结果：`{result['result']}`；最大修正 {summary['maximum_offset_mm']:.3f} mm。", '',
             '| 阶段末1秒 | 实际−物理参考 Z / mm | 实际−发出目标 Z / mm | 发出修正 / mm |',
             '|---|---:|---:|---:|']
    for phase, metrics in summary['phases'].items():
        lines.append(f"| {phase} | {metrics['actual_minus_nominal_z_mm']:.3f} | {metrics['actual_minus_issued_z_mm']:.3f} | {metrics['offset_mm']:.3f} |")
    lines += ['', '启动瞬间的最低点和稳定后的误差分开报告，不能用后者宣称首次接管完全不下沉。', '',
              '结束时保留最后修正。重新启动另一个未补偿控制入口可能撤掉已有支撑；停发不等同机器人急停。', '',
              '算法限速按修正计算时间执行；实际命令日志时刻还受检查和发送排程影响，二者差分速率分别保留，不能混称硬件速度保证。', '',
              f"停发后均值误差：{summary.get('post_stop_actual_minus_nominal_z_mm')} mm；最终单次只读误差：{summary.get('final_readonly_actual_minus_nominal_z_mm')} mm。", '',
              '![高度、参考和修正](z_compensation.png)', '', '详细数值、源哈希及限制见 `summary.json`。', '']
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'analysis': str(out), 'execution_result': result['result'],
                      'phases': summary['phases']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
