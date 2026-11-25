#!/usr/bin/env python3
"""Prior-only train/validation diagnostics and fixed held-out test breakdown.

Does not train, select a new checkpoint, connect hardware or run simulation.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def breakdown(arrays):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from tron2_act.geometry import rotation6d_to_matrix
    predicted,target,valid=arrays['predicted'],arrays['target'],~arrays['is_pad']
    def metrics(mask):
        p,t=predicted[mask],target[mask]
        if not len(p):return {'valid_steps':0}
        e=p[:,:3]-t[:,:3]
        angles=np.rad2deg(Rotation.from_matrix(rotation6d_to_matrix(p[:,3:9])@np.swapaxes(rotation6d_to_matrix(t[:,3:9]),1,2)).magnitude())
        return {'valid_steps':len(p),'position_rmse_mm':float(np.sqrt(np.mean(np.sum(e*e,axis=-1)))*1000),
            'position_p95_mm':float(np.percentile(np.linalg.norm(e,axis=-1),95)*1000),
            'position_axis_rmse_mm':(np.sqrt(np.mean(e*e,axis=0))*1000).tolist(),
            'position_bias_mm':(e.mean(0)*1000).tolist(),
            'rotation_mean_deg':float(angles.mean()),'rotation_p95_deg':float(np.percentile(angles,95)),
            'o6_mae_per_channel':np.abs(p[:,9:]-t[:,9:]).mean(0).tolist(),
            'o6_mae':float(np.abs(p[:,9:]-t[:,9:]).mean())}
    result={'all':metrics(valid),'horizons':{}}
    for k in [0,4,9,19,39,49]:
        mask=np.zeros_like(valid);mask[:,k]=valid[:,k]
        result['horizons'][str((k+1)/20)]=metrics(mask)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=ROOT/'artifacts/tron2_act/cup_stack_baseline_20261117_01')
    parser.add_argument('--prepared',type=Path,default=ROOT/'artifacts/cup_task_audit_20261117/prepared_24')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from tron2_act.engine import evaluate
    torch.set_num_threads(4)
    checkpoint=args.baseline/'train/policy_best.ckpt'
    bundle=torch.load(checkpoint,map_location='cpu',weights_only=False)
    report={'checkpoint':str(checkpoint),'best_step':bundle['step'],'latent':'zero prior on all splits',
        'checkpoint_selection':'unchanged; original full-validation selection',
        'test_usage':'reuse saved fixed-checkpoint test predictions; no test-based checkpoint selection',
        'hardware_connections':0,'training_steps':20000,'splits':{}}
    for split in ['val','train']:
        print('Evaluating prior on',split,flush=True)
        summary=evaluate(args.prepared,checkpoint,args.output/split,split=split,device='cuda',batch_size=8)
        report['splits'][split]={'summary':summary,**breakdown(np.load(args.output/split/'predictions.npz'))}
    test=args.baseline/'test'
    arrays=dict(np.load(test/'predictions.npz'))
    report['splits']['test']={'summary':json.loads((test/'report.json').read_text()),**breakdown(arrays),'episodes':{}}
    items=json.loads((test/'sample_index.json').read_text())
    for source in sorted({x['source'] for x in items}):
        indices=np.array([i for i,x in enumerate(items) if x['source']==source])
        report['splits']['test']['episodes'][Path(source).stem]=breakdown({k:v[indices] for k,v in arrays.items()})
    manifest=bundle['manifest'];report['episode_counts']={k:len(v) for k,v in manifest['splits'].items()}
    report['interpretation']=['Train/validation/test errors are all prior-only and comparable.',
        'Small held-out episode count does not establish a real-robot success rate.',
        'Wrist targets are tracker-origin spatial proxies; O6 targets are glove-derived commands, not future q_pos.',
        'Baseline weights and original test split are unchanged; no additional demonstrations are required to run this review.']
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[json.loads(line) for line in (args.baseline/'train/metrics.jsonl').read_text().splitlines()]
    val=[x for x in records if 'validation' in x]
    fig,axes=plt.subplots(1,2,figsize=(11,4),dpi=160)
    axes[0].plot([x['step'] for x in val],[x['validation']['normalized_l1'] for x in val],label='Full validation, prior')
    axes[0].axvline(bundle['step'],color='r',ls='--',label='Original selected checkpoint')
    axes[0].set(xlabel='Training step',ylabel='Normalized L1',title='Validation convergence');axes[0].legend(fontsize=8)
    for split,color in [('train','gray'),('val','orange'),('test','blue')]:
        h=report['splits'][split]['horizons'];axes[1].plot([float(t) for t in h],[x['position_rmse_mm'] for x in h.values()],marker='o',label=split,color=color)
    axes[1].set(xlabel='Physical prediction horizon (s)',ylabel='Position RMSE (mm)',title='Same checkpoint, prior only');axes[1].legend()
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(args.output/'training_review.png');plt.close(fig)
    print(json.dumps({k:v['summary']['normalized_l1'] for k,v in report['splits'].items()},indent=2),flush=True)


if __name__=='__main__':main()
