#!/usr/bin/env python3
"""Audit frozen speed/scale simulations; retain failed cases in denominators."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_tron2_episode_batch import describe
from tools.validate_tron2_episode_batch import read_episode, motion_digest, inspect_arrays, EpisodeReference
from tools.sweep_tron2_speed_scale import cell_key
from tracker_replay_mapping import load_calibration


def rejection_category(message):
    for phrase,category in [
        ('Model velocity reference','reference_velocity_guard'),
        ('Combined 6D reference','combined_pose_correction_guard'),
        ('Model pose feedforward exceeds','model_pose_correction_guard'),
        ('Cartesian re-solve','corrected_pose_IK_position'),
        ('angular re-solve','corrected_pose_IK_rotation'),
        ('Simulation IK residual','command_pose_IK_position'),
        ('Simulation IK angle','command_pose_IK_rotation'),
        ('6D feedback error','feedback_error_guard'),('diverged','simulation_diverged')]:
        if phrase in message:return category
    return 'other'


def summarize_cell(records,ids):
    by={(r['episode_id'],r['case']):r for r in records}
    models={}
    for label in ['continuous_Z','full_pose']:
        group=[by[i,label] for i in ids if (i,label) in by]
        done=[r for r in group if r['status']=='completed']
        models[label]=dict(requested=len(ids),finished=len(group),completed=len(done),
            rejected=len(group)-len(done),complete_ids=[r['episode_id'] for r in done],
            position_P95_mm=describe([r['metrics']['position_norm_P95_mm'] for r in done]),
            Z_RMSE_mm=describe([r['metrics']['Z_trajectory_RMSE_mm'] for r in done]),
            orientation_P95_deg=describe([r['metrics']['orientation_P95_deg'] for r in done]),
            passed_5mm_3deg=sum(r['metrics']['position_norm_P95_mm']<=5 and
                r['metrics']['orientation_P95_deg']<=3 for r in done),
            rejection_categories=dict(Counter(rejection_category(r['error']) for r in group if r['status']!='completed')))
    pairs=sorted(set(models['continuous_Z']['complete_ids'])&set(models['full_pose']['complete_ids']))
    improvements=[100*(1-by[i,'full_pose']['metrics']['position_norm_P95_mm']/
        by[i,'continuous_Z']['metrics']['position_norm_P95_mm']) for i in pairs]
    return dict(models=models,paired_complete_count=len(pairs),paired_ids=pairs,
        paired_baseline_position_P95_mm=describe([by[i,'continuous_Z']['metrics']['position_norm_P95_mm'] for i in pairs]),
        paired_candidate_position_P95_mm=describe([by[i,'full_pose']['metrics']['position_norm_P95_mm'] for i in pairs]),
        paired_position_reduction_percent=describe(improvements),
        candidate_position_worsened=sum(int(x<0) for x in improvements))


def audit_record(record,source_reference):
    if record['status']!='completed':return None
    folder=Path(record['artifact_directory']);m=record['metrics'];stem=f"mass_0.396_{m['mode']}"
    d=np.load(folder/(stem+'.npz'));p=np.load(folder/(stem+'_pose.npz'))
    t=d['time_s'];active=(t>=30)&(t<=m['trajectory_end_s'])
    # Rebuild nominal targets independently of the new scaled wrapper, using
    # the original unscaled loader and source clock. This checks the sweep's
    # speed/scale semantics, beyond merely recomputing a self-consistent error.
    query=np.clip((t-30)*record['speed'],0,source_reference.duration)
    expected_position=d['nominal_xyz_m'][0]+record['scale']*np.column_stack([
        np.interp(query,source_reference.times,source_reference.delta[:,j]) for j in range(3)])
    expected_rotation=source_reference.rotations(query).as_matrix()@p['nominal_rotation'][0]
    assert np.max(abs(expected_position-d['nominal_xyz_m']))<1e-10
    assert np.max(abs(expected_rotation-p['nominal_rotation']))<1e-9
    error=(d['actual_xyz_m']-d['nominal_xyz_m'])*1000
    angular=np.rad2deg(Rotation.from_matrix(p['actual_rotation']@np.swapaxes(p['nominal_rotation'],1,2)).magnitude())
    assert abs(np.percentile(np.linalg.norm(error[active],axis=1),95)-m['position_norm_P95_mm'])<1e-8
    assert abs(np.sqrt(np.mean(error[active,2]**2))-m['Z_trajectory_RMSE_mm'])<1e-8
    assert abs(np.percentile(angular[active],95)-m['orientation_P95_deg'])<1e-8
    assert np.max(abs(d['issued_z_m']-d['issued_xyz_m'][:,2]))<1e-12
    dt=np.diff(np.rint(t*1e9).astype(np.int64))/1e9
    fp=p['residual_translation_m'];fr=Rotation.from_rotvec(p['residual_rotation_vector_rad'])
    assert np.linalg.norm(fp,axis=1).max()<=.01+1e-10
    assert np.rad2deg(fr.magnitude()).max()<=2+1e-8
    assert (np.linalg.norm(np.diff(fp,axis=0),axis=1)/dt).max()<=.003+1e-10
    assert (np.rad2deg((fr[1:]*fr[:-1].inv()).magnitude())/dt).max()<=1+1e-8
    generated=d['generated_joint_reference_rad'];delivered=d['delivered_joint_reference_rad']
    expected=np.column_stack([np.interp(t-record['physics_settings']['reference_delay_s'],t,generated[:,j]) for j in range(16)])
    assert np.max(abs(expected-delivered))<1e-9
    return dict(episode_id=record['episode_id'],cell=record['cell'],case=record['case'],
        trajectory_samples=int(active.sum()),position_max_mm=float(np.linalg.norm(error[active],axis=1).max()),
        pose_rate_and_amplitude_bounds_verified=True,metrics_recomputed=True,reference_delay_verified=True,
        original_source_time_translation_scale_and_unscaled_rotation_verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();out=args.directory
    manifest=json.loads((out/'manifest.json').read_text())
    records=json.loads((out/'cases.json').read_text())
    ids=[r['episode_id'] for r in manifest['cohort']]
    expected_count=len(ids)*len(manifest['speeds'])*len(manifest['translation_scales'])*2
    assert len(records)==expected_count and len({(r['cell'],r['episode_id'],r['case']) for r in records})==expected_count
    for path,sha in manifest['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    calibration_hash=records[0]['reference']['calibration_sha256']
    calibration_paths=[Path(path) for path,sha in manifest['source_sha256'].items() if sha==calibration_hash]
    assert len(calibration_paths)==1
    mapping=load_calibration(calibration_paths[0])
    source_references={r['episode_id']:EpisodeReference(Path(r['source']),mapping,1.) for r in manifest['cohort']}
    source_kinematics=[]
    for row in manifest['cohort']:
        arrays,attrs=read_episode(Path(row['source']))
        assert motion_digest(arrays)==row['audit']['motion_input_sha256']
        _,indices=inspect_arrays(arrays,attrs)
        t=arrays['timestamps/master_timestamp_ns'][indices]
        dt=np.diff(t)/1e9;p=arrays['vive_tracker/pos'][indices,0].astype(float)
        velocity=np.diff(p,axis=0)/dt[:,None]
        acceleration=np.diff(velocity,axis=0)/(.5*(dt[1:]+dt[:-1]))[:,None]
        source_kinematics.append(dict(episode_id=row['episode_id'],
            maximum_linear_velocity_m_s=float(np.linalg.norm(velocity,axis=1).max()),
            maximum_discrete_acceleration_m_s2=float(np.linalg.norm(acceleration,axis=1).max()),
            maximum_angular_velocity_deg_s=row['audit']['source_angular_speed_deg_s']['maximum'],
            maximum_displacement_mm=row['audit']['mapped_maximum_displacement_mm']))
    summary=dict(planned_cases=expected_count,finished_cases=len(records),
        completed=sum(r['status']=='completed' for r in records),
        rejected=sum(r['status']!='completed' for r in records),
        reused=sum('reused_baseline' in r for r in records),episodes=ids,cells={},common_cohort_by_scale={},
        same_linear_velocity_factor_groups=[],
        source_kinematics=source_kinematics,
        acceleration_semantics='Finite difference of segment velocities at source samples; piecewise-linear targets have velocity corners, not continuous bounded acceleration')
    index={(r['speed'],r['scale'],r['episode_id'],r['case']):r for r in records}
    for speed in manifest['speeds']:
        for scale in manifest['translation_scales']:
            key=cell_key(speed,scale)
            group=[r for r in records if r['cell']==key]
            summary['cells'][key]=dict(speed=speed,scale=scale,**summarize_cell(group,ids),
                nominal_maximum_linear_velocity_m_s=describe([r['maximum_linear_velocity_m_s']*speed*scale for r in source_kinematics]),
                nominal_maximum_discrete_acceleration_m_s2=describe([r['maximum_discrete_acceleration_m_s2']*speed**2*scale for r in source_kinematics]),
                nominal_maximum_angular_velocity_deg_s=describe([r['maximum_angular_velocity_deg_s']*speed for r in source_kinematics]),
                nominal_maximum_displacement_mm=describe([r['maximum_displacement_mm']*scale for r in source_kinematics]))
    for scale in manifest['translation_scales']:
        common=[i for i in ids if all(index[speed,scale,i,mode]['status']=='completed'
            for speed in manifest['speeds'] for mode in ['continuous_Z','full_pose'])]
        summary['common_cohort_by_scale'][str(scale)]=dict(episodes=common,
            speed_comparisons=[dict(speed=speed,**summarize_cell(
                [index[speed,scale,i,mode] for i in common for mode in ['continuous_Z','full_pose']],common))
                for speed in manifest['speeds']])
    factor_groups={}
    for cell in summary['cells'].values():
        factor_groups.setdefault(round(cell['speed']*cell['scale'],10),[]).append(cell)
    for factor,cells in factor_groups.items():
        if len(cells)<2:continue
        common=[i for i in ids if all(index[c['speed'],c['scale'],i,'full_pose']['status']=='completed' for c in cells)]
        summary['same_linear_velocity_factor_groups'].append(dict(translation_velocity_factor=factor,
            common_episodes=common,cells=[dict(speed=c['speed'],scale=c['scale'],
                discrete_acceleration_factor=c['scale']*c['speed']**2,
                angular_velocity_factor=c['speed'],
                position_P95_mm=describe([index[c['speed'],c['scale'],i,'full_pose']['metrics']['position_norm_P95_mm'] for i in common]))
                for c in cells]))
    audit=[];flat=[]
    for j,r in enumerate(records):
        if 'physics_settings' in r:
            expected_settings=dict(manifest['frozen_profile']['physics_settings'])
            expected_settings['reference_preview_s']=manifest['frozen_profile']['physics_settings']['reference_preview_s'] if r['case']=='full_pose' else 0.
            expected_settings['feedback_pairing']='capture'
            assert r['physics_settings']==expected_settings
            assert r['reference']['speed']==r['speed'] and r['reference']['scale']==r['scale']
            assert r['reference']['orientation_scale']==1.
        row={k:r[k] for k in ['episode_id','speed','scale','case','status']}
        row.update(error=r.get('error',''),rejection_category=rejection_category(r['error']) if r['status']!='completed' else '',
                   reused='reused_baseline' in r)
        a=audit_record(r,source_references[r['episode_id']])
        if a:
            audit.append(a);m=r['metrics']
            row.update(position_P95_mm=m['position_norm_P95_mm'],Z_RMSE_mm=m['Z_trajectory_RMSE_mm'],
                orientation_P95_deg=m['orientation_P95_deg'],position_max_mm=a['position_max_mm'],
                residual_max_mm=m['maximum_pose_feedback_translation_mm'],
                model_shift_max_mm=m['maximum_model_pose_shift_mm'],
                screening_pass=m['position_norm_P95_mm']<=5 and m['orientation_P95_deg']<=3)
        flat.append(row)
        if j%100==0:print('AUDITED',j+1,'of',len(records),flush=True)
    fields=['episode_id','speed','scale','case','status','position_P95_mm','Z_RMSE_mm','orientation_P95_deg',
        'position_max_mm','residual_max_mm','model_shift_max_mm','screening_pass','reused','rejection_category','error']
    with (out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(flat)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'independent_audit.json').write_text(json.dumps(dict(passed=True,cases=audit,
        unchanged_motion_inputs=True,unchanged_frozen_sources=True,
        completed_cases_audited=len(audit),rejected_cases_not_scored=summary['rejected']),indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    specs=[('median','Completed runs: median position P95 (mm)','viridis'),
        ('completion','Whole-trajectory completion (%)','Blues'),('pass','Pass P95 <=5 mm / <=3 deg (%)','Greens')]
    for ax,(kind,title,cmap) in zip(axes,specs):
        values=np.full((len(manifest['speeds']),len(manifest['translation_scales'])),np.nan)
        for y,speed in enumerate(manifest['speeds']):
            for x,scale in enumerate(manifest['translation_scales']):
                m=summary['cells'][cell_key(speed,scale)]['models']['full_pose']
                value=m['position_P95_mm'].get('median') if kind=='median' else 100*(m['completed'] if kind=='completion' else m['passed_5mm_3deg'])/len(ids)
                if value is not None:values[y,x]=value
        im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=0,vmax=None if kind=='median' else 100)
        for y,speed in enumerate(manifest['speeds']):
            for x,scale in enumerate(manifest['translation_scales']):
                m=summary['cells'][cell_key(speed,scale)]['models']['full_pose'];v=values[y,x]
                label=(f'{v:.2f} mm\n' if kind=='median' and np.isfinite(v) else 'No result\n' if kind=='median' else f'{v:.0f}%\n')
                label+=f"{m['completed'] if kind!='pass' else m['passed_5mm_3deg']}/{len(ids)}"
                ax.text(x,y,label,ha='center',va='center',fontsize=10,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=3))
        ax.set_xticks(range(len(manifest['translation_scales'])),manifest['translation_scales'])
        ax.set_yticks(range(len(manifest['speeds'])),manifest['speeds'])
        ax.set_xlabel('Translation scale; orientation scale = 1');ax.set_ylabel('Speed');ax.set_title(title,fontsize=10)
        fig.colorbar(im,ax=ax,shrink=.65)
    fig.suptitle('Frozen full-pose model: speed / scale sensitivity; rejected runs remain in denominators',fontsize=12)
    fig.tight_layout();fig.savefig(out/'speed_scale_grid.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for scale in manifest['translation_scales']:
        s=summary['common_cohort_by_scale'][str(scale)];n=len(s['episodes'])
        if not n:continue
        data=s['speed_comparisons'];x=[c['speed'] for c in data]
        axes[0].plot(x,[c['models']['full_pose']['position_P95_mm']['median'] for c in data],'.-',label=f'scale {scale}, n={n}')
        axes[1].plot(x,[c['models']['full_pose']['orientation_P95_deg']['median'] for c in data],'.-',label=f'scale {scale}, n={n}')
        axes[2].plot(x,[c['models']['full_pose']['Z_RMSE_mm']['median'] for c in data],'.-',label=f'scale {scale}, n={n}')
    axes[0].set_ylabel('Median position P95 (mm)');axes[0].axhline(5,color='grey',linestyle='--')
    axes[1].set_ylabel('Median orientation P95 (deg)')
    axes[2].set_ylabel('Median Z RMSE (mm)')
    for ax in axes:ax.set_xlabel('Speed');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same completed episode cohort at every speed, separately for each scale')
    fig.tight_layout();fig.savefig(out/'matched_speed_curves.png',dpi=170);plt.close(fig)
    for key,c in summary['cells'].items():
        m=c['models']['full_pose']
        print(key,'complete',m['completed'],'/',len(ids),'P95',m['position_P95_mm'].get('median'),
              'pass',m['passed_5mm_3deg'],'paired',c['paired_complete_count'],flush=True)


if __name__=='__main__':
    main()
