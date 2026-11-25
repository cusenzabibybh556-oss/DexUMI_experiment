"""Speed/scale contracts and frozen-profile invariants, without hardware."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from tools.sweep_tron2_speed_scale import ScaledEpisodeReference, cached_baseline, run_job
from tools.validate_tron2_episode_batch import DEFAULT_PROFILE, DEFAULT_MAPPING
from tracker_replay_mapping import load_calibration
from tools.analyze_tron2_speed_scale import summarize_cell, rejection_category


class SweepReferenceTests(unittest.TestCase):
    def fixture(self, folder, duplicates=False):
        p=Path(folder)/'episode_000001.hdf5'
        t=np.array([0, .05, .1])
        position=np.array([[0,0,0], [.01,.002,-.003], [.03,.006,-.005]])
        q=Rotation.from_euler('z',[0,10,20],degrees=True).as_quat()[:,[3,0,1,2]][:,[1,0,3,2]]
        if duplicates:
            indices=[0,1,1,2];t=t[indices];position=position[indices];q=q[indices]
        with h5py.File(p,'w') as f:
            f.create_dataset('timestamps/master_timestamp_ns',data=np.rint(t*1e9).astype(np.int64))
            f.create_dataset('vive_tracker/pos',data=position[:,None,:])
            f.create_dataset('vive_tracker/quat',data=q[:,None,:])
            f.create_dataset('vive_tracker/valid',data=np.ones((len(t),1),bool))
            f['vive_tracker'].attrs.update(position_unit='meter',quat_order='[w, x, y, z]')
            f.attrs['hand_side']='left'
        return p

    def test_speed_changes_time_scale_changes_only_translation_about_anchor(self):
        mapping=dict(T_BW=np.eye(4).tolist(),T_ET=np.eye(4).tolist(),
            tracker_quaternion_convention='raw-xyzw',sha256='fixture')
        with TemporaryDirectory() as folder:
            path=self.fixture(folder);original=path.read_bytes()
            slow=ScaledEpisodeReference(path,mapping,.25,.5)
            fast=ScaledEpisodeReference(path,mapping,1.,1.5)
            self.assertAlmostEqual(slow.duration/fast.duration,4)
            anchor=np.array([.45,.25,.7]);initial=Rotation.from_euler('x',30,degrees=True).as_matrix()
            ps,rs=slow.target(.16,anchor,initial);pf,rf=fast.target(.04,anchor,initial)
            np.testing.assert_allclose(pf-anchor,3*(ps-anchor),atol=1e-14)
            np.testing.assert_allclose(rs,rf,atol=1e-14)
            np.testing.assert_allclose(slow.target(0,anchor,initial)[0],anchor)
            slow_dt=np.diff(slow.times);fast_dt=np.diff(fast.times)
            slow_v=np.diff(slow.delta,axis=0)/slow_dt[:,None]
            fast_v=np.diff(fast.delta,axis=0)/fast_dt[:,None]
            np.testing.assert_allclose(fast_v,12*slow_v,atol=1e-12)
            slow_a=np.diff(slow_v,axis=0)/(.5*(slow_dt[1:]+slow_dt[:-1]))[:,None]
            fast_a=np.diff(fast_v,axis=0)/(.5*(fast_dt[1:]+fast_dt[:-1]))[:,None]
            self.assertGreater(np.linalg.norm(slow_a),0)
            np.testing.assert_allclose(fast_a,48*slow_a,atol=1e-12)
            self.assertEqual(path.read_bytes(),original)

    def test_invalid_scale_and_duplicate_source_cannot_silently_enter_strict_sweep(self):
        with TemporaryDirectory() as folder:
            p=self.fixture(folder,duplicates=True);mapping=load_calibration(DEFAULT_MAPPING)
            for scale in [0,-1,float('nan'),3.1]:
                with self.subTest(scale=scale),self.assertRaises(ValueError):
                    ScaledEpisodeReference(p,mapping,.5,scale)
            with self.assertRaisesRegex(ValueError,'strictly replay-eligible'):
                ScaledEpisodeReference(p,mapping,.5,1.)

    def test_stress_cells_cannot_reuse_unit_scale_half_speed_results(self):
        for speed,scale in [(1.,1.),(.5,.5),(1.5,1.5)]:
            job=dict(cache_verified=True,speed=speed,scale=scale)
            self.assertIsNone(cached_baseline(job,None,None,Path('/does-not-exist')))

    def test_physical_preview_and_feedback_settings_do_not_change_with_speed(self):
        with TemporaryDirectory() as folder:
            p=self.fixture(folder);mapping=load_calibration(DEFAULT_MAPPING)
            profile=json.loads(DEFAULT_PROFILE.read_text())
            expected=profile['physics_settings'];seen=[]
            ref=ScaledEpisodeReference(p,mapping,.2,.5)
            def fake_experiment(*args):
                settings=args[8];seen.append(settings)
                return dict(explicit_torque_feedforward=False),None
            with patch('tools.sweep_tron2_speed_scale.experiment',side_effect=fake_experiment):
                for speed in [.2,1.5]:
                    job=dict(episode_id=1,speed=speed,scale=.5,case='full_pose',source=str(p),
                        expected_motion_sha256=ref.metadata['source_motion_sha256'],
                        profile=str(DEFAULT_PROFILE),calibration=str(DEFAULT_MAPPING),cache_verified=False,
                        folder=str(Path(folder)/f'out_{speed}'))
                    result=run_job(job)
                    self.assertEqual(result['status'],'completed',result.get('error'))
            self.assertEqual(len(seen),2)
            for settings in seen:
                self.assertEqual(settings.reference_preview_s,expected['reference_preview_s'])
                self.assertEqual(settings.reference_delay_s,expected['reference_delay_s'])
                self.assertEqual(settings.pose_feedback.gain_per_s,expected['pose_feedback']['gain_per_s'])
                self.assertEqual(settings.feedback_delay_s,expected['feedback_delay_s'])


class SweepSummaryTests(unittest.TestCase):
    def record(self,i,mode,position):
        return dict(episode_id=i,case=mode,status='completed',metrics=dict(
            position_norm_P95_mm=position,Z_trajectory_RMSE_mm=position/2,orientation_P95_deg=.2))

    def test_failures_remain_in_completion_and_screening_denominators(self):
        records=[self.record(0,'continuous_Z',30),self.record(0,'full_pose',2),
            self.record(1,'continuous_Z',50),dict(episode_id=1,case='full_pose',status='rejected',
                error='Model velocity reference exceeds experimental 2 rad/s bound'),
            dict(episode_id=2,case='continuous_Z',status='rejected',error='Simulation IK residual 2 mm'),
            self.record(2,'full_pose',4)]
        s=summarize_cell(records,[0,1,2]);m=s['models']['full_pose']
        self.assertEqual(m['requested'],3);self.assertEqual(m['completed'],2)
        self.assertEqual(m['passed_5mm_3deg'],2)
        self.assertEqual(s['paired_ids'],[0])
        self.assertEqual(s['paired_baseline_position_P95_mm']['median'],30)
        self.assertEqual(m['rejection_categories'],{'reference_velocity_guard':1})
        json.dumps(s)

    def test_all_rejected_cell_has_no_survivor_error_statistic(self):
        records=[dict(episode_id=0,case=mode,status='rejected',error='Model pose feedforward Cartesian re-solve failed')
            for mode in ['continuous_Z','full_pose']]
        s=summarize_cell(records,[0])
        self.assertEqual(s['models']['full_pose']['position_P95_mm'],dict(n=0))
        self.assertEqual(s['paired_complete_count'],0)
        self.assertEqual(rejection_category(records[0]['error']),'corrected_pose_IK_position')


if __name__=='__main__':
    unittest.main()
