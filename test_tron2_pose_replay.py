"""Offline deployment checks: real model, fake clocks/transports, no hardware."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

import replay_tron2_fixed_start as fixed
import replay_tron2_tracker as tracker
import tron2_pose_replay as backend
from tron2_pose_reference import (VERSION, DEFAULT_PROFILE, load_profile,
    PoseTrajectory, PoseReferenceModel, limit_correction, clamp_correction, preflight)
from tools.diagnose_tron2_z_compensation import PosePacket
from test_replay_tron2_z_compensation_integration import Clock, anchor_pose


def options(*extra):
    return fixed.parse_args(['--episode','91','--pose-compensation','--speed','0.2',
                             '--scale','1','--orientation-scale','1',*extra])


def trajectory(anchor):
    return PoseTrajectory([0.,1.], [anchor.left_position,anchor.left_position+[.01,0,0]],
                          np.tile(anchor.left_quaternion,(2,1)))


class ProfileAndModelTests(unittest.TestCase):
    def test_cli_opt_in_and_mutual_exclusion(self):
        self.assertFalse(fixed.parse_args(['--episode','91']).pose_compensation)
        for extra in (['--z-compensation'],['--settle-handover'],['--z-adaptation','continuous'],
                      ['--speed','1'],['--scale','1.5'],['--pose-preview-ms','nan'],
                      ['--pose-feedback-age-ms','101'],['--plan-ik']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                backend.validate_pose_options(options(*extra))
        a=options();backend.validate_pose_options(a)
        self.assertEqual(a.pose_preview_ms,124.)
        self.assertEqual(a.pose_feedback_age_ms,0.)

    def test_simulation_profile_or_widened_bounds_rejected(self):
        p=json.loads(DEFAULT_PROFILE.read_text())
        variants=[dict(p,schema='simulation_only'),dict(p,maximum_total_rotation_deg=6),
                  dict(p,maximum_correction_rate_m_s=float('nan')),dict(p,handover_model_ramp_s=1)]
        bad=deepcopy(p);bad['model']['sha256']='0'*64;variants.append(bad)
        with TemporaryDirectory() as d:
            f=Path(d)/'profile.json'
            for v in variants:
                f.write_text(json.dumps(v))
                with self.assertRaises(ValueError):load_profile(f)

    def test_slew_translation_and_world_rotation_rate(self):
        p=np.array([.001,.002,-.003]);r=Rotation.from_euler('xyz',[1,2,3],degrees=True).as_matrix()
        wanted_r=Rotation.from_rotvec([.03,-.04,.02]).as_matrix()@r
        out,rotation,limited=limit_correction(p,r,p+[.01,.02,-.01],wanted_r,.01,.04,5.)
        self.assertTrue(limited)
        self.assertIs(type(limited),bool)
        json.dumps({'correction_rate_limited':limited})
        # Angular-only limiting used to return np.bool_ via the RHS of `or`,
        # killing the asynchronous trace writer late in the real trajectory.
        _,_,angular_only=limit_correction(p,r,p,wanted_r,.01,.04,5.)
        self.assertIs(type(angular_only),bool)
        self.assertTrue(angular_only)
        json.dumps({'correction_rate_limited':angular_only})
        self.assertAlmostEqual(np.linalg.norm(out-p),.0004)
        self.assertAlmostEqual(np.rad2deg(Rotation.from_matrix(rotation@r.T).magnitude()),.05)

    def test_total_correction_is_clamped_without_changing_direction(self):
        p=np.array([.06,.08,0.]);r=Rotation.from_euler('z',10,degrees=True).as_matrix()
        bounded_p,bounded_r,limited=clamp_correction(p,r,.04,5.)
        self.assertTrue(limited)
        self.assertAlmostEqual(np.linalg.norm(bounded_p),.04)
        self.assertAlmostEqual(np.rad2deg(Rotation.from_matrix(bounded_r).magnitude()),5.)
        np.testing.assert_allclose(bounded_p/np.linalg.norm(bounded_p),p/np.linalg.norm(p))

    def test_actual_model_sdk_anchor_and_zero_weight_first_reference(self):
        q=np.r_[fixed.FIXED_LEFT_Q,np.zeros(9)]
        model=PoseReferenceModel(load_profile(),q)
        p,r,_=model.forward(q)
        np.testing.assert_allclose(p,fixed.FIXED_LEFT_POSITION,atol=1e-9)
        expected=Rotation.from_quat(fixed.FIXED_LEFT_QUATERNION[[1,2,3,0]]).as_matrix()
        np.testing.assert_allclose(r,expected,atol=1e-9)
        pi,ri,_=model.step(p,r,1/300,weight=0.)
        np.testing.assert_allclose(pi,p,atol=1e-12)
        np.testing.assert_allclose(ri,r,atol=1e-12)
        pi,ri,stats=model.step(p,r,1/300)
        self.assertGreater(pi[2]-p[2],.01)
        qi=model.qissued.copy();actual,orientation,_=model.forward(qi)
        self.assertLess(np.linalg.norm(actual-pi),.001)
        self.assertLess(Rotation.from_matrix(orientation@ri.T).magnitude(),np.deg2rad(.5))
        np.testing.assert_array_equal(qi[7:],q[7:])

    def test_unreachable_ik_projects_to_furthest_feasible_pose(self):
        q=np.r_[fixed.FIXED_LEFT_Q,np.zeros(9)]
        model=PoseReferenceModel(load_profile(),q)
        start_p,start_r,_=model.forward(q)
        p,r,projected,fraction=model.solve_toward(start_p+[1.,0,0],start_r,q)
        self.assertGreaterEqual(fraction,0.)
        self.assertLess(fraction,1.)
        solved_p,solved_r,_=model.forward(projected)
        np.testing.assert_allclose(solved_p,p,atol=.001)
        self.assertLess(Rotation.from_matrix(solved_r@r.T).magnitude(),np.deg2rad(.5))

    def test_internal_reference_velocity_is_saturated_instead_of_aborting(self):
        q=np.r_[fixed.FIXED_LEFT_Q,np.zeros(9)]
        model=PoseReferenceModel(load_profile(),q)
        p,r,_=model.forward(q);model.velocity[:]=3.
        _,_,stats=model.step(p,r,1/300)
        self.assertTrue(stats['reference_velocity_limited'])
        self.assertLessEqual(np.max(np.abs(model.velocity)),2.)

    def test_failed_ik_does_not_return_partial_preflight(self):
        q=np.r_[fixed.FIXED_LEFT_Q,np.zeros(9)]
        poses=np.array([fixed.FIXED_LEFT_POSITION, [10,10,10]])
        t=PoseTrajectory([0.,1.],poses,np.tile(fixed.FIXED_LEFT_QUATERNION,(2,1)))
        with self.assertRaises((ValueError,RuntimeError)):preflight(t,q,load_profile())


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.clock.ns=2_000_000_000
        self.timer=patch.object(backend,'time',self.clock);self.timer.start();self.addCleanup(self.timer.stop)
        self.prepare_session()

    def prepare_session(self):
        self.clock.ns=2_000_000_000
        self.anchor=anchor_pose();self.a=options();backend.validate_pose_options(self.a)
        self.a._pose_profile=load_profile();self.a._pose_trajectory=trajectory(self.anchor)
        self.client=Mock();self.client.failure.return_value=None
        # No live start: inject all telemetry and the model to test transport guards.
        with patch.object(backend.ZReplaySession,'__init__',return_value=None):
            self.s=backend.PoseReplaySession(self.client,self.anchor,self.a)
        s=self.s;s.client=self.client;s.anchor=self.anchor;s.args=self.a;s.trace=Mock()
        s.last_stream_ns=None;s.latest_guid=None;s.samples=[];s.initial_q=np.zeros(16)
        s.model=Mock();s.guard=Mock();s.telemetry=Mock();s.feedback_client=None
        s.telemetry.latest_joint=(self.clock.ns,{'q':np.zeros(16),'dq':np.zeros(16)})
        self.packet(self.anchor)
        s.engine=Mock();s.engine.step.side_effect=lambda p,r,dt,weight:(p+np.array([0,0,.02])*weight,r,{})
        self.clock.perf_counter=lambda:0.

    def packet(self,pose,source_progress=True):
        self.s.telemetry.packet=PosePacket('fresh',self.clock.ns-10_000_000,self.clock.ns,
                                          pose,123.,source_progress)

    def test_first_command_zero_ramp_and_measured_right_hold(self):
        self.s.stream('settle_pose')
        name,data=self.client.send.call_args.args
        self.assertEqual(name,'request_servop')
        np.testing.assert_allclose(data['left_pos'][:3],self.anchor.left_position)
        np.testing.assert_allclose(data['right_pos'],np.r_[self.anchor.right_position,self.anchor.right_quaternion])
        self.assertNotIn('head',data)
        self.assertEqual(self.s.engine.step.call_args.kwargs['weight'],0.)

    def test_right_arm_drift_is_relaxed_without_relaxing_head_guard(self):
        q=np.zeros(16);q[10]=.1
        backend.check_passive_joint_drift(q,np.zeros(16))
        q[10]=backend.RIGHT_ARM_PASSIVE_DRIFT_LIMIT_RAD+.001
        with self.assertRaisesRegex(RuntimeError,'右臂冗余关节偏移'):
            backend.check_passive_joint_drift(q,np.zeros(16))
        q=np.zeros(16);q[14]=backend.HEAD_PASSIVE_DRIFT_LIMIT_RAD+.001
        with self.assertRaisesRegex(RuntimeError,'头部出现非计划关节偏移'):
            backend.check_passive_joint_drift(q,np.zeros(16))

    def test_feedback_pairs_original_nominal_instead_of_preview(self):
        self.s.start_ns=self.clock.ns-500_000_000
        controller=Mock();controller.advance.return_value=(np.zeros(3),np.eye(3));self.s.pose_controller=controller
        self.s.stream('trajectory',.5)
        args=controller.observe.call_args.args
        # Request midpoint is t=.495. Sent command uses t=.624.
        np.testing.assert_allclose(args[4],self.anchor.left_position+[.00495,0,0])
        np.testing.assert_allclose(self.s.engine.step.call_args.args[0],self.anchor.left_position+[.00624,0,0])

    def test_explicit_measurement_age_changes_pairing(self):
        self.a.pose_feedback_age_ms=50.;self.s.start_ns=self.clock.ns-500_000_000
        c=Mock();c.advance.return_value=(np.zeros(3),np.eye(3));self.s.pose_controller=c
        self.s.stream('trajectory',.5)
        np.testing.assert_allclose(c.observe.call_args.args[4],self.anchor.left_position+[.00445,0,0])

    def test_unprogressed_source_cannot_accumulate_feedback(self):
        self.packet(self.anchor,False);c=Mock();c.advance.return_value=(np.zeros(3),np.eye(3));self.s.pose_controller=c
        self.s.stream('settle_pose');c.observe.assert_not_called();c.freeze.assert_called_once()

    def test_guards_abort_before_motion(self):
        for mode in ['stale','position','rotation','right','joint','gap','model']:
            with self.subTest(mode=mode):
                self.prepare_session()
                if mode=='stale':self.s.telemetry.require_fresh.side_effect=RuntimeError('stale')
                if mode=='position':self.packet(tracker.DualArmPose(self.anchor.left_position+[.051,0,0],self.anchor.left_quaternion,self.anchor.right_position,self.anchor.right_quaternion))
                if mode=='rotation':self.packet(tracker.DualArmPose(self.anchor.left_position,Rotation.from_euler('x',16,degrees=True).as_quat()[[3,0,1,2]],self.anchor.right_position,self.anchor.right_quaternion))
                if mode=='right':self.packet(tracker.DualArmPose(self.anchor.left_position,self.anchor.left_quaternion,self.anchor.right_position+[.011,0,0],self.anchor.right_quaternion))
                if mode=='joint':self.s.model.check_joints.side_effect=RuntimeError('limits')
                if mode=='gap':self.s.last_stream_ns=self.clock.ns-100_000_001
                if mode=='model':self.s.engine.step.side_effect=RuntimeError('IK')
                with self.assertRaises(RuntimeError):self.s.stream('trajectory')
                self.client.send.assert_not_called()

    def test_compute_overrun_holds_last_valid_target_and_continues(self):
        held=np.r_[self.anchor.left_position+[.001,0,0],self.anchor.left_quaternion]
        self.s.last_issued=held.copy()
        self.clock.perf_counter=Mock(side_effect=[0.,.021])
        self.s.stream('trajectory')
        name,data=self.client.send.call_args.args
        self.assertEqual(name,'request_servop')
        np.testing.assert_allclose(data['left_pos'],held)
        self.s.trace.record.assert_any_call('pose_replay_compute_overrun',phase='trajectory',
            calculation_ms=21.,action='hold_last_valid_target',held_left_pos=held.tolist())

    def test_settle_requires_fresh_full_pose_for_one_second_after_ramp(self):
        seen=[]
        def stream(phase):
            seen.append(self.clock.ns)
            self.s.telemetry.packet=PosePacket(str(self.clock.ns//50_000_000),
                self.clock.ns-10_000_000,self.clock.ns,self.anchor,123.,True)
        self.s.stream=stream
        self.s.settle_pose()
        self.assertGreaterEqual(self.clock.ns-seen[0],6_000_000_000)
        self.assertLess(self.clock.ns-seen[0],6_200_000_000)

    def test_good_position_with_wrong_orientation_cannot_pass_settle(self):
        pose=tracker.DualArmPose(self.anchor.left_position,
            Rotation.from_euler('x',2,degrees=True).as_quat()[[3,0,1,2]],
            self.anchor.right_position,self.anchor.right_quaternion)
        def stream(phase):
            self.s.telemetry.packet=PosePacket(str(self.clock.ns),self.clock.ns-10_000_000,
                self.clock.ns,pose,123.,True)
        self.s.stream=stream
        with self.assertRaisesRegex(RuntimeError,'45秒'):self.s.settle_pose()

    def test_stale_static_packet_cannot_pass_settle(self):
        self.s.stream=Mock()
        with self.assertRaisesRegex(RuntimeError,'45秒'):self.s.settle_pose()


class LifecycleTests(unittest.TestCase):
    def run_replay(self,settle_error=None,answer='REPLAY',stream_error=None):
        clock=Clock();anchor=anchor_pose();a=options();a.countdown=0
        a._pose_trajectory=trajectory(anchor);a._pose_profile=load_profile()
        events=[];client=Mock();hand=Mock();session=Mock();session.samples=[]
        session.start.side_effect=lambda:events.append('session_start')
        def settle(health):
            events.append('settle')
            if settle_error:raise settle_error
            clock.sleep(6.)
        session.settle_pose.side_effect=settle
        session.hold_until.side_effect=lambda start,health:clock.sleep((start-clock.ns)/1e9)
        def stream(*args):
            if stream_error:raise stream_error
        session.stream.side_effect=stream
        session.final_pose.return_value=anchor
        session.close.side_effect=lambda:events.append('session_close')
        hand.start_at.side_effect=lambda stamp:events.append(('hand_start',stamp,clock.ns))
        hand.close.side_effect=lambda:events.append('hand_close')
        try:
            with patch.object(backend,'time',clock),patch.object(backend,'PoseReplaySession',return_value=session),patch('builtins.input',return_value=answer),patch.object(backend,'verify_observed_motion',return_value=True):
                backend.execute_pose_compensated_replay(client,anchor,'left',np.array([0.,.01]),np.zeros((2,3)),np.tile([1.,0,0,0],(2,1)),a,hand_replay=hand)
        except RuntimeError:pass
        return events,hand,session,client

    def test_shared_clock_scheduled_only_after_settle(self):
        events,hand,session,_=self.run_replay()
        event=next(e for e in events if isinstance(e,tuple))
        self.assertEqual(event[1]-event[2],250_000_000)
        self.assertGreaterEqual(event[2],7_000_000_000)
        self.assertLess(events.index('settle'),events.index(event))

    def test_settle_failure_never_starts_hand_and_stops_hand_before_cleanup(self):
        events,hand,session,_=self.run_replay(settle_error=RuntimeError('settle failed'))
        hand.start_at.assert_not_called();session.stream.assert_not_called()
        self.assertLess(events.index('hand_close'),events.index('session_close'))

    def test_cancel_never_acquires_or_starts(self):
        _,hand,session,client=self.run_replay(answer='cancel')
        hand.acquire.assert_not_called();session.start.assert_not_called();client.send.assert_not_called()

    def test_stream_failure_stops_hand_before_cleanup(self):
        events,_,_,_=self.run_replay(stream_error=RuntimeError('stream failed'))
        self.assertLess(events.index('hand_close'),events.index('session_close'))

    def test_latest_flag_dispatches_only_latest_backend(self):
        with patch.object(backend,'execute_pose_compensated_replay',return_value=True) as new,patch('tron2_z_replay.execute_z_compensated_replay') as old:
            tracker.execute_replay(Mock(),anchor_pose(),'left',[],[],[],options())
            new.assert_called_once();old.assert_not_called()


if __name__=='__main__':unittest.main()
