"""Offline integration contract: fake clock, arm client, hand and Z session only."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

import replay_tron2_fixed_start as fixed
import replay_tron2_tracker as tracker
import tron2_z_replay as backend
from tools.diagnose_tron2_z_compensation import PosePacket
from tron2_z_compensation import Config


def anchor_pose():
    return tracker.DualArmPose(np.array([.45,.25,-.31]),np.array([1.,0.,0.,0.]),
                               np.array([.1,-.2,-.75]),np.array([1.,0.,0.,0.]))


def args(**kwargs):
    result=SimpleNamespace(arm_control='servop',countdown=0,feedback_timeout=.5,rate=300.,
                           z_compensation=True,z_max_correction_mm=20.,web_control=False,
                           settle_handover=False)
    result.__dict__.update(kwargs)
    return result


class Clock:
    def __init__(self):self.ns=1_000_000_000
    def monotonic_ns(self):return self.ns
    def monotonic(self):return self.ns/1e9
    def sleep(self,seconds):self.ns+=max(1,int(round(seconds*1e9))) if seconds>0 else 0


class FakeSession:
    """The fake owns a bias; the backend must give it uncompensated nominal poses."""
    def __init__(self,client,anchor,clock,events,hand=None):
        self.client,self.anchor,self.clock,self.events,self.hand=client,anchor,clock,events,hand
        self.offset=.012
        self.samples=[]
        self.nominal=[]
        self.start_error=None
        self.settle_error=None
        self.fail_phase=None
        self.last_pose=anchor

    def start(self):
        self.events.append(('session_start',self.clock.ns))
        if self.start_error is not None:
            raise self.start_error
        self.events.append(('session_ready',self.clock.ns))

    def stream(self,position,quaternion,phase,adaptive=False):
        self.events.append(('stream',phase))
        if self.fail_phase==phase:raise RuntimeError('fake stream failure')
        position=np.array(position,copy=True);quaternion=np.array(quaternion,copy=True)
        self.nominal.append((phase,position,quaternion,adaptive))
        issued=position.copy();issued[2]+=self.offset
        self.client.send('request_servop',tracker.servop_data(np.r_[issued,quaternion],
            np.r_[self.anchor.right_position,self.anchor.right_quaternion]))
        self.last_pose=tracker.DualArmPose(position,quaternion,
            self.anchor.right_position,self.anchor.right_quaternion)
        self.samples.append(tracker.PoseSample(self.clock.monotonic(),self.last_pose))

    def settle_handover(self, *, health_check=None):
        self.events.append(('settle_start',self.clock.ns))
        if self.settle_error is not None:
            raise self.settle_error
        if health_check is not None:health_check()
        self.stream(self.anchor.left_position,self.anchor.left_quaternion,'settle_handover',True)
        self.clock.sleep(2.)
        self.events.append(('settle_ready',self.clock.ns))

    def keep_holding_until(self,start_ns,*,health_check=None):
        while self.clock.ns<start_ns:
            if health_check is not None:health_check()
            self.stream(self.anchor.left_position,self.anchor.left_quaternion,'waiting_start',False)
            self.clock.sleep(min(1/300.,(start_ns-self.clock.ns)/1e9))

    def final_pose(self):return self.last_pose
    def close(self):self.events.append(('session_close',self.offset))


class DispatchAndCliTests(unittest.TestCase):
    def test_continuous_adaptation_requires_compensation_and_is_opt_in(self):
        for parser in (fixed.parse_args, tracker.parse_args):
            self.assertEqual(parser(['--episode', '91']).z_adaptation, 'stationary')
            options = parser(['--episode', '91', '--z-adaptation', 'continuous'])
            with self.assertRaisesRegex(ValueError, '必须显式配合'):
                tracker.execute_replay(Mock(), anchor_pose(), 'left', np.array([0., .1]),
                    np.zeros((2, 3)), np.tile([1., 0., 0., 0.], (2, 1)), options)
        options = fixed.parse_args(['--episode', '91', '--z-adaptation', 'continuous'])
        with self.assertRaisesRegex(ValueError, '必须显式配合'):
            fixed.validate_args(options)
        options.z_compensation = True
        fixed.validate_args(options)

    def test_new_flag_defaults_off_and_has_explicit_twenty_mm_cap(self):
        parsed=fixed.parse_args(['--episode','82'])
        self.assertFalse(parsed.z_compensation)
        self.assertEqual(parsed.z_max_correction_mm,20.)
        fixed.validate_args(parsed)

    def test_settle_handover_requires_explicit_position_compensation(self):
        for parser in (fixed.parse_args,tracker.parse_args):
            options=parser(['--episode','82','--settle-handover'])
            self.assertTrue(options.settle_handover)
            if parser is fixed.parse_args:
                with self.assertRaisesRegex(ValueError,'显式配合'):
                    fixed.validate_args(options)
            with self.assertRaisesRegex(ValueError,'显式配合'):
                tracker.execute_replay(Mock(),anchor_pose(),'left',np.array([0.,.1]),
                    np.zeros((2,3)),np.tile([1.,0.,0.,0.],(2,1)),options)
        options=fixed.parse_args(['--episode','82','--z-compensation','--settle-handover'])
        fixed.validate_args(options)
        parsed=fixed.parse_args(['--episode','82','--z-compensation'])
        self.assertTrue(parsed.z_compensation)
        fixed.validate_args(parsed)

    def test_invalid_caps_rate_or_plan_mode_are_rejected(self):
        cases=[['--z-max-correction-mm','10'],
               ['--z-compensation','--z-max-correction-mm','4.99'],
               ['--z-compensation','--z-max-correction-mm','20.01'],
               ['--z-compensation','--z-max-correction-mm','nan'],
               ['--z-compensation','--z-max-correction-mm','inf'],
               ['--z-compensation','--rate','200'],['--z-compensation','--plan-ik']]
        for options in cases:
            with self.subTest(options=options),self.assertRaises(ValueError):
                fixed.validate_args(fixed.parse_args(['--episode','82']+options))
        for cap in ['5','20']:
            fixed.validate_args(fixed.parse_args(['--episode','82','--z-compensation',
                                                  '--z-max-correction-mm',cap]))

    def test_opt_in_lazy_dispatch_forwards_exact_existing_contract(self):
        client,anchor,hand=Mock(),anchor_pose(),Mock()
        times=np.array([0.,.1]);positions=np.zeros((2,3));quats=np.tile([1.,0.,0.,0.],(2,1));options=args()
        with patch.object(backend,'execute_z_compensated_replay',return_value='sentinel') as execute:
            actual=tracker.execute_replay(client,anchor,'left',times,positions,quats,options,
                                         confirmation_word='REPLAY',hand_replay=hand)
        self.assertEqual(actual,'sentinel')
        execute.assert_called_once_with(client,anchor,'left',times,positions,quats,options,
                                         confirmation_word='REPLAY',hand_replay=hand)

    def test_no_flag_still_runs_when_z_backend_is_unavailable(self):
        clock=Clock();anchor=anchor_pose();client=Mock(trace=Mock())
        client.current_pose.return_value=anchor;client.current_servo_pose.return_value=anchor
        client.joint_snapshot.return_value=(np.zeros(14),np.zeros(14))
        client.failure.return_value=None;client.pose_samples.return_value=[]
        client.latest_pose_sample.side_effect=lambda:tracker.PoseSample(clock.monotonic(),anchor)
        options=SimpleNamespace(arm_control='servop',countdown=0,feedback_timeout=.5,rate=20.)
        positions=np.array([[0.,0.,0.],[.002,-.001,.003]])
        with patch.dict(sys.modules,{'tron2_z_replay':None}), \
                patch('builtins.input',return_value='DIRECT'),patch.object(tracker,'FeedbackGuard'), \
                patch.object(tracker,'POST_STOP_OBSERVE_SECONDS',0), \
                patch.object(tracker,'verify_observed_motion',return_value=True), \
                patch.object(tracker.time,'monotonic',side_effect=clock.monotonic), \
                patch.object(tracker.time,'monotonic_ns',side_effect=clock.monotonic_ns), \
                patch.object(tracker.time,'sleep',side_effect=clock.sleep),redirect_stdout(io.StringIO()):
            self.assertTrue(tracker.execute_replay(client,anchor,'left',np.array([0.,.01]),positions,
                    np.tile([1.,0.,0.,0.],(2,1)),options))
        np.testing.assert_allclose(client.send.call_args_list[0].args[1]['left_pos'][:3],anchor.left_position)
        np.testing.assert_allclose(client.send.call_args_list[-1].args[1]['left_pos'][:3],anchor.left_position+positions[-1])
        self.assertTrue(all(c.args[0]=='request_servop' for c in client.send.call_args_list))


class ReplayLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.anchor=anchor_pose();self.original=deepcopy(self.anchor)
        self.events=[];self.client=Mock(trace=Mock());self.client.failure.return_value=None
        self.hand=Mock()
        self.hand.acquire.side_effect=lambda:self.events.append(('hand_acquire',self.clock.ns))
        self.hand.start_at.side_effect=lambda stamp:self.events.append(('hand_start',stamp))
        self.hand.close.side_effect=lambda:self.events.append(('hand_close',self.clock.ns))
        self.session=FakeSession(self.client,self.anchor,self.clock,self.events,self.hand)
        self.times=np.array([0.,.05,.1]);self.deltas=np.array([[0.,0.,0.],[.002,-.001,.003],[.004,-.002,.006]])
        self.quats=np.tile([1.,0.,0.,0.],(3,1))

    def run_replay(self,**options):
        with patch.object(backend,'ZReplaySession',return_value=self.session) as prepared, \
                patch.object(backend,'verify_observed_motion',return_value=True) as verified, \
                patch('builtins.input',return_value='REPLAY'), \
                patch.object(backend.time,'monotonic',side_effect=self.clock.monotonic), \
                patch.object(backend.time,'monotonic_ns',side_effect=self.clock.monotonic_ns), \
                patch.object(backend.time,'sleep',side_effect=self.clock.sleep),redirect_stdout(io.StringIO()):
            value=tracker.execute_replay(self.client,self.anchor,'left',self.times,self.deltas,self.quats,
                args(**options),confirmation_word='REPLAY',hand_replay=self.hand)
        return value,prepared,verified

    def test_direct_first_frame_uses_same_future_integer_clock_without_warmup(self):
        value,prepared,verified=self.run_replay()
        self.assertTrue(value)
        self.hand.acquire.assert_called_once();self.hand.start_at.assert_called_once()
        stamp=self.hand.start_at.call_args.args[0]
        self.assertIsInstance(stamp,int)
        done=next(v for name,v in self.events if name=='session_ready')
        self.assertEqual(stamp,done+250_000_000)
        order=[event[0] for event in self.events]
        self.assertLess(order.index('hand_acquire'),order.index('session_start'))
        self.assertLess(order.index('session_ready'),order.index('hand_start'))
        starts=[c.kwargs['start_monotonic_ns'] for c in self.client.trace.record.call_args_list
                if c.args[0]=='replay_start']
        self.assertEqual(starts,[stamp])
        first=next(v for name,v in self.events if name=='session_start')
        self.assertEqual(done,first)  # no preheat delay
        self.assertEqual(self.session.nominal[0][0],'trajectory')
        self.assertFalse(hasattr(backend.ZReplaySession,'warmup'))
        self.assertTrue(all(s.received_at>=stamp/1e9 for s in verified.call_args.args[4]))

    def test_nominal_anchor_unchanged_and_bias_owned_once_by_stream(self):
        self.run_replay()
        trajectory=[r for r in self.session.nominal if r[0]=='trajectory']
        for i,(_,position,quaternion,_) in enumerate(trajectory):
            np.testing.assert_allclose(position,self.original.left_position+self.deltas[i])
            np.testing.assert_array_equal(quaternion,self.original.left_quaternion)
        self.assertFalse(any(r[0]=='waiting_start' for r in self.session.nominal))
        right=np.r_[self.original.right_position,self.original.right_quaternion]
        for call,(_,nominal,_,_) in zip(self.client.send.call_args_list,self.session.nominal):
            np.testing.assert_allclose(call.args[1]['left_pos'][:3],nominal+[0.,0.,.012])
            np.testing.assert_array_equal(call.args[1]['right_pos'],right)
        np.testing.assert_array_equal(self.anchor.left_position,self.original.left_position)
        np.testing.assert_array_equal(self.anchor.left_quaternion,self.original.left_quaternion)
        np.testing.assert_allclose(self.client.send.call_args_list[-1].args[1]['left_pos'][:3],
                                   self.original.left_position+self.deltas[-1]+[0.,0.,.012])
        self.assertEqual(self.events[-1],('session_close',.012))

    def test_continuous_mode_updates_during_motion_without_changing_nominal_or_hand_clock(self):
        self.run_replay(z_adaptation='continuous', settle_handover=True)
        trajectory = [r for r in self.session.nominal if r[0] == 'trajectory']
        self.assertTrue(all(r[3] for r in trajectory))
        for i, (_, position, _, _) in enumerate(trajectory):
            np.testing.assert_allclose(position, self.original.left_position+self.deltas[i])
        contract = next(c.kwargs for c in self.client.trace.record.call_args_list
                        if c.args[0] == 'replay_contract')
        self.assertEqual(contract['z_adaptation'], 'continuous')
        self.assertFalse(contract['torque_continuity_repaired'])
        start = next(c.kwargs['start_monotonic_ns'] for c in self.client.trace.record.call_args_list
                     if c.args[0] == 'replay_start')
        self.assertEqual(start, self.hand.start_at.call_args.args[0])

    def test_no_interpolation_contract_keeps_o6_and_arm_start_equal(self):
        self.assertTrue(self.run_replay(interpolation='none', settle_handover=True)[0])
        contract=next(c.kwargs for c in self.client.trace.record.call_args_list
                      if c.args[0]=='replay_contract')
        self.assertEqual(contract['tracker_interpolation'],'none')
        start=next(c.kwargs['start_monotonic_ns'] for c in self.client.trace.record.call_args_list
                   if c.args[0]=='replay_start')
        self.assertEqual(start,self.hand.start_at.call_args.args[0])

    def test_settle_success_precedes_shared_clock_and_keeps_bias_during_wait(self):
        sent_times=[]
        self.client.send.side_effect=lambda *a,**k:sent_times.append(self.clock.ns)
        value,_,verified=self.run_replay(settle_handover=True)
        self.assertTrue(value)
        ready=next(v for name,v in self.events if name=='settle_ready')
        start=self.hand.start_at.call_args.args[0]
        self.assertEqual(start,ready+250_000_000)
        self.assertTrue(any(row[0]=='waiting_start' for row in self.session.nominal))
        self.assertLessEqual(max(np.diff([t for t in sent_times if ready<=t<=start])),4_000_000)
        order=[r[0] for r in self.events]
        self.assertLess(order.index('settle_ready'),order.index('hand_start'))
        self.assertTrue(all(s.received_at>=start/1e9 for s in verified.call_args.args[4]))
        for call,row in zip(self.client.send.call_args_list,self.session.nominal):
            np.testing.assert_allclose(call.args[1]['left_pos'][:3],row[1]+[0.,0.,.012])
        np.testing.assert_array_equal(self.anchor.left_position,self.original.left_position)

    def test_settle_timeout_never_schedules_hand_or_trajectory(self):
        self.session.settle_error=RuntimeError('恢复超时')
        with self.assertRaisesRegex(RuntimeError,'恢复超时'):
            self.run_replay(settle_handover=True)
        self.hand.start_at.assert_not_called()
        self.assertFalse(any(row[0]=='trajectory' for row in self.session.nominal))
        self.hand.close.assert_called_once()
        self.assertEqual(self.events[-1],('session_close',.012))

    def test_missing_first_feedback_does_not_start_fingers_and_stops_hand_first(self):
        self.session.start_error=RuntimeError('first feedback failed')
        with self.assertRaisesRegex(RuntimeError,'first feedback failed'):self.run_replay()
        self.hand.start_at.assert_not_called();self.client.send.assert_not_called()
        self.hand.close.assert_called_once()
        order=[e[0] for e in self.events]
        self.assertLess(order.index('hand_close'),order.index('session_close'))
        self.assertEqual(self.session.offset,.012)

    def test_after_scheduling_failure_stops_hand_first_and_never_clears_bias(self):
        original_stream=self.session.stream
        def fail_after_first(position,quaternion,phase,adaptive=False):
            if self.session.nominal:
                raise RuntimeError('fake stream failure')
            original_stream(position,quaternion,phase,adaptive)
        self.session.stream=fail_after_first
        with self.assertRaisesRegex(RuntimeError,'fake stream failure'):self.run_replay()
        self.hand.start_at.assert_called_once();self.hand.close.assert_called_once()
        order=[e[0] for e in self.events]
        self.assertLess(order.index('hand_close'),order.index('session_close'))
        self.assertEqual(self.session.offset,.012)
        self.assertTrue(self.client.send.call_args_list)
        self.assertAlmostEqual(self.client.send.call_args_list[-1].args[1]['left_pos'][2],
                               self.anchor.left_position[2]+.012)
        self.client.current_servo_pose.assert_not_called()

    def test_final_hold_refuses_catch_up_after_two_hundred_ms_delay(self):
        original_stream=self.session.stream

        def slow_hold(position,quaternion,phase,adaptive=False):
            original_stream(position,quaternion,phase,adaptive)
            if phase=='final_hold':self.clock.sleep(.2)

        self.session.stream=slow_hold
        with self.assertRaisesRegex(RuntimeError,'末帧保持发送落后超过100ms'):
            self.run_replay()
        self.assertEqual(sum(row[0]=='final_hold' for row in self.session.nominal),1)
        self.hand.close.assert_called_once()
        order=[e[0] for e in self.events]
        self.assertLess(order.index('hand_close'),order.index('session_close'))
        self.assertEqual(self.session.offset,.012)
        self.assertNotIn('z_replay_sending_stopped',
                         [c.args[0] for c in self.client.trace.record.call_args_list])

    def test_right_arm_request_is_rejected_before_preparation_or_o6_acquire(self):
        with patch.object(backend,'ZReplaySession') as prepare, \
                self.assertRaisesRegex(ValueError,'左臂ServoP'):
            backend.execute_z_compensated_replay(self.client,self.anchor,'right',self.times,
                self.deltas,self.quats,args(),hand_replay=self.hand)
        prepare.assert_not_called();self.hand.acquire.assert_not_called()


class SessionAndQuietTests(unittest.TestCase):
    def setUp(self):
        self.anchor=anchor_pose();self.client=Mock(trace=Mock(),accid='DACH_TRON2A_253');self.client.failure.return_value=None
        self.model=Mock()
        self.reader=Mock(accid=self.client.accid)
        self.reader_patch=patch.object(backend,'RawWSClient',return_value=self.reader)
        self.reader_patch.start();self.addCleanup(self.reader_patch.stop)

    def new_session(self):
        with patch.object(backend,'strict_model',return_value=self.model):
            return backend.ZReplaySession(self.client,self.anchor,args())

    def telemetry(self, q=None, dq=None, pose=None):
        pose=self.anchor if pose is None else pose
        packet=PosePacket('first',998_000_000,999_000_000,pose,1000.,True)
        return SimpleNamespace(packet=packet,
            latest_joint=(999_000_000,{'q':np.zeros(16) if q is None else q,
                                     'dq':np.zeros(16) if dq is None else dq}),
            error=None,thread=Mock(),require_fresh=Mock())

    def test_start_only_reads_feedback_without_planning_preheat_or_payload_queries(self):
        session=self.new_session()
        telemetry=self.telemetry()
        with patch.object(backend,'CompensationTelemetry',return_value=telemetry), \
                patch.dict(sys.modules,{'tron2_z_envelope':None}), \
                patch.object(backend.time,'monotonic',return_value=1.):
            session.start()
        self.client.send.assert_not_called()
        self.client.request.assert_not_called()
        self.assertEqual(session.controller.applied_offset,0.)
        self.assertEqual(session.controller.active_limit,.02)
        np.testing.assert_array_equal(session.initial_q,np.zeros(16))
        self.assertFalse(hasattr(session,'warmup'))
        self.assertEqual(self.client.trace.record.call_args.args[0],'z_replay_direct_start')
        self.reader.wait_for_identity.assert_called_once()
        self.assertIs(session.feedback_client,self.reader)

    def test_feedback_identity_mismatch_prevents_first_motion(self):
        self.reader.accid='another_robot'
        session=self.new_session()
        with self.assertRaisesRegex(RuntimeError,'身份不一致'):
            session.start()
        self.client.send.assert_not_called()
        session.close()
        self.reader.close.assert_called_once()

    def test_first_live_joint_motion_or_invalid_values_block_first_servo(self):
        for q,dq in [(np.zeros(16),np.r_[.050001,np.zeros(15)]),
                     (np.r_[float('nan'),np.zeros(15)],np.zeros(16)),
                     (np.zeros(16),np.r_[float('nan'),np.zeros(15)]),
                     (np.zeros(14),np.zeros(16)),(np.zeros(16),np.zeros(15))]:
            session=self.new_session()
            with self.subTest(q=q,dq=dq), \
                    patch.object(backend,'CompensationTelemetry',return_value=self.telemetry(q,dq)), \
                    self.assertRaisesRegex(RuntimeError,'首帧前关节反馈无效'):
                session.start()
            self.client.send.assert_not_called()

    def test_first_pose_drift_stops_without_reanchoring_or_sending(self):
        session=self.new_session()
        changed=deepcopy(self.anchor);changed.left_position[2]-=.004
        with patch.object(backend,'CompensationTelemetry',return_value=self.telemetry(pose=changed)), \
                self.assertRaisesRegex(RuntimeError,'倒计时期间变化'):
            session.start()
        self.client.send.assert_not_called()
        np.testing.assert_array_equal(session.anchor.left_position,self.anchor.left_position)

    def test_missing_feedback_times_out_without_a_servo_command(self):
        session=self.new_session()
        telemetry=SimpleNamespace(packet=None,latest_joint=None,error=None,thread=Mock())
        clock=Clock()
        with patch.object(backend,'CompensationTelemetry',return_value=telemetry), \
                patch.object(backend.time,'monotonic',side_effect=clock.monotonic), \
                patch.object(backend.time,'sleep',side_effect=clock.sleep), \
                self.assertRaisesRegex(RuntimeError,'未收到完整遥测'):
            session.start()
        self.client.send.assert_not_called()

    def test_real_session_adds_bias_once_and_retains_original_right_target(self):
        session=self.new_session();session.initial_q=np.zeros(16);session.guard=Mock()
        nominal=self.anchor.left_position+[.002,-.003,.02]
        actual=tracker.DualArmPose(nominal,self.anchor.left_quaternion,
                                   self.anchor.right_position+[.004,0.,0.],self.anchor.right_quaternion)
        packet=PosePacket('fresh',998_000_000,999_000_000,actual,1000.,True)
        session.telemetry=SimpleNamespace(packet=packet,latest_joint=(999_000_000,{'q':np.zeros(16)}),
            require_fresh=Mock(),stop=Mock(),thread=Mock())
        session.controller=Mock(applied_offset=.012,status='tracking')
        session.controller.advance.return_value=.012
        with patch.object(backend.time,'monotonic_ns',return_value=1_000_000_000):
            session.stream(nominal,self.anchor.left_quaternion,'trajectory',adaptive=False)
        call=self.client.send.call_args
        np.testing.assert_allclose(call.args[1]['left_pos'][:3],nominal+[0.,0.,.012])
        np.testing.assert_array_equal(call.args[1]['right_pos'],
            np.r_[self.anchor.right_position,self.anchor.right_quaternion])
        np.testing.assert_allclose(nominal,self.anchor.left_position+[.002,-.003,.02])
        before=self.client.send.call_count;session.close()
        self.assertEqual(self.client.send.call_count,before)
        self.assertFalse(self.client.trace.record.call_args.kwargs['correction_cleared'])

    def test_no_interpolation_feedback_reference_uses_previous_target_at_query_midpoint(self):
        session=self.new_session();session.args.interpolation='none'
        session.initial_q=np.zeros(16);session.guard=Mock()
        old_z=self.anchor.left_position[2]
        session.history.append((980_000_000,old_z,True))
        packet=PosePacket('fresh',988_000_000,992_000_000,self.anchor,1000.,True)
        session.telemetry=SimpleNamespace(packet=packet,
            latest_joint=(992_000_000,{'q':np.zeros(16)}),require_fresh=Mock())
        session.controller=Mock(applied_offset=0.,status='frozen')
        session.controller.advance.return_value=0.
        new_position=self.anchor.left_position+[0.,0.,.02]
        with patch.object(backend.time,'monotonic_ns',return_value=1_000_000_000):
            session.stream(new_position,self.anchor.left_quaternion,'trajectory',adaptive=False)
        # The query midpoint predates the new target; it must not see a made-up halfway pose.
        self.assertEqual(session.controller.observe.call_args.args[4],old_z)

    def test_two_hundred_ms_send_block_stops_immediately_and_preserves_bias(self):
        session=self.new_session();session.initial_q=np.zeros(16);session.guard=Mock()
        clock=Clock()
        packet=PosePacket('fresh',998_000_000,999_000_000,self.anchor,1000.,True)
        session.telemetry=SimpleNamespace(packet=packet,latest_joint=(999_000_000,{'q':np.zeros(16)}),
            require_fresh=Mock(),stop=Mock(),thread=Mock())
        session.controller=Mock(applied_offset=.012,status='tracking')
        session.controller.advance.return_value=.012
        self.client.send.side_effect=lambda *a,**k:clock.sleep(.2)
        with patch.object(backend.time,'monotonic_ns',side_effect=clock.monotonic_ns):
            with self.assertRaisesRegex(RuntimeError,'发送阻塞超过100ms'):
                session.stream(self.anchor.left_position,self.anchor.left_quaternion,'final_hold')
            self.assertEqual(session.last_stream_ns,1_000_000_000)
            with self.assertRaisesRegex(RuntimeError,'发送间隔超过100ms'):
                session.stream(self.anchor.left_position,self.anchor.left_quaternion,'final_hold')
        self.client.send.assert_called_once()
        self.assertAlmostEqual(self.client.send.call_args.args[1]['left_pos'][2],
                               self.anchor.left_position[2]+.012)
        session.close()
        self.assertEqual(session.controller.applied_offset,.012)

    def test_quiet_window_rejects_fast_out_and_back_translation(self):
        times=np.array([0.,.05,.1]);positions=np.array([[0.,0.,0.],[.002,0.,0.],[0.,0.,0.]])
        quats=np.tile([1.,0.,0.,0.],(3,1))
        self.assertFalse(backend.nominal_quiet_flags(times,positions,quats).any())
        self.assertTrue(backend.nominal_quiet_flags(times,np.zeros((3,3)),quats).all())

    def test_quiet_window_rejects_fast_out_and_back_rotation(self):
        times=np.array([0.,.05,.1]);quats=Rotation.from_euler('z',[0.,1.,0.],degrees=True).as_quat()[:,[3,0,1,2]]
        self.assertFalse(backend.nominal_quiet_flags(times,np.zeros((3,3)),quats).any())


class HandoverRecoveryTests(unittest.TestCase):
    def session(self,clock,error_source,*,progress=True):
        anchor=anchor_pose();client=Mock(trace=Mock(),accid='DACH_TRON2A_253')
        client.failure.return_value=None
        with patch.object(backend,'strict_model',return_value=Mock()):
            session=backend.ZReplaySession(client,anchor,args())
        session.initial_q=np.zeros(16)
        class Feedback:
            packet=None
            latest_joint=None
            last_packet_ns=None
            def require_fresh(self):
                now=clock.ns
                if self.last_packet_ns is not None and now-self.last_packet_ns<50_000_000:
                    return
                self.last_packet_ns=now
                actual=deepcopy(anchor)
                actual.left_position[2]-=error_source(session,now)
                self.packet=PosePacket(str(now),now-2_000_000,now-1_000_000,
                                       actual,now/1e6,progress)
                self.latest_joint=(now-1_000_000,{'q':np.zeros(16)})
        session.telemetry=Feedback()
        return session,client

    def clock_patches(self,clock):
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch.object(backend.time,'monotonic_ns',side_effect=clock.monotonic_ns))
        stack.enter_context(patch.object(backend.time,'monotonic',side_effect=clock.monotonic))
        stack.enter_context(patch.object(backend.time,'sleep',side_effect=clock.sleep))
        return stack

    def test_online_bias_recovery_converges_then_retains_same_controller_output(self):
        clock=Clock();session,client=self.session(clock,lambda s,n:.011-s.controller.applied_offset)
        original=deepcopy(session.anchor)
        with self.clock_patches(clock):
            session.settle_handover()
            retained=session.controller.applied_offset
            ready_at=clock.ns
            self.assertLessEqual(abs(.011-retained),.001)
            self.assertGreater(clock.ns,11_000_000_000)  # 1mm/s bound prevents an instantaneous jump.
            self.assertEqual(session.controller.target_offset,retained)
            session.keep_holding_until(clock.ns+250_000_000)
            self.assertEqual(session.controller.applied_offset,retained)
            self.assertGreaterEqual(clock.ns,ready_at+250_000_000)
            session.stream(original.left_position+[0.,0.,.003],original.left_quaternion,'trajectory')
        self.assertAlmostEqual(client.send.call_args.args[1]['left_pos'][2],original.left_position[2]+.003+retained)
        np.testing.assert_array_equal(session.anchor.left_position,original.left_position)
        self.assertTrue(all(c.args[0]=='request_servop' for c in client.send.call_args_list))
        self.assertTrue(any(c.args[0]=='z_handover_ready' for c in client.trace.record.call_args_list))

    def test_failed_convergence_respects_hard_cap_and_aborts_without_reanchoring(self):
        clock=Clock();session,client=self.session(clock,lambda s,n:.03)
        with self.clock_patches(clock),self.assertRaisesRegex(RuntimeError,'未开始Tracker/O6轨迹'):
            session.settle_handover(timeout_s=.3)
        self.assertLessEqual(session.controller.applied_offset,.02)
        self.assertTrue(all(c.kwargs.get('phase')!='trajectory' for c in client.trace.record.call_args_list))
        np.testing.assert_array_equal(session.anchor.left_position,anchor_pose().left_position)

    def test_nonprogressing_sensor_frames_cannot_satisfy_stability(self):
        clock=Clock();session,_=self.session(clock,lambda s,n:0.,progress=False)
        with self.clock_patches(clock),self.assertRaisesRegex(RuntimeError,'未开始Tracker/O6轨迹'):
            session.settle_handover(timeout_s=.3,stable_s=.1)
        self.assertEqual(session.controller.applied_offset,0.)

    def test_one_bad_sample_restarts_the_one_second_stability_window(self):
        clock=Clock();initial=clock.ns
        def error(session,now):
            elapsed=(now-initial)/1e9
            return .003 if .55<=elapsed<=.7 else 0.
        session,_=self.session(clock,error)
        with self.clock_patches(clock):session.settle_handover(timeout_s=3.)
        self.assertGreater(clock.ns-initial,1_650_000_000)


if __name__=='__main__':unittest.main()
