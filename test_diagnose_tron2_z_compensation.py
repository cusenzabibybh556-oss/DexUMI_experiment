"""Offline adapter tests: no SDK initialization, robot, Tracker or network."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from replay_tron2_tracker import DualArmPose
from tools import diagnose_tron2_z_compensation as diagnostic
from tron2_z_compensation import Config


def pose():
    return DualArmPose(np.array([.4, .2, -.3]), np.array([1., 0., 0., 0.]),
                       np.array([.3, -.4, -.35]), np.array([1., 0., 0., 0.]))


def pose_reply(stamp=1000., guid='pose-guid'):
    anchor=pose()
    return {'guid':guid, 'data':{'result':'success', 'timestamp':stamp,
        'left_position':anchor.left_position.tolist(), 'left_quat':anchor.left_quaternion.tolist(),
        'right_position':anchor.right_position.tolist(), 'right_quat':anchor.right_quaternion.tolist()}}


def payload_reply(mass=.396):
    return {'data':{'result':'success','data':[mass,0.,0.,0.,.001,0.,0.,0.]}}


class SourceTimestampTests(unittest.TestCase):
    def setUp(self):
        self.client=Mock()
        self.telemetry=diagnostic.CompensationTelemetry(self.client,Mock())

    def query(self,event,stamp,received,guid='new-guid'):
        reply=pose_reply(stamp,guid) if event=='diagnostic_pose' else {
            'guid':guid,'data':{'result':'success','timestamp':stamp,'q':[0.]*16}}
        with patch.object(diagnostic.Telemetry,'query',return_value=(received,reply)), \
                patch.object(diagnostic.time,'monotonic_ns',return_value=received-1_000_000):
            return self.telemetry.query('request','response',event)

    def test_new_guid_with_unchanged_source_stamp_is_not_new_measurement(self):
        self.query('diagnostic_pose',1000.,1_000_000_000,'first')
        self.assertTrue(self.telemetry.packet.source_progress)
        self.query('diagnostic_pose',1000.,1_050_000_000,'different-guid')
        packet=self.telemetry.packet
        self.assertEqual(packet.guid,'different-guid')
        self.assertEqual(packet.source_timestamp_ms,1000.)
        self.assertFalse(packet.source_progress)
        self.assertEqual(self.telemetry.source_clocks['diagnostic_pose'][1],1_000_000_000)
        self.client.send.assert_not_called()

    def test_retry_recovery_is_not_misclassified_as_successful_packet_rtt(self):
        reply=pose_reply(1001.,'retry-success')
        self.client.request.side_effect=[TimeoutError('first attempt'),reply]
        ticks=iter([0,1_000_000,110_000_000,150_000_000])
        with patch.object(diagnostic.time,'monotonic_ns',side_effect=lambda:next(ticks)):
            received,actual=self.telemetry.query('request','response','diagnostic_pose')
        self.assertIs(actual,reply);self.assertEqual(received,150_000_000)
        self.assertEqual(self.telemetry.packet.sent_ns,110_000_000)
        self.assertEqual(self.telemetry.packet.received_ns,150_000_000)
        recovered=[c for c in self.telemetry.trace.record.call_args_list
                   if c.args[0]=='telemetry_query_recovered']
        self.assertEqual(len(recovered),1)
        self.assertEqual(recovered[0].kwargs['attempts'],2)
        self.assertEqual(recovered[0].kwargs['successful_rtt_ms'],40.)

    def test_frozen_pose_allows_half_second_but_fails_after_it(self):
        self.query('diagnostic_pose',1000.,1_000_000_000,'first')
        self.query('diagnostic_pose',1000.,1_500_000_000,'at-boundary')
        self.assertFalse(self.telemetry.packet.source_progress)
        with self.assertRaisesRegex(RuntimeError,'0.5秒未推进'):
            self.query('diagnostic_pose',1000.,1_500_000_001,'past-boundary')

    def test_progressing_timestamp_restarts_only_its_own_stall_timer(self):
        self.query('diagnostic_pose',1000.,1_000_000_000,'first')
        self.query('diagnostic_pose',1001.,1_400_000_000,'progress')
        self.assertTrue(self.telemetry.packet.source_progress)
        self.query('diagnostic_pose',1001.,1_800_000_000,'same-again')
        self.assertFalse(self.telemetry.packet.source_progress)
        with self.assertRaisesRegex(RuntimeError,'0.5秒未推进'):
            self.query('diagnostic_pose',1001.,1_900_000_001,'stalled')

    def test_pose_and_joint_source_clocks_are_not_compared_to_each_other(self):
        self.query('diagnostic_pose',900000.,1_000_000_000,'p1')
        self.query('diagnostic_joint',5.,1_050_000_000,'j1')
        self.query('diagnostic_pose',900001.,1_100_000_000,'p2')
        self.query('diagnostic_joint',6.,1_150_000_000,'j2')
        self.assertEqual(self.telemetry.source_clocks['diagnostic_pose'],(900001.,1_100_000_000))
        self.assertEqual(self.telemetry.source_clocks['diagnostic_joint'],(6.,1_150_000_000))

    def test_other_topic_progress_cannot_hide_a_stalled_source(self):
        for frozen,progressing in [('diagnostic_pose','diagnostic_joint'),
                                  ('diagnostic_joint','diagnostic_pose')]:
            with self.subTest(frozen=frozen):
                self.telemetry=diagnostic.CompensationTelemetry(self.client,Mock())
                self.query(frozen,1000.,1_000_000_000,'f1')
                self.query(progressing,1.,1_400_000_000,'p1')
                self.query(progressing,2.,1_500_000_000,'p2')
                with self.assertRaisesRegex(RuntimeError,'0.5秒未推进'):
                    self.query(frozen,1000.,1_500_000_001,'f2')

    def test_backward_source_stamp_is_rejected_per_topic(self):
        for event in ['diagnostic_pose','diagnostic_joint']:
            with self.subTest(event=event):
                self.telemetry=diagnostic.CompensationTelemetry(self.client,Mock())
                self.query(event,1000.,1_000_000_000)
                with self.assertRaisesRegex(RuntimeError,'时间戳倒退'):
                    self.query(event,999.,1_050_000_000,'backward')

    def test_missing_nonfinite_boolean_or_text_timestamp_is_rejected(self):
        for event in ['diagnostic_pose','diagnostic_joint']:
            for bad in [None,float('nan'),float('inf'),True,'1000']:
                with self.subTest(event=event,stamp=bad), \
                        self.assertRaisesRegex(RuntimeError,'缺少有限时间戳'):
                    self.query(event,bad,1_000_000_000)
        reply=pose_reply()
        del reply['data']['timestamp']
        with patch.object(diagnostic.Telemetry,'query',return_value=(1_000_000_000,reply)), \
                patch.object(diagnostic.time,'monotonic_ns',return_value=999_000_000), \
                self.assertRaisesRegex(RuntimeError,'缺少有限时间戳'):
            self.telemetry.query('request','response','diagnostic_pose')

    def test_control_loop_queries_only_required_pose_and_joint(self):
        requested=[]
        def query(request,response,event):
            requested.append(request)
            if event=='diagnostic_pose':
                return 1_000_000_000,pose_reply()
            self.telemetry.stop.set()
            return 1_001_000_000,{'data':{'q':[0.]*16,'dq':[0.]*16}}
        with patch.object(self.telemetry,'query',side_effect=query):
            self.telemetry.run()
        self.assertEqual(requested,['request_get_servo_pose','request_get_joint_state'])
        self.assertIsNone(self.telemetry.error)
        self.assertEqual(self.telemetry.latest_joint[0],1_001_000_000)
        np.testing.assert_allclose(self.telemetry.latest[1].left_position,pose().left_position)

    def test_failed_critical_query_stops_loop_with_error(self):
        with patch.object(self.telemetry,'query',side_effect=TimeoutError('critical timeout')):
            self.telemetry.run()
        self.assertEqual(self.telemetry.error,'critical timeout')
        self.assertIsNone(self.telemetry.latest)


class TrialAdapterTests(unittest.TestCase):
    def test_frozen_source_packet_disables_integrator_even_during_adaptive_phase(self):
        anchor=pose()
        packet=diagnostic.PosePacket('new-guid',1_005_000_000,1_010_000_000,anchor,1000.,False)
        telemetry=SimpleNamespace(packet=packet,latest_joint=(1_010_000_000,{'q':[0.]*16}),
                                  require_fresh=Mock())
        controller=Mock(active_limit=.005,applied_offset=.002,target_offset=.002,status='disabled')
        controller.observe.return_value=False
        controller.advance.return_value=.002
        model=Mock(lower=np.full(16,-np.pi),upper=np.full(16,np.pi))
        client=Mock();client.failure.return_value=None
        trace=Mock()
        trial=diagnostic.Trial(client,telemetry,trace,model,anchor,np.zeros(16),controller)
        with patch.object(diagnostic.time,'monotonic_ns',side_effect=[
                1_000_000_000,1_010_000_000,1_020_000_000,1_120_000_000,1_120_000_000]):
            rows,settled=trial.segment('adaptive',.1,lambda t:0.,True)
        self.assertFalse(settled)
        self.assertFalse(controller.observe.call_args.args[-1])
        self.assertEqual(len(rows),1)
        self.assertFalse(rows[0]['quality_ok'])
        self.assertFalse(rows[0]['integrator_accepted'])
        # Freezing correction retains the already applied offset; it does not jump to zero.
        client.send.assert_called_once()
        self.assertAlmostEqual(client.send.call_args.args[1]['left_pos'][2],-.298)

    def test_joint_value_and_recorded_stamp_come_from_one_snapshot(self):
        anchor=pose()
        joint={'q':[0.]*16}

        class ChangingTelemetry:
            packet=diagnostic.PosePacket('new',1_005_000_000,1_010_000_000,anchor,1000.,True)
            reads=0
            require_fresh=Mock(side_effect=[(anchor,joint),RuntimeError('end fake stream')])

            @property
            def latest_joint(self):
                self.reads+=1
                # Simulate the worker publishing a newer tuple between separate reads.
                return (self.reads, joint)

        telemetry=ChangingTelemetry()
        controller=Mock(active_limit=.005,applied_offset=0.,target_offset=0.,status='primed')
        controller.observe.return_value=True;controller.advance.return_value=0.
        model=Mock(lower=np.full(16,-np.pi),upper=np.full(16,np.pi))
        client=Mock();client.failure.return_value=None
        trial=diagnostic.Trial(client,telemetry,Mock(),model,anchor,np.zeros(16),controller)
        with patch.object(diagnostic.time,'monotonic_ns',side_effect=[
                1_000_000_000,1_010_000_000,1_020_000_000,1_030_000_000]), \
                self.assertRaisesRegex(RuntimeError,'end fake stream'):
            trial.segment('adaptive',1.,lambda t:0.,True)
        self.assertEqual(telemetry.reads,1)
        self.assertEqual(trial.last_joint,1)
        model.check_joints.assert_called_once()
        np.testing.assert_array_equal(model.check_joints.call_args.args[0],joint['q'])


class TailAcceptanceTests(unittest.TestCase):
    @staticmethod
    def rows(start,count,*,quality=True,value=1.):
        return [{'received_ns':start+i*50_000_000,'quality_ok':quality,'actual_z':value}
                for i in range(count)]

    def test_old_samples_cannot_make_up_insufficient_recent_tail(self):
        rows=self.rows(0,20)+self.rows(3_000_000_000,9)
        with self.assertRaisesRegex(RuntimeError,'阶段末有效反馈不足'):
            diagnostic.mean_tail(rows,'actual_z')

    def test_low_quality_recent_packets_cannot_make_up_tail_quorum(self):
        rows=self.rows(0,10,quality=False)+self.rows(500_000_000,9)
        with self.assertRaisesRegex(RuntimeError,'阶段末有效反馈不足'):
            diagnostic.mean_tail(rows,'actual_z')

    def test_tail_mean_excludes_old_and_low_quality_outliers(self):
        rows=self.rows(0,10,value=100.)+self.rows(2_000_000_000,10,value=-.3)
        rows.insert(-1,{'received_ns':2_420_000_000,'quality_ok':False,'actual_z':1000.})
        self.assertAlmostEqual(diagnostic.mean_tail(rows,'actual_z'),-.3)


class EnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.anchor=pose()
        self.snapshot={'samples':[{'joint_reply':{'data':{'q':[0.]*16}}}]}
        self.model=Mock(lower=np.full(16,-np.pi),upper=np.full(16,np.pi))
        self.model.plan_tracking.side_effect=lambda q,p,u,t:(np.zeros((len(t),16)),{'frames':len(t)})
        self.prepared=(self.model,self.anchor,None,None,None,None,{'height_m':.02})

    def test_planner_receives_full_compensation_plus_lift_envelope(self):
        with patch.object(diagnostic,'prepare',return_value=self.prepared), \
                patch.object(diagnostic,'_check_path',return_value=None):
            _,_,report=diagnostic.prepare_envelope(self.snapshot,.02,.02)
        _,positions,orientations,times=self.model.plan_tracking.call_args.args
        np.testing.assert_allclose(positions[0],self.anchor.left_position)
        np.testing.assert_allclose(positions[-1],self.anchor.left_position+[0.,0.,.04])
        np.testing.assert_allclose(positions[:,:2],np.tile(self.anchor.left_position[:2],(len(times),1)))
        np.testing.assert_allclose(orientations,np.tile(self.anchor.left_quaternion,(len(times),1)))
        self.assertEqual(report['maximum_command_raise_m'],.04)
        np.testing.assert_array_equal(self.anchor.left_position,[.4,.2,-.3])

    def test_unreachable_upper_part_cannot_pass_using_only_nominal_lift(self):
        def limited_planner(q,positions,rotations,times):
            if np.max(positions[:,2]) > self.anchor.left_position[2]+.03:
                raise RuntimeError('upper envelope unreachable')
            return np.zeros((len(times),16)),{}
        self.model.plan_tracking.side_effect=limited_planner
        with patch.object(diagnostic,'prepare',return_value=self.prepared), \
                patch.object(diagnostic,'_check_path',return_value=None), \
                self.assertRaisesRegex(RuntimeError,'upper envelope unreachable'):
            diagnostic.prepare_envelope(self.snapshot,.02,.02)


class FinalConfigurationTests(unittest.TestCase):
    def test_final_payload_failure_turns_completed_motion_into_unverified_exit_two(self):
        anchor=pose()
        snapshot={'payload_reply':payload_reply(), 'samples':[{
            'joint_reply':{'data':{'q':[0.]*16}},'pose_reply':pose_reply()}]}
        client=Mock(accid='DACH_TRON2A_253')
        client.request.side_effect=[pose_reply(),payload_reply(.36)]
        telemetry=SimpleNamespace(latest=(1,anchor),latest_joint=(1,{'q':[0.]*16}),
            packet=object(),error=None,thread=Mock(),stop=Mock())
        controller=Mock(config=Config(max_offset_m=.02),applied_offset=.012)
        observer=Mock();observer.summary.return_value={}
        trial=Mock(count=1200)
        initial=TailAcceptanceTests.rows(0,20,value=-.311)
        direction=TailAcceptanceTests.rows(2_000_000_000,20,value=-.306)
        learned=TailAcceptanceTests.rows(4_000_000_000,20,value=-.3005)
        for rows,offset in [(initial,0.),(direction,.005),(learned,.012)]:
            for row in rows:row.update(offset_m=offset,error_m=.0005)
        trial.segment.side_effect=[(initial,False),(direction,False),(learned,True),(learned,True)]
        lock=Mock()
        with TemporaryDirectory() as temp, \
                patch.object(diagnostic,'Tron2JsonClient',return_value=client), \
                patch.object(diagnostic,'MotorObserver',return_value=observer), \
                patch.object(diagnostic,'CompensationTelemetry',return_value=telemetry), \
                patch.object(diagnostic,'ZCompensationController',return_value=controller), \
                patch.object(diagnostic,'Trial',return_value=trial), \
                patch.object(diagnostic,'baseline',return_value=snapshot), \
                patch.object(diagnostic,'readonly_configuration',return_value={}), \
                patch.object(diagnostic,'prepare_envelope',return_value=(Mock(),anchor,{'maximum_command_raise_m':.02})), \
                patch.object(diagnostic,'TraceRecorder'), \
                patch.object(diagnostic,'acquire_capture_lock',return_value=lock), \
                patch.object(diagnostic,'_ACTIVE_CLIENTS',[]), \
                patch.object(diagnostic.time,'sleep'),redirect_stdout(io.StringIO()):
            out=Path(temp)/'trial'
            result=diagnostic.main(['--execute','--max-correction-mm','20','--output',str(out)])
            report=json.loads((out/'result.json').read_text())
        self.assertEqual(result,2)
        self.assertEqual(report['result'],'unverified_final_configuration')
        self.assertIn('负载配置与本轮预期不符',report['final_payload_error'])
        self.assertEqual(report['retained_offset_m'],.012)
        self.assertEqual([c.args[0] for c in client.request.call_args_list],
                         ['request_get_servo_pose','request_get_payload_param'])
        client.send.assert_not_called()
        self.assertEqual([c.args[0] for c in trial.segment.call_args_list],
                         ['baseline_hold','direction_5mm','learn_hold','final_hold'])
        telemetry.stop.set.assert_called_once()
        lock.close.assert_called_once()


if __name__=='__main__':unittest.main()
