import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tron2_act.deploy_core import (Timeline,PendingObservation,ReferencePoint,
    SpatialChunk,ChunkPool,LiveReference,observation_state)
from tron2_act.geometry import matrix_to_rotation6d


def actions(delta=(.01,0,0),rotation=None,hand=120.):
    r=np.eye(3) if rotation is None else rotation
    return np.tile(np.r_[delta,matrix_to_rotation6d(r),np.full(6,hand)],(50,1))


def test_observation_uses_capture_time_and_real_100_50ms_history():
    p,r,h=Timeline(),Timeline(rotations=True),Timeline()
    for ns in range(0,301_000_000,25_000_000):
        p.add(ns,np.array([ns/1e9,0,0]));r.add(ns,np.eye(3));h.add(ns,np.full(6,200.))
    state,position,_=observation_state(200_000_000,p,r,h,ReferencePoint.identity())
    np.testing.assert_allclose(state[[0,9]],[-.1,-.05],atol=1e-7)
    np.testing.assert_allclose(state[-6:],200.)
    np.testing.assert_allclose(position,[.2,0,0])
    with pytest.raises(PendingObservation):p.at(400_000_000)


def test_observation_clips_only_interpolation_roundoff_at_o6_byte_limits():
    class HandInterpolation:
        def __init__(self,value):self.value=np.asarray(value,float)
        def at(self,stamp):return self.value.copy()

    p,r=Timeline(),Timeline(rotations=True)
    for ns in range(0,301_000_000,25_000_000):
        p.add(ns,np.zeros(3));r.add(ns,np.eye(3))
    near_limits=HandInterpolation([-1e-10,255+1e-10,1,2,3,4])
    state,_,_=observation_state(200_000_000,p,r,near_limits,ReferencePoint.identity())
    np.testing.assert_array_equal(state[-6:],[0,255,1,2,3,4])
    with pytest.raises(ValueError,match='range'):
        observation_state(200_000_000,p,r,
                          HandInterpolation([0,255.001,1,2,3,4]),ReferencePoint.identity())


def test_observation_never_bridges_feedback_dropouts():
    history=Timeline(max_gap_s=.15);history.add(0,np.zeros(3));history.add(300_000_000,np.ones(3))
    with pytest.raises(RuntimeError,match='gap'):history.at(150_000_000)
    with pytest.raises(ValueError,match='backwards'):history.add(200_000_000,np.zeros(3))


def test_reference_point_compensates_rotation_lever_arm_without_changing_spatial_rotation():
    matrix=np.eye(4);matrix[:3,3]=[.1,0,0];matrix[:3,:3]=Rotation.from_euler('x',40,degrees=True).as_matrix()
    ref=ReferencePoint.from_matrix(matrix)
    delta=Rotation.from_euler('z',90,degrees=True).as_matrix()
    chunk=SpatialChunk(0,np.zeros(3),np.eye(3),actions((0,0,0),delta),ref)
    p,r,_=chunk.target(50_000_000)
    np.testing.assert_allclose(r,delta,atol=1e-10)
    np.testing.assert_allclose(p,[.1,-.1,0],atol=1e-10)
    # The virtual tracker point stays fixed while the SDK TCP moves around it.
    np.testing.assert_allclose(p+r@matrix[:3,3],[.1,0,0],atol=1e-10)


def test_pool_averages_absolute_targets_not_relative_deltas():
    pool=ChunkPool(decay=0.)
    pool.add(0,np.zeros(3),np.eye(3),actions((.02,0,0)),0)
    pool.add(50_000_000,np.array([.01,0,0]),np.eye(3),actions((.01,0,0)),50_000_000)
    # Both observations predict the same absolute target despite different deltas.
    p,_,hand=pool.target(100_000_000,hand=True)
    np.testing.assert_allclose(p,[.02,0,0]);np.testing.assert_allclose(hand,120.)


def test_pool_retains_predictions_for_their_complete_chunk_horizon():
    pool=ChunkPool(decay=0.,execution_speed=.5)
    action=actions((.01,0,0))
    pool.add(0,np.zeros(3),np.eye(3),action,0)
    for stamp in range(100_000_000,1_100_000_000,100_000_000):
        pool.add(stamp,np.zeros(3),np.eye(3),action,stamp)
    # The first 50-step prediction lasts five seconds at half speed.  It must
    # not be discarded by a short fixed history window before later steps are
    # due.
    assert pool.chunks[0].stamp==0
    pool.add(5_100_000_000,np.zeros(3),np.eye(3),action,5_100_000_000)
    assert all(chunk.end>=5_100_000_000 for chunk in pool.chunks)


def test_temporal_ensemble_prefers_earlier_act_predictions():
    pool=ChunkPool(decay=1.)
    pool.add(0,np.zeros(3),np.eye(3),actions((0,0,0)),0)
    pool.add(50_000_000,np.zeros(3),np.eye(3),actions((.1,0,0)),50_000_000)
    position,_,_=pool.target(100_000_000)
    # ACT temporal ensembling weights the earlier commitment more strongly;
    # reversing the weights would put this target above 50 mm.
    assert 0 < position[0] < .05


def test_later_hand_action_reaches_ensemble_at_its_due_time():
    pool=ChunkPool(decay=.1,execution_speed=.5)
    old=actions(hand=245.)[:1].repeat(125,axis=0)
    old[49:,9:]=100.
    pool.add(0,np.zeros(3),np.eye(3),old,0)
    open_chunk=actions(hand=245.)[:1].repeat(125,axis=0)
    for stamp in range(100_000_000,5_000_000_000,100_000_000):
        pool.add(stamp,np.zeros(3),np.eye(3),open_chunk,stamp)
    # Step 50 is due at five seconds at half speed.  The old prediction must
    # still contribute its closing command instead of being evicted at 0.5 s.
    assert np.all(pool.target(5_000_000_000,hand=True)[2] < 245.)


def test_execution_speed_retimes_chunk_without_scaling_spatial_targets():
    action=actions((.02,0,0),hand=100.)
    chunk=SpatialChunk(0,np.zeros(3),np.eye(3),action,ReferencePoint.identity(),execution_speed=.5)
    assert chunk.step==100_000_000
    assert chunk.end==5_000_000_000
    p,_,hand=chunk.target(100_000_000)
    np.testing.assert_allclose(p,[.02,0,0]);np.testing.assert_allclose(hand,100.)


@pytest.mark.parametrize('length',[25,50,75,100])
def test_spatial_chunk_accepts_model_query_lengths_without_changing_timing(length):
    action=actions((.01,0,0))[:1].repeat(length,axis=0)
    chunk=SpatialChunk(0,np.zeros(3),np.eye(3),action,ReferencePoint.identity())
    assert chunk.length==length
    assert chunk.end==length*50_000_000
    np.testing.assert_allclose(chunk.target(chunk.end)[0],[.01,0,0])


def test_hand_commands_are_not_taken_from_a_prediction_before_its_first_due_time():
    pool=ChunkPool(decay=0.)
    pool.add(0,np.zeros(3),np.eye(3),actions(hand=100),0)
    pool.add(50_000_000,np.zeros(3),np.eye(3),actions(hand=20),50_000_000)
    np.testing.assert_allclose(pool.target(75_000_000,hand=True)[2],100.)
    # Safety clamp on thumb flexion is applied before aggregation.
    np.testing.assert_allclose(pool.target(100_000_000,hand=True)[2],[90,60,60,60,60,60])


def test_old_or_out_of_order_model_results_cannot_replace_the_current_chunk():
    pool=ChunkPool();pool.add(100_000_000,np.zeros(3),np.eye(3),actions(),100_000_000)
    with pytest.raises(RuntimeError):pool.add(99_000_000,np.zeros(3),np.eye(3),actions(),100_000_000)
    with pytest.raises(PendingObservation):pool.target(3_000_000_000)


def test_configured_300ms_freshness_is_shared_by_result_and_observation_paths():
    pool=ChunkPool(max_result_age_ms=300.)
    pool.add(0,np.zeros(3),np.eye(3),actions(),299_000_000)
    with pytest.raises(RuntimeError,match='300ms'):
        pool.add(1,np.zeros(3),np.eye(3),actions(),300_000_002)
    from tron2_act.live_runtime import Observations
    obs=Observations(ReferencePoint.identity(),freshness_ms=300.)
    assert obs.freshness_ns==300_000_000
    assert obs.positions.max_gap_ns==300_000_000


def test_future_target_guard_runs_before_startup_blending():
    pool=ChunkPool();pool.add(0,np.zeros(3),np.eye(3),actions((.4,0,0)),0)
    ref=LiveReference(np.zeros(3),np.eye(3),pool,max_displacement=.35)
    ref.epoch_ns=0
    with pytest.raises(RuntimeError,match='envelope'):ref.target(.01)


def test_default_cli_does_not_dispatch_hardware(monkeypatch,tmp_path):
    from tools import deploy_tron2_act as cli
    calls=[]
    monkeypatch.setattr(cli,'offline',lambda c,args,out:calls.append('offline') or {'hardware_connections':0})
    monkeypatch.setattr(cli,'live',lambda *args:pytest.fail('Default invoked hardware'))
    assert cli.main(['--output',str(tmp_path/'run')])==0
    assert calls==['offline']


def test_shadow_requires_an_existing_hand_bridge_without_initializing_hardware(tmp_path):
    from tools import deploy_tron2_act as cli
    with pytest.raises(SystemExit) as exc:cli.main(['--shadow','--output',str(tmp_path/'run')])
    assert exc.value.code==2


def test_live_modes_are_mutually_exclusive(tmp_path):
    from tools import deploy_tron2_act as cli
    with pytest.raises(SystemExit):cli.main(['--shadow','--execute','--output',str(tmp_path/'run')])


def test_cli_supports_manual_stop_and_retimed_execution(monkeypatch,tmp_path):
    from tools import deploy_tron2_act as cli
    captured={}
    monkeypatch.setattr(cli,'live',lambda c,args,out:captured.update(c) or {'completed':True})
    assert cli.main(['--execute','--max-duration','0','--execution-speed','.5',
                     '--output',str(tmp_path/'run')])==0
    assert captured['max_duration_s'] is None
    assert captured['execution_speed']==.5


def test_initial_o6_guard_rejects_wrong_start_shape():
    from tools.deploy_tron2_act import require_initial_o6_state
    config={'initial_o6_qpos':[245,237,210,187,183,174],'initial_o6_tolerance':5.}
    require_initial_o6_state(config,[245,237,209,187,183,174])
    with pytest.raises(RuntimeError,match='reset_o6_act_start'):
        require_initial_o6_state(config,[245,53,81,118,97,108])


def test_live_reference_holds_last_target_when_prediction_chunk_expires():
    pool=ChunkPool(decay=0.)
    pool.add(0,np.zeros(3),np.eye(3),actions((.02,0,0)),0)
    ref=LiveReference(np.zeros(3),np.eye(3),pool,ramp_s=.01);ref.epoch_ns=0
    first=ref.target(.05)
    held=ref.target(10.)
    np.testing.assert_allclose(held[0],first[0])
    np.testing.assert_allclose(held[1],first[1])


def test_insufficient_foreground_is_a_recoverable_observation_rejection():
    from tron2_act.live_policy import ForegroundUnavailable,LivePolicy
    fg={'hand':np.zeros((8,8),bool),'objects':np.zeros((8,8),bool)}
    with pytest.raises(ForegroundUnavailable,match='insufficient'):
        LivePolicy.check_masks(fg)


def test_live_io_rejects_stale_camera_and_missing_applied_ack():
    from tron2_act.live_runtime import ROSInputs
    io=ROSInputs({},None)
    with pytest.raises(PendingObservation):io.healthy(1_000_000_000)
    io.camera={'received_ns':950_000_000}
    with pytest.raises(RuntimeError,match='confirmed'):io.healthy(1_000_000_000,need_applied=True)
    io.last_applied_ns=950_000_000
    io.healthy(1_000_000_000,need_applied=True)
    with pytest.raises(PendingObservation):io.healthy(1_200_000_000)


def test_live_io_uses_configured_300ms_camera_and_ack_age():
    from types import SimpleNamespace
    from tron2_act.live_runtime import ROSInputs
    io=ROSInputs({'io_freshness_ms':300.},None);io.process=SimpleNamespace(poll=lambda:None)
    io.camera={'received_ns':700_000_001};io.last_applied_ns=700_000_001
    io.healthy(1_000_000_000,need_applied=True)
    io.camera={'received_ns':699_999_999}
    with pytest.raises(PendingObservation):io.healthy(1_000_000_000)
    # Active rollout treats camera loss as a policy hold, while the I/O
    # process and applied O6 acknowledgements remain hard requirements.
    io.last_applied_ns=700_000_001
    io.healthy(1_000_000_000,need_applied=True,require_camera=False)


def test_live_observation_uses_monotonic_receipt_discards_backlog_and_tolerates_wall_clock_step(monkeypatch):
    import queue
    from types import SimpleNamespace
    import tron2_act.live_runtime as runtime
    from tron2_act.live_runtime import Observations
    obs=Observations(ReferencePoint.identity())
    io=SimpleNamespace(hand_events=queue.Queue())
    now=1_050_000_000
    monkeypatch.setattr(runtime.time,'monotonic_ns',lambda:now)
    event={'received_ns':1_000_000_000,'wall_received_ns':10_000_000_000,
           'source_stamp_ns':9_980_000_000,'q_pos':[200.]*6,
           'names':['thumb_flexion','thumb_abduction','index_flexion','middle_flexion','ring_flexion','pinky_flexion']}
    io.hand_events.put(event);obs.pump(SimpleNamespace(packet=None),io)
    assert obs.last_hand_ns==1_000_000_000
    io.hand_events.put(dict(event,received_ns=800_000_000,source_stamp_ns=9_000_000_000))
    obs.pump(SimpleNamespace(packet=None),io)
    assert obs.last_hand_ns==1_000_000_000
    # A backward wall-clock step does not invalidate a fresh monotonic receipt.
    io.hand_events.put(dict(event,received_ns=1_040_000_000,source_stamp_ns=9_000_000_000))
    obs.pump(SimpleNamespace(packet=None),io)
    assert obs.last_hand_ns==1_040_000_000
    io.hand_events.put(dict(event,received_ns=1_060_000_000))
    with pytest.raises(RuntimeError,match='future'):obs.pump(SimpleNamespace(packet=None),io)


def test_live_observation_rejects_actual_raw_o6_out_of_range(monkeypatch):
    import queue
    from types import SimpleNamespace
    import tron2_act.live_runtime as runtime
    from tron2_act.live_runtime import Observations
    obs=Observations(ReferencePoint.identity())
    io=SimpleNamespace(hand_events=queue.Queue())
    monkeypatch.setattr(runtime.time,'monotonic_ns',lambda:1_050_000_000)
    io.hand_events.put({
        'received_ns':1_000_000_000,
        'q_pos':[245,256,210,187,183,174],
        'names':['thumb_flexion','thumb_abduction','index_flexion',
                 'middle_flexion','ring_flexion','pinky_flexion'],
    })
    with pytest.raises(RuntimeError,match='Raw O6 q_pos'):
        obs.pump(SimpleNamespace(packet=None),io)


def test_startup_discards_unrecoverable_image_but_waits_for_right_endpoint():
    from tron2_act.live_runtime import Observations
    obs = Observations(ReferencePoint.identity())
    start = 1_000_000_000

    def add_feedback(stamp):
        obs.positions.add(stamp, np.zeros(3))
        obs.rotations.add(stamp, np.eye(3))
        obs.hands.add(stamp, np.full(6, 120.))
        obs.last_hand_ns = stamp

    def frame(stamp):
        return dict(sequence=1, received_ns=stamp, jpeg='', info={})

    add_feedback(start)
    # Its t-100 ms can never be bracketed, however many new samples arrive.
    early = start + 50_000_000
    assert obs.packet(frame(early), early, startup_cutoff=start) is None
    add_feedback(early)
    add_feedback(start + 100_000_000)
    usable = start + 125_000_000
    with pytest.raises(PendingObservation):
        obs.packet(frame(usable), usable, startup_cutoff=start)
    add_feedback(start + 150_000_000)
    assert obs.packet(frame(usable), obs.last_hand_ns, startup_cutoff=start)['stamp_ns'] == usable
    # Stale input in active control still triggers a stop.
    with pytest.raises(RuntimeError, match='older'):
        obs.packet(frame(usable), usable + 200_000_000)


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('sign', [-1, 1])
def test_act_servo_sender_opposes_injected_base_frame_error(monkeypatch, axis, sign):
    """Isolate feedback from feedforward, capture the actual paired ServoP payload."""
    from types import SimpleNamespace
    from unittest.mock import Mock
    import tron2_act.live_runtime as runtime
    import tron2_pose_replay as servo
    from replay_tron2_tracker import DualArmPose
    from replay_tron2_fixed_start import FIXED_LEFT_POSITION, FIXED_LEFT_QUATERNION, FIXED_LEFT_Q

    pos = FIXED_LEFT_POSITION
    orientation = Rotation.from_quat(FIXED_LEFT_QUATERNION[[1, 2, 3, 0]])
    right = np.array([.1, -.2, -.75])
    anchor = DualArmPose(pos, FIXED_LEFT_QUATERNION, right, np.array([1., 0., 0., 0.]))
    q = np.r_[FIXED_LEFT_Q, np.zeros(9)]
    client = Mock()
    client.failure.return_value = None
    ref = LiveReference(pos, orientation.as_matrix(), ChunkPool())
    session = runtime.make_pose_session(client, anchor, q, ref,
        dict(pose_preview_ms=124., pose_feedback_age_ms=0., robot_ip='unused'), client.trace)
    # Inject numerical timing/feedback only. Never start a robot client.
    clock = SimpleNamespace(monotonic_ns=lambda: now, perf_counter=lambda: 0.)
    monkeypatch.setattr(runtime, 'time', clock)
    monkeypatch.setattr(servo, 'time', clock)
    session.engine = SimpleNamespace(step=lambda p, r, dt, weight: (p, r, {}))
    session.initial_q = q
    session.telemetry = Mock()
    epoch = 1_000_000_000
    # A valid constant ACT reference decoded through the live chunk pool.
    ref.pool.add(epoch, pos, orientation.as_matrix(), actions((0, 0, 0)), epoch)
    ref.epoch_ns = session.start_ns = epoch
    direction = sign*np.eye(3)[axis]
    actual_r = Rotation.from_rotvec(np.deg2rad(direction)) * orientation
    actual = DualArmPose(pos+.004*direction, actual_r.as_quat()[[3,0,1,2]],
                         right, anchor.right_quaternion)
    for i in range(46):
        now = epoch+round(i*1e9/300)
        if i%15 == 0:
            session.telemetry.packet = SimpleNamespace(guid=str(i), pose=actual, sent_ns=now-2_000_000,
                received_ns=now, source_timestamp_ms=now/1e6, source_progress=True)
            session.telemetry.latest_joint = (now, {'q':q})
        session.stream('act_control')
    name, data = client.send.call_args.args
    assert name == 'request_servop'
    left = np.array(data['left_pos'])
    issued_r = Rotation.from_quat(left[[4,5,6,3]])
    assert (left[:3]-pos) @ direction < -1e-6
    assert (issued_r*orientation.inv()).as_rotvec() @ direction < -1e-6
    np.testing.assert_allclose(data['right_pos'], np.r_[right, anchor.right_quaternion])
