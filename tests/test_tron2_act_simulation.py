import numpy as np
from scipy.spatial.transform import Rotation

from tron2_act.geometry import matrix_to_rotation6d
from tron2_act.simulation import (ActionChunk, PoseHistory, hand_to_angles,
                                  angles_to_hand, HAND_LIMITS, SimServoP)
from tron2_pose_compensation import PoseCorrection


def test_chunk_uses_spatial_left_rotation_and_physical_time():
    anchor = Rotation.from_euler('x',90,degrees=True).as_matrix()
    delta = Rotation.from_euler('z',30,degrees=True).as_matrix()
    a = np.r_[.01,0,0,matrix_to_rotation6d(delta),np.full(6,100.)][None]
    chunk = ActionChunk(10.,np.zeros(3),anchor,np.full(6,255.),a)
    p,r,hand = chunk.target(10.05)
    np.testing.assert_allclose(p,[.01,0,0])
    np.testing.assert_allclose(r,delta@anchor,atol=1e-10)
    np.testing.assert_allclose(hand,100.)
    np.testing.assert_allclose(chunk.target(10.025)[0],[.005,0,0])
    np.testing.assert_allclose(chunk.target(10.025)[2],255.)


def test_history_is_measured_past_at_physical_100_and_50ms():
    history = PoseHistory()
    for t in np.linspace(0,.2,61):
        history.append(t,np.array([t,0,0]),np.eye(3))
    state = history.state(.2,np.array([.2,0,0]),np.eye(3),np.arange(6))
    np.testing.assert_allclose(state[[0,9]],[-.1,-.05],atol=1e-7)
    np.testing.assert_allclose(state[-6:],np.arange(6))


def test_hand_mapping_matches_existing_open_close_direction():
    np.testing.assert_allclose(hand_to_angles(np.full(6,255)),0)
    np.testing.assert_allclose(hand_to_angles(np.zeros(6)),HAND_LIMITS)
    command = np.array([20,40,60,100,200,250])
    np.testing.assert_allclose(angles_to_hand(hand_to_angles(command)), command)


def test_delayed_feedback_uses_capture_time_and_current_controller_clock():
    adapter = object.__new__(SimServoP)
    adapter.feedback = True
    adapter.controller = PoseCorrection()
    adapter.controller.advance(1_080_000_000)
    adapter.observe(.05,np.zeros(3),np.eye(3),np.zeros(3),np.eye(3),now=.1)
    adapter.controller.advance(1_100_000_000)
    assert adapter.controller.status == 'primed'
    adapter.observe(.06,np.zeros(3),np.eye(3),np.zeros(3),np.eye(3),now=.4)
    assert adapter.controller.status == 'stale_or_high_rtt'


def test_o6_command_remains_held_across_replanning_with_lagging_q_pos():
    from tron2_act.simulation import HandCommandClock
    initial = np.full(6,255.)
    a = np.r_[np.zeros(3),matrix_to_rotation6d(np.eye(3)),np.full(6,100.)][None]
    first = ActionChunk(0.,np.zeros(3),np.eye(3),initial,a)
    clock = HandCommandClock(initial)
    clock.submit(first)
    np.testing.assert_allclose(clock.advance(.049),255.)
    np.testing.assert_allclose(clock.advance(.05),100.)
    a[:,9:] = 80.
    # Physics is still open at the next query; this measured state is not a command.
    clock.submit(ActionChunk(.05,np.zeros(3),np.eye(3),initial,a))
    np.testing.assert_allclose(clock.advance(.06),100.)
    np.testing.assert_allclose(clock.advance(.1),80.)
