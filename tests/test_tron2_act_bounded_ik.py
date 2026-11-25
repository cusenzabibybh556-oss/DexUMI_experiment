import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tron2_act.simulation_ik import BoundedPoseReferenceModel


def analytic_model():
    model=object.__new__(BoundedPoseReferenceModel)
    model.lower=np.full(16,-1.)
    model.upper=np.full(16,1.)
    model.forward=lambda q:(q[:3].copy(),Rotation.from_rotvec(q[3:6]).as_matrix(),q[:3].copy())
    return model


def test_bounded_ik_reaches_a_small_pose_change_without_moving_held_arm():
    model=analytic_model()
    seed=np.zeros(16);seed[7:]=.1
    q=model.solve(np.array([.005,-.003,.002]),Rotation.from_rotvec([.005,0,0]).as_matrix(),seed)
    assert np.linalg.norm(q[:3]-[.005,-.003,.002])<.001
    assert np.max(np.abs(q[:7]-seed[:7]))<=.02
    np.testing.assert_array_equal(q[7:],seed[7:])


def test_unreachable_request_is_rejected_without_expanding_joint_or_step_bounds():
    model=analytic_model()
    with pytest.raises(RuntimeError,match='residual'):
        model.solve(np.array([2.,0,0]),np.eye(3),np.zeros(16))
    np.testing.assert_array_equal(model.lower,np.full(16,-1.))
    np.testing.assert_array_equal(model.upper,np.full(16,1.))
