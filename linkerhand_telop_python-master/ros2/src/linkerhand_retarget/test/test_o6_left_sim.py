from pathlib import Path

import mujoco
import numpy as np

from linkerhand_retarget.o6_left_sim import (
    COMMAND_TO_URDF,
    DEFAULT_OPEN_POSE,
    MIMIC_JOINTS,
    O6Kinematics,
)


def _model_and_kinematics():
    package_root = Path(__file__).resolve().parents[1]
    model_path = (
        package_root
        / "resource"
        / "o6_left"
        / "linkerhand_o6_left.urdf"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return model, O6Kinematics(model)


def test_left_open_pose_uses_o6_channel_order():
    model, kinematics = _model_and_kinematics()
    qpos, clamp_count = kinematics.command_to_qpos(DEFAULT_OPEN_POSE)

    assert clamp_count == 0
    assert qpos[COMMAND_TO_URDF[0]] == 0.0
    assert qpos[COMMAND_TO_URDF[1]] > 0.0
    assert np.allclose(qpos[list(COMMAND_TO_URDF[2:])], 0.0)


def test_full_flexion_respects_model_limits_and_mimics():
    model, kinematics = _model_and_kinematics()
    qpos, clamp_count = kinematics.command_to_qpos([0.0] * 6)

    assert clamp_count == 0
    for joint_index in COMMAND_TO_URDF:
        assert np.isclose(qpos[joint_index], model.jnt_range[joint_index, 1])
    for _, target, _ in MIMIC_JOINTS:
        lower, upper = model.jnt_range[target]
        assert lower <= qpos[target] <= upper


def test_out_of_range_commands_are_clamped():
    model, kinematics = _model_and_kinematics()
    qpos, clamp_count = kinematics.command_to_qpos(
        [-20.0, 300.0, 255.0, 255.0, 255.0, 255.0]
    )

    assert clamp_count == 2
    assert np.isclose(qpos[COMMAND_TO_URDF[0]], model.jnt_range[1, 1])
    assert np.isclose(qpos[COMMAND_TO_URDF[1]], model.jnt_range[0, 0])
