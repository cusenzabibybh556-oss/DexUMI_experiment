"""Declarative hand-model profiles shared by simulation adapters."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MimicJoint:
    source: int
    target: int
    multiplier: float


@dataclass(frozen=True)
class HandProfile:
    key: str
    display_name: str
    command_names: Tuple[str, ...]
    joint_names: Tuple[str, ...]
    command_to_model: Tuple[int, ...]
    mimic_joints: Tuple[MimicJoint, ...]
    default_open_pose: Tuple[float, ...]

    def __post_init__(self):
        channel_count = len(self.command_names)
        if len(self.command_to_model) != channel_count:
            raise ValueError("command_to_model must match command_names")
        if len(self.default_open_pose) != channel_count:
            raise ValueError("default_open_pose must match command_names")


O6_LEFT_PROFILE = HandProfile(
    key="o6_left",
    display_name="O6 left",
    command_names=(
        "thumb_flexion",
        "thumb_abduction",
        "index_flexion",
        "middle_flexion",
        "ring_flexion",
        "pinky_flexion",
    ),
    joint_names=(
        "thumb_cmc_yaw",
        "thumb_cmc_pitch",
        "thumb_ip",
        "index_mcp_pitch",
        "index_dip",
        "middle_mcp_pitch",
        "middle_dip",
        "ring_mcp_pitch",
        "ring_dip",
        "pinky_mcp_pitch",
        "pinky_dip",
    ),
    command_to_model=(1, 0, 3, 5, 7, 9),
    mimic_joints=(
        MimicJoint(1, 2, 1.86),
        MimicJoint(3, 4, 0.89),
        MimicJoint(5, 6, 0.89),
        MimicJoint(7, 8, 0.89),
        MimicJoint(9, 10, 0.89),
    ),
    default_open_pose=(255.0, 179.0, 255.0, 255.0, 255.0, 255.0),
)


HAND_PROFILES = {O6_LEFT_PROFILE.key: O6_LEFT_PROFILE}
