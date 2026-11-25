"""Declarative teleoperation device profiles used by the platform."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TeleoperationProfile:
    control_source: str
    robot_model: str
    hand_sides: Tuple[str, ...]
    action_dim: int
    state_dim: int
    joint_names: Tuple[str, ...]
    command_min: Tuple[float, ...]
    command_max: Tuple[float, ...]
    driver: str
    command_topic: str = ""
    action_topic: str = ""
    state_topic: str = ""
    operator_topic: str = ""
    tactile_topic: str = ""
    force_topic: str = ""
    tactile_shape: Optional[Tuple[int, ...]] = None
    requires_can: bool = False
    requires_hand_serial: bool = False
    requires_calibration: bool = False

    @property
    def key(self) -> str:
        return f"{self.control_source}:{self.robot_model}"

    def public(self) -> dict:
        value = asdict(self)
        value["key"] = self.key
        return value

    def topic(self, name: str, hand_side: str) -> str:
        value = getattr(self, name)
        return value.format(side=hand_side) if value else ""


O6_NAMES = ("thumb_flexion", "thumb_abduction", "index_flexion",
            "middle_flexion", "ring_flexion", "pinky_flexion")

PROFILES: Dict[str, TeleoperationProfile] = {}


def _register(profile: TeleoperationProfile) -> None:
    PROFILES[profile.key] = profile


for model, dim, limit in (("inspire", 6, 1000.0), ("l10", 10, 255.0), ("l21", 25, 255.0)):
    _register(TeleoperationProfile(
        control_source="vision_pro", robot_model=model,
        hand_sides=("left", "right"), action_dim=dim, state_dim=dim,
        joint_names=tuple(f"joint_{i + 1}" for i in range(dim)),
        command_min=(0.0,) * dim, command_max=(limit,) * dim,
        driver="visionpro_multi_hand",
    ))

_register(TeleoperationProfile(
    control_source="linkerforce", robot_model="o6", hand_sides=("left", "right"),
    action_dim=6, state_dim=6, joint_names=O6_NAMES,
    command_min=(100.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    command_max=(255.0,) * 6, driver="linkerforce_o6_ros",
    command_topic="/cb_{side}_hand_control_cmd",
    action_topic="/robot/o6/{side}/action_applied",
    state_topic="/robot/o6/{side}/joint_states",
    operator_topic="/operator/linkerforce/{side}/joint_states",
    tactile_topic="/robot/o6/{side}/tactile_matrix",
    tactile_shape=(5, 12, 6),
    requires_can=True,
    requires_calibration=True,
))

_register(TeleoperationProfile(
    control_source="linkerforce", robot_model="inspire",
    hand_sides=("left", "right"), action_dim=6, state_dim=6,
    joint_names=("little", "ring", "middle", "index", "thumb_bend", "thumb_rotate"),
    command_min=(0.0,) * 6, command_max=(1000.0,) * 6,
    driver="linkerforce_inspire_ros",
    command_topic="/cb_{side}_hand_control_cmd",
    action_topic="/robot/inspire/{side}/action_applied",
    state_topic="/robot/inspire/{side}/joint_states",
    operator_topic="/operator/linkerforce/{side}/joint_states",
    requires_hand_serial=True,
    force_topic="/robot/inspire/{side}/force_states",
    requires_calibration=True,
))


def get_profile(control_source: str, robot_model: str, hand_side: str) -> TeleoperationProfile:
    key = f"{control_source}:{robot_model}"
    try:
        profile = PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported control/robot combination: {key}") from exc
    if hand_side not in profile.hand_sides:
        raise ValueError(f"{key} does not support {hand_side} hand")
    return profile


def public_profiles() -> List[dict]:
    return [profile.public() for profile in PROFILES.values()]
