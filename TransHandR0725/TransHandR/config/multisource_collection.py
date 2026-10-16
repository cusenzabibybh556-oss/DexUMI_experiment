"""Quality-first V2 multisource collection defaults.

All timing thresholds are intentionally independent.  Tune them from measured
device rate/jitter rather than replacing them with one global threshold.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


HAND_SPECS: Dict[str, Dict[str, object]] = {
    "inspire": {
        "state_dim": 6,
        "action_dim": 6,
        "state_unit": "device_angle_register_0_1000",
        "action_unit": "device_angle_register_0_1000",
        "state_joint_names": ["little", "ring", "middle", "index", "thumb_bend", "thumb_rotate"],
    },
    "l10": {
        "state_dim": 10,
        "action_dim": 10,
        "state_unit": "actuator_position_0_255",
        "action_unit": "actuator_position_0_255",
        "state_joint_names": [f"actuator_{i}" for i in range(10)],
    },
    "l21": {
        "state_dim": 25,
        "action_dim": 25,
        "state_unit": "actuator_position_0_255",
        "action_unit": "actuator_position_0_255",
        "state_joint_names": [f"actuator_{i}" for i in range(25)],
    },
    "o6": {
        "state_dim": 6,
        "action_dim": 6,
        "state_unit": "actuator_position_0_255",
        "action_unit": "actuator_position_0_255",
        "state_joint_names": [
            "thumb_flexion", "thumb_abduction", "index_flexion",
            "middle_flexion", "ring_flexion", "pinky_flexion",
        ],
    },
}

DEFAULT_RECORD_CHANNELS = ["d405", "d435i", "vision_hand", "operator_exoskeleton",
                           "hand_state", "action", "tactile", "inspire_force", "vive_tracker"]


@dataclass
class CollectionConfig:
    control_source: str = "vision_pro"
    target_hand_model: str = ""
    avp_ip: str = "10.227.197.16"
    enable_right_hand: bool = True
    right_hand_source: str = "udp"
    vision_hand_side: str = "right"
    enable_hand_io: bool = True
    hand_source: str = "udp"
    hand_type: str = "l21"
    hand_side: str = "right"
    hand_serial_port: str = "/dev/ttyUSB0"
    hand_can_interface: str = "can0"
    enable_exoskeleton_operator: bool = False
    enable_tactile: bool = False
    enable_inspire_force: bool = False

    d405_serial: str = ""
    image_width: int = 640
    image_height: int = 480
    camera_fps: int = 30
    depth_aligned_to_color: bool = True

    enable_head_camera: bool = False
    head_camera_serial: str = ""
    head_image_width: int = 640
    head_image_height: int = 480
    head_camera_fps: int = 30
    head_depth_aligned_to_color: bool = True
    head_camera_max_gap_ms: float = 45.0

    enable_vive_tracker: bool = False
    tracker_mode: str = "pyvut"
    tracker_index: int = 0

    right_hand_max_gap_ms: float = 45.0
    hand_state_max_gap_ms: float = 35.0
    tracker_max_gap_ms: float = 30.0
    # Action is a stateful stream of commands successfully applied by the
    # bridge. Zero means keep the last applied command until it is replaced.
    action_max_age_ms: float = 0.0
    operator_max_gap_ms: float = 45.0
    tactile_max_gap_ms: float = 120.0
    tactile_contact_threshold: float = 1.0
    telemetry_protocol_version: str = "transhandr.telemetry.v2"

    output_dir: str = "data/multisource"
    task_name: str = "transhandr_demo"

    enable_realtime_dashboard: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8080
    dashboard_update_hz: int = 10
    hdf5_chunk_frames: int = 1
    compression: str = "lzf"
    max_episode_duration_s: float = 300.0
    max_episode_size_gb: float = 5.0
    selected_channels: List[str] = field(default_factory=lambda: list(DEFAULT_RECORD_CHANNELS))

    def __post_init__(self) -> None:
        if not self.target_hand_model:
            self.target_hand_model = self.hand_type

    @property
    def alignment_lag_ms(self) -> float:
        gaps = [0.0]
        if self.enable_right_hand: gaps.append(self.right_hand_max_gap_ms)
        if self.enable_hand_io: gaps.append(self.hand_state_max_gap_ms)
        if self.enable_exoskeleton_operator: gaps.append(self.operator_max_gap_ms)
        if self.enable_vive_tracker: gaps.append(self.tracker_max_gap_ms)
        if self.enable_head_camera: gaps.append(self.head_camera_max_gap_ms)
        return max(gaps)

    @property
    def hand_spec(self) -> Dict[str, object]:
        key = self.hand_type.lower()
        if key not in HAND_SPECS:
            raise ValueError(f"Unsupported hand_type={self.hand_type!r}; expected {sorted(HAND_SPECS)}")
        return HAND_SPECS[key]

    @property
    def episode_dir(self) -> Path:
        return Path(self.output_dir).expanduser() / self.task_name / self.hand_type.lower()

    @property
    def state_joint_names(self) -> List[str]:
        return list(self.hand_spec["state_joint_names"])

    @property
    def action_names(self) -> List[str]:
        return list(self.hand_spec["state_joint_names"])
