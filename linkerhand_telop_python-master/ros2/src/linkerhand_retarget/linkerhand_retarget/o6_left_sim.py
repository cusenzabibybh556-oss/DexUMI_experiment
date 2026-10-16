#!/usr/bin/env python3
"""MuJoCo preview for LinkerHand O6 left-hand commands."""

import math
import signal
import threading
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

from linkerhand_retarget.command_limits import CommandLimiter
from linkerhand_retarget.hand_profiles import O6_LEFT_PROFILE


JOINT_NAMES = list(O6_LEFT_PROFILE.joint_names)
COMMAND_TO_URDF = O6_LEFT_PROFILE.command_to_model
MIMIC_JOINTS = tuple(
    (joint.source, joint.target, joint.multiplier)
    for joint in O6_LEFT_PROFILE.mimic_joints
)
DEFAULT_OPEN_POSE = O6_LEFT_PROFILE.default_open_pose


def find_default_model() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("linkerhand_retarget"))
        installed = share / "o6_left" / "linkerhand_o6_left.urdf"
        if installed.is_file():
            return installed
    except (ImportError, LookupError):
        pass

    source = Path(__file__).resolve().parents[1]
    return source / "resource" / "o6_left" / "linkerhand_o6_left.urdf"


class O6Kinematics:
    """Convert six O6 byte commands to the eleven-joint visual model."""

    def __init__(self, model):
        self.model = model
        if model.nq != len(JOINT_NAMES):
            raise RuntimeError(
                f"Expected {len(JOINT_NAMES)} model positions, got {model.nq}"
            )

    def command_to_qpos(
        self, command: Iterable[float]
    ) -> Tuple[np.ndarray, int]:
        values = list(command)
        if len(values) != 6:
            raise ValueError(f"O6 expects 6 command values, got {len(values)}")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("O6 command contains NaN or infinity")

        clipped = np.clip(np.asarray(values, dtype=float), 0.0, 255.0)
        clamp_count = int(np.count_nonzero(clipped != values))
        qpos = np.zeros(self.model.nq, dtype=float)

        for command_value, joint_index in zip(clipped, COMMAND_TO_URDF):
            lower, upper = self.model.jnt_range[joint_index]
            flexion = (255.0 - command_value) / 255.0
            qpos[joint_index] = lower + flexion * (upper - lower)

        for source, target, multiplier in MIMIC_JOINTS:
            lower, upper = self.model.jnt_range[target]
            qpos[target] = np.clip(qpos[source] * multiplier, lower, upper)

        return qpos, clamp_count


class O6LeftSimNode(Node):
    def __init__(self, model, data, kinematics: O6Kinematics):
        super().__init__("o6_left_sim_node")
        self.declare_parameter("topic", "/cb_left_hand_control_cmd")
        self.declare_parameter("open_pose", list(DEFAULT_OPEN_POSE))
        self.declare_parameter("command_min", [0.0] * 6)
        self.declare_parameter("command_max", [255.0] * 6)
        self.declare_parameter("stale_timeout", 1.0)

        self.topic = str(self.get_parameter("topic").value)
        stale_timeout = float(self.get_parameter("stale_timeout").value)
        self.stale_timeout = max(0.1, stale_timeout)
        open_pose = self._six_values(
            self.get_parameter("open_pose").value, "open_pose"
        )
        self.command_limiter = CommandLimiter(
            self.get_parameter("command_min").value,
            self.get_parameter("command_max").value,
        )

        self.model = model
        self.data = data
        self.kinematics = kinematics
        self._lock = threading.Lock()
        self._command = open_pose
        self._last_message_time = 0.0
        self._message_count = 0
        self._clamp_count = 0
        self._bad_message_count = 0

        self.publisher = self.create_publisher(
            JointState, "/o6_left_sim/joint_states", 10
        )
        self.subscription = self.create_subscription(
            JointState, self.topic, self._on_command, 10
        )
        self.get_logger().info(
            f"O6 left simulation ready: topic={self.topic}; "
            "no CAN device is used"
        )

    @staticmethod
    def _six_values(values, name: str) -> List[float]:
        result = [float(value) for value in values]
        if len(result) != 6:
            raise ValueError(
                f"{name} must contain 6 values, got {len(result)}"
            )
        return result

    def _on_command(self, msg: JointState):
        try:
            command = self._six_values(msg.position, "JointState.position")
            command, limited = self.command_limiter.apply(command)
        except (TypeError, ValueError) as exc:
            self._bad_message_count += 1
            first_or_periodic = (
                self._bad_message_count == 1
                or self._bad_message_count % 50 == 0
            )
            if first_or_periodic:
                self.get_logger().warn(f"Ignored invalid O6 command: {exc}")
            return

        with self._lock:
            self._command = command
            self._last_message_time = time.monotonic()
            self._message_count += 1
            self._clamp_count += limited

    def update_model(self):
        with self._lock:
            command = list(self._command)
            last_message_time = self._last_message_time
            message_count = self._message_count

        qpos, clamped = self.kinematics.command_to_qpos(command)
        self._clamp_count += clamped
        self.data.qpos[:] = qpos

        import mujoco

        mujoco.mj_forward(self.model, self.data)
        self._publish_joint_state(qpos)
        age = (
            time.monotonic() - last_message_time
            if last_message_time
            else math.inf
        )
        return command, qpos, message_count, age

    def _publish_joint_state(self, qpos: np.ndarray):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = qpos.tolist()
        self.publisher.publish(msg)

    @property
    def clamp_count(self) -> int:
        return self._clamp_count


def _non_base_contacts(model, data, mujoco) -> int:
    contacts = set()
    for contact in data.contact[:data.ncon]:
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        # The fixed palm is fused into MuJoCo's world body. Its contacts with
        # directly attached links are expected for this vendor mesh model.
        if body1 == 0 or body2 == 0:
            continue
        contacts.add(tuple(sorted((body1, body2))))
    return len(contacts)


def _overlay(
    viewer, node, command, qpos, message_count, age, contacts, mujoco
):
    status = "LIVE" if age <= node.stale_timeout else "WAITING / STALE"
    command_text = " ".join(f"{value:3.0f}" for value in command)
    angle_text = " ".join(f"{qpos[index]:.2f}" for index in COMMAND_TO_URDF)
    viewer.set_texts([
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "O6 LEFT PREVIEW\nStatus\nTopic\nMessages\nClamped\n"
            "Finger contacts",
            f"\n{status}\n{node.topic}\n{message_count}\n"
            f"{node.clamp_count}\n{contacts}",
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "Command [0..255]\nJoint target [rad]",
            f"{command_text}\n{angle_text}",
        ),
    ])


def _run_viewer(node, model, data, update_hz: float, stop_event):
    import mujoco
    import mujoco.viewer

    reset_requested = threading.Event()

    def on_key(keycode):
        if keycode in (ord("R"), ord("r")):
            reset_requested.set()

    period = 1.0 / max(1.0, update_hz)
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=on_key,
        show_left_ui=False,
        show_right_ui=True,
    ) as viewer:
        viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance = model.stat.extent * 2.0
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -25.0

        while not stop_event.is_set() and rclpy.ok() and viewer.is_running():
            start = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.0)
            if reset_requested.is_set():
                with node._lock:
                    node._command = list(DEFAULT_OPEN_POSE)
                reset_requested.clear()

            command, qpos, count, age = node.update_model()
            contacts = _non_base_contacts(model, data, mujoco)
            _overlay(viewer, node, command, qpos, count, age, contacts, mujoco)
            viewer.sync()
            time.sleep(max(0.0, period - (time.monotonic() - start)))

        # close() requests GUI shutdown asynchronously. Wait for the native
        # GLFW thread before Python starts destroying MuJoCo objects.
        viewer.close()
        deadline = time.monotonic() + 2.0
        while viewer.is_running() and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.25)


def _run_headless(node, update_hz: float, stop_event):
    period = 1.0 / max(1.0, update_hz)
    while not stop_event.is_set() and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.0)
        node.update_model()
        time.sleep(period)


def main(args=None):
    from rclpy.signals import SignalHandlerOptions

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        bootstrap = Node("o6_left_sim_bootstrap")
        bootstrap.declare_parameter("model_path", "")
        bootstrap.declare_parameter("headless", False)
        bootstrap.declare_parameter("update_hz", 60.0)
        model_parameter = str(bootstrap.get_parameter("model_path").value)
        headless = bool(bootstrap.get_parameter("headless").value)
        update_hz = float(bootstrap.get_parameter("update_hz").value)
        bootstrap.destroy_node()

        model_path = (
            Path(model_parameter).expanduser()
            if model_parameter
            else find_default_model()
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"O6 left model not found: {model_path}")

        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo is not installed. Run: python3 -m pip install "
                "--user 'mujoco>=3.2,<4'"
            ) from exc

        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        kinematics = O6Kinematics(model)
        node = O6LeftSimNode(model, data, kinematics)

        if headless:
            _run_headless(node, update_hz, stop_event)
        else:
            _run_viewer(node, model, data, update_hz, stop_event)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
