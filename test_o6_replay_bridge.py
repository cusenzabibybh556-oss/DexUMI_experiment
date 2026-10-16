"""回放专用 O6 桥接的离线进程生命周期测试；不访问 CAN 或 ROS。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import o6_replay_bridge as bridge


class FakeProcess:
    pid = 98765

    def __init__(self):
        self.running = True
        self.signals = []

    def poll(self):
        return None if self.running else 0

    def wait(self, timeout):
        self.running = False
        return 0


class O6ReplayBridgeTests(unittest.TestCase):
    def setUp(self):
        # Offline lifecycle tests must not require a prebuilt ROS workspace.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for constant, filename in (
            ("ROS_SETUP", "setup.bash"),
            ("BRIDGE_EXECUTABLE", "o6_bridge"),
            ("LIMITS", "o6_command_limits.yaml"),
        ):
            path = root / filename
            path.touch()
            patcher = patch.object(bridge, constant, path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_missing_installation_never_opens_bridge(self):
        bridge.ROS_SETUP.unlink()
        owner = bridge.O6ReplayBridge("can1", bridge.ROS_SETUP.parent / "bridge.log")
        with patch.object(bridge.subprocess, "run") as probe, \
             patch.object(bridge.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "安装或配置文件不存在"):
                owner.start(Mock())
        probe.assert_not_called()
        launch.assert_not_called()

    def test_can_name_and_command_are_scoped_to_left_o6(self):
        for invalid in ("/dev/ttyUSB0", "can1;echo bad", "eth0", ""):
            with self.assertRaises(ValueError):
                bridge.validate_o6_can(invalid)
        with tempfile.TemporaryDirectory() as directory:
            owner = bridge.O6ReplayBridge("can1", Path(directory) / "bridge.log")
            shell = owner.command()[-1]
            self.assertIn("can:=can1", shell)
            self.assertIn("hand_type:=left", shell)
            self.assertIn("o6_command_limits.yaml", shell)
            self.assertNotIn(" handretarget --ros-args", shell)

    def test_existing_bridge_prevents_second_can_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = bridge.O6ReplayBridge("can1", Path(directory) / "bridge.log")
            up = Mock(returncode=0, stdout="can1 UP <NOARP,UP>")
            with patch.object(bridge.subprocess, "run", return_value=up), \
                 patch.object(bridge, "existing_o6_bridges", return_value=[123]), \
                 patch.object(bridge.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "已有 O6 桥接"):
                    owner.start(Mock())
            launch.assert_not_called()

    def test_owned_bridge_is_closed_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = bridge.O6ReplayBridge("can1", Path(directory) / "bridge.log")
            up = Mock(returncode=0, stdout="can1 UP <NOARP,UP>")
            process = FakeProcess()
            probe = Mock()
            def launch(_command, **kwargs):
                kwargs["stdout"].write(
                    b"[INFO] O6 bridge ready: hand_type=left, can=can1\n"
                )
                return process
            with patch.object(bridge.subprocess, "run", return_value=up), \
                 patch.object(bridge, "existing_o6_bridges", return_value=[]), \
                 patch.object(bridge.subprocess, "Popen", side_effect=launch), \
                 patch.object(bridge.os, "killpg") as stop:
                owner.start(probe)
                probe.probe_bridge.assert_called_once()
                self.assertIs(owner.process, process)
                owner.close()
            stop.assert_called_once_with(process.pid, bridge.signal.SIGINT)
            self.assertIsNone(owner.process)

    def test_unrecognized_hardware_aborts_before_robot_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = bridge.O6ReplayBridge("can1", Path(directory) / "bridge.log")
            up = Mock(returncode=0, stdout="can1 UP <NOARP,UP>")
            process = FakeProcess()
            def launch(_command, **kwargs):
                kwargs["stdout"].write(
                    b"Warning: Hardware version number not recognized\n"
                    b"[INFO] O6 bridge ready: hand_type=left, can=can1\n"
                )
                return process
            with patch.object(bridge.subprocess, "run", return_value=up), \
                 patch.object(bridge, "existing_o6_bridges", return_value=[]), \
                 patch.object(bridge.subprocess, "Popen", side_effect=launch), \
                 patch.object(bridge.os, "killpg") as stop:
                with self.assertRaisesRegex(RuntimeError, "硬件版本未识别"):
                    owner.start(Mock())
            stop.assert_called_once_with(process.pid, bridge.signal.SIGINT)

    def test_racing_second_bridge_is_rejected_and_owned_process_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = bridge.O6ReplayBridge("can1", Path(directory) / "bridge.log")
            up = Mock(returncode=0, stdout="can1 UP <NOARP,UP>")
            process = FakeProcess()
            with patch.object(bridge.subprocess, "run", return_value=up), \
                 patch.object(bridge, "existing_o6_bridges",
                              side_effect=[[], [process.pid, 123]]), \
                 patch.object(bridge.subprocess, "Popen", return_value=process), \
                 patch.object(bridge.os, "killpg") as stop:
                with self.assertRaisesRegex(RuntimeError, "又出现 O6 桥接"):
                    owner.start(Mock())
            stop.assert_called_once_with(process.pid, bridge.signal.SIGINT)
            self.assertIsNone(owner.process)

    def test_can_down_never_opens_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = bridge.O6ReplayBridge("can1", Path(directory) / "bridge.log")
            down = Mock(returncode=0, stdout="can1 DOWN <NOARP>")
            with patch.object(bridge.subprocess, "run", return_value=down), \
                 patch.object(bridge.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "未启动"):
                    owner.start(Mock())
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
