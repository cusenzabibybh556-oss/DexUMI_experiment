"""Control-source arbitration test; does not instantiate ROS or open CAN."""

import unittest
from types import SimpleNamespace

from linkerhand_retarget.command_limits import CommandLimiter
from linkerhand_retarget.o6_bridge import O6BridgeNode
from sensor_msgs.msg import JointState


class FakeBridge:
    _accept_command = O6BridgeNode._accept_command
    _clamp_byte = staticmethod(O6BridgeNode._clamp_byte)

    def __init__(self):
        self.control_mode = "live"
        self.command_limiter = CommandLimiter([80, 0, 0, 0, 0, 0], [255] * 6)
        self.latest_pose = None
        self.recv_count = 0
        self._log_error_periodically = lambda *_: None
        self.logger = SimpleNamespace(warn=lambda *_: None)

    def get_logger(self):
        return self.logger


class O6BridgeReplayModeTests(unittest.TestCase):
    def test_glove_and_replay_are_mutually_exclusive(self):
        bridge = FakeBridge()
        live = JointState()
        live.position = [255.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        replay = JointState()
        replay.position = [250.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        O6BridgeNode._on_command(bridge, live)
        self.assertEqual(bridge.latest_pose, [255, 1, 2, 3, 4, 5])

        response = O6BridgeNode._set_replay_mode(
            bridge, SimpleNamespace(data=True), SimpleNamespace()
        )
        self.assertTrue(response.success)
        O6BridgeNode._on_command(bridge, live)
        self.assertEqual(bridge.latest_pose, [255, 1, 2, 3, 4, 5])
        O6BridgeNode._on_replay_command(bridge, replay)
        self.assertEqual(bridge.latest_pose, [250, 6, 7, 8, 9, 10])
        self.assertFalse(O6BridgeNode._set_replay_mode(
            bridge, SimpleNamespace(data=True), SimpleNamespace()
        ).success)

        O6BridgeNode._set_replay_mode(
            bridge, SimpleNamespace(data=False), SimpleNamespace()
        )
        O6BridgeNode._on_command(bridge, live)
        self.assertEqual(bridge.latest_pose, [250, 6, 7, 8, 9, 10])
        self.assertTrue(O6BridgeNode._resume_live(
            bridge, None, SimpleNamespace()
        ).success)
        O6BridgeNode._on_command(bridge, live)
        self.assertEqual(bridge.latest_pose, [255, 1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
