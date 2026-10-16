"""独立固定起点恢复入口的离线测试；不会连接或移动真机。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import replay_tron2_fixed_start as fixed
import restore_tron2_fixed_start as restore


class FakeClient:
    def __init__(self, *_args, **_kwargs):
        self.accid = fixed.FIXED_ROBOT_ID

    def wait_for_identity(self):
        pass

    def joint_snapshot(self, label):
        return fixed.FIXED_Q.copy(), np.zeros(14)

    def current_servo_pose(self, label):
        return Mock(
            left_position=fixed.FIXED_LEFT_POSITION.copy(),
            right_position=fixed.FIXED_RIGHT_POSITION.copy(),
        )


class RestoreOnlyTests(unittest.TestCase):
    def test_default_is_offline_and_does_not_connect(self):
        with patch.object(restore, "Tron2JsonClient", side_effect=AssertionError(
            "offline restore information must not connect"
        )):
            self.assertEqual(restore.main([]), 0)

    def test_execute_only_calls_fixed_start_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Mock()
            with patch.object(restore, "acquire_replay_lock", return_value=lock), \
                 patch.object(restore, "default_trace_path",
                              return_value=Path(directory) / "restore.jsonl"), \
                 patch.object(restore, "Tron2JsonClient", FakeClient), \
                 patch.object(restore, "restore_fixed_start",
                              return_value=Mock()) as prepare:
                code = restore.main(["--execute", "--countdown", "5"])
            self.assertEqual(code, 0)
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(prepare.call_args.args[4].countdown, 5)
            lock.close.assert_called_once()

    def test_wrong_robot_rejected_before_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            client.accid = "another_robot"
            with patch.object(restore, "acquire_replay_lock", return_value=Mock()), \
                 patch.object(restore, "default_trace_path",
                              return_value=Path(directory) / "restore.jsonl"), \
                 patch.object(restore, "Tron2JsonClient", return_value=client), \
                 patch.object(restore, "restore_fixed_start") as prepare:
                code = restore.main(["--execute"])
            self.assertEqual(code, 2)
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
