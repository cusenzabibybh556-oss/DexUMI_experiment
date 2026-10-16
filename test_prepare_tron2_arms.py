"""MoveJ 反馈超时恢复测试；不发送任何真机指令。"""

import unittest
from unittest.mock import patch

import numpy as np

import prepare_tron2_arms as prepare


class MoveJFeedbackTests(unittest.TestCase):
    def test_one_read_timeout_then_three_stable_samples_succeeds(self):
        class Client:
            trace = None

        target = np.zeros(14)
        stable = (target.copy(), np.zeros(14))
        with patch.object(prepare, "get_joint_state", side_effect=[
            TimeoutError("no response_get_joint_state within 4.0s"),
            stable, stable, stable,
        ]) as read, patch.object(prepare.time, "sleep"):
            actual = prepare.wait_for_target(Client(), target, slice(7, 14),
                                             timeout_s=1, tolerance=0.015)
        np.testing.assert_allclose(actual, target)
        self.assertEqual(read.call_count, 4)

    def test_no_feedback_never_counts_as_arrival(self):
        class Client:
            trace = None

        with patch.object(prepare, "get_joint_state",
                          side_effect=TimeoutError("no reply")):
            with self.assertRaisesRegex(RuntimeError, "是否完成未知"):
                prepare.wait_for_target(Client(), np.zeros(14), slice(7, 14),
                                        timeout_s=0.01, tolerance=0.015)


if __name__ == "__main__":
    unittest.main()
