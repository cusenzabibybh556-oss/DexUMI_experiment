import pytest

from linkerhand_retarget.motion.linkerforce.retarget import Retarget


class FakeReader:
    def __init__(self):
        self.commands = []
        self.stopped = False

    def send_force_feedback(self, values):
        self.commands.append(list(values))

    def stop(self):
        self.stopped = True


class FakeLogger:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass


class FakeNode:
    def get_logger(self):
        return FakeLogger()


@pytest.mark.parametrize("enabled", [False, True])
def test_inspire_stop_clears_only_selected_glove(enabled):
    retarget = Retarget.__new__(Retarget)
    left = FakeReader()
    right = FakeReader()
    retarget.node = FakeNode()
    retarget.calibration_profile = "inspire"
    retarget.active_hand_side = "left"
    retarget.enable_force_feedback = enabled
    retarget.force_reader_left = left
    retarget.force_reader_right = right
    retarget.feedback_last_sent = {"left": [10.0] * 5, "right": [10.0] * 5}
    retarget.feedback_last_write_ns = {"left": 0, "right": 0}

    retarget.stop_serial_threads()

    assert left.commands == [[0.0] * 5]
    assert right.commands == []
    assert left.stopped and right.stopped


def test_disabled_feedback_initial_clear_is_not_repeated():
    retarget = Retarget.__new__(Retarget)
    reader = FakeReader()
    retarget.node = FakeNode()
    retarget.feedback_last_sent = {"left": None}
    retarget.feedback_last_write_ns = {"left": 0}

    retarget._clear_inspire_feedback("left", reader)
    retarget._clear_inspire_feedback("left", reader)

    assert reader.commands == [[0.0] * 5]
