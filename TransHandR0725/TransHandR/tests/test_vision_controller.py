import ast
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_first_inspire_command_does_not_crash_control_process():
    source = (ROOT / "scripts/main_visionpro_multi_hand.py").read_text(encoding="utf-8")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "real_robot_process")
    controllers = []

    class FakeController:
        def __init__(self, config, serial_port):
            self.init_success = False
            self.last_command = None
            self.sent = []
            controllers.append(self)

        def init(self):
            self.init_success = True

        def send_angles(self, angles):
            self.sent.append(angles.copy())
            self.last_command = [1] * 6
            return True

        def read_state(self):
            return [1] * 6

        def close(self):
            pass

    class StopAfterOneCycle:
        def __init__(self):
            self.cycles = 0

        def is_set(self):
            self.cycles += 1
            return self.cycles > 1

    namespace = {"InspireController": FakeController, "time": time}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "vision_controller", "exec"), namespace)
    angles = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    namespace["real_robot_process"](
        {"angles_inspire": angles}, StopAfterOneCycle(), "inspire", object(),
        serial_port="unused", telemetry_port=0,
    )
    assert len(controllers) == 1
    assert len(controllers[0].sent) == 1
    np.testing.assert_array_equal(controllers[0].sent[0], angles)
