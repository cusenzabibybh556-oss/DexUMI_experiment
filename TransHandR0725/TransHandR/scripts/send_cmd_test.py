import sys
from pathlib import Path

SDK_ROOT = Path("/home/zmy/Projects/Dexumi/project1yu/tele/linker_hand_python_sdk")
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from LinkerHand.linker_hand_api import LinkerHandApi
import time

hand = LinkerHandApi(hand_joint="O6", hand_type="right", can="can0")
hand.set_speed([80, 80, 80, 80, 80, 80])
hand.set_torque([20, 20, 20, 20, 20, 20])

open_pose = [254, 70, 200, 255, 255, 0]
for _ in range(10):
    hand.finger_move(open_pose)
    time.sleep(0.1)

print("status:", hand.get_state())
print("fault:", hand.get_fault())
