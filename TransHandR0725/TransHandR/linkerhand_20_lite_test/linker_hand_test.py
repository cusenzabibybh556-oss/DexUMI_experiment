# python
from linkerhand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi
import time

# 初始化 API（右手，L21，CAN 接口 can0）
hand = LinkerHandApi(hand_joint="L21", hand_type="right", can="can0")

# hand.open_can()
# 可选：设置速度/力矩（根据设备与模型可设置）
hand.set_speed(speed=[120]*5)        # 高层接口，会映射到底层实现
hand.set_torque(torque=[200]*5)

# # 构造一个长度为25的姿态（示例值，0~255）
# pose = [128] * 25

# # 发送到手（高层封装，内部会调用 L21 的 CAN 发送）
# hand.finger_move(pose=pose)

# 等待或循环发送
while True:
    for i in range(10):
        start = time.time()
        pose = [i*20] * 25
        hand.finger_move(pose=pose)
        print(f"send_time: {time.time()-start} s")
        time.sleep(0.1)
# pose = [0, 0, 0, 0, 0, 0, 
#         0, 0, 0, 0, 0, 
#         255, 255, 255, 255,
#         0, 
#         255, 255, 255, 255, 
#         0, 0, 0, 0, 0]
# pose = [255, 255, 255, 255, 255, 255, 
#         0, 0, 255, 0, 0, 
#         255, 255, 255, 255,
#         255, 
#         255, 255, 255, 255, 
#         255, 255, 255, 255, 255]
# pose = [226, 120, 110, 80, 87, 75, 216, 253, 254, 0, 3, 255, 255, 255, 255, 5, 255, 255, 255, 255, 3, 217, 248, 246, 181]
# pose = [254, 116, 106, 80, 87, 141, 214, 252, 254, 0, 242, 255, 255, 255, 255, 101, 255, 255, 255, 255, 5, 214, 246, 246, 181]
    hand.finger_move(pose=pose)
# 
    time.sleep(0.1)

# 关闭 CAN（可选）
# hand.close_can()
# 关闭CAN可以在终端中：sudo ip link set can0 down