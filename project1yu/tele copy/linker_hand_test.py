# python
from linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi
import time
import numpy as np
import os

# ======================== 默认关节配置 ========================
可在此处直接调整默认的自定义关节值
DEFAULT_POSE = [128, 255, 255, 255, 255,
                0, 255, 255, 255, 255,
                255, 255, 255, 255, 255,
                128, 255, 255, 255, 255,
                0, 255, 255, 255, 255]  # 默认全255，可直接修改此处

# DEFAULT_POSE = [255, 255, 255, 255, 255,
#                 255, 255, 255, 255, 255,
#                 ]  

# ======================== 读取关节序列文件 ========================
def load_joint_sequences(base_path="/home/ub/TeleOperation_DataCollection"):
    """
    读取关节序列文件
    
    Args:
        base_path: 文件所在的基础路径
        
    Returns:
        decoded_joints: 解码后的关节序列 (N, 25)
        input_joints: 输入关节序列 (M, 25)
    """
    decoded_path = os.path.join(base_path, "decoded_joints_raw255.npy")
    input_path = os.path.join(base_path, "input_joints_raw255.npy")
    
    if not os.path.exists(decoded_path):
        print(f"✗ 文件不存在: {decoded_path}")
        return None, None
    
    if not os.path.exists(input_path):
        print(f"✗ 文件不存在: {input_path}")
        return None, None
    
    try:
        decoded_joints = np.load(decoded_path)
        input_joints = np.load(input_path)
        
        print(f"✓ 成功加载解码关节序列: {decoded_joints.shape}")
        print(f"✓ 成功加载输入关节序列: {input_joints.shape}")
        
        return decoded_joints, input_joints
    except Exception as e:
        print(f"✗ 读取文件失败: {e}")
        return None, None


# ======================== 传输关节序列给灵巧手 ========================
def play_joint_sequence(hand, joint_sequence, sequence_name="Unknown", frame_rate=30):
    """
    将关节序列逐帧传输给灵巧手
    
    Args:
        hand: LinkerHandApi 实例
        joint_sequence: 关节序列数组 (N, 25)
        sequence_name: 序列名称（用于日志输出）
        frame_rate: 播放帧率 (Hz)
    """
    if joint_sequence is None:
        print(f"✗ 序列为空，无法播放")
        return
    
    num_frames = joint_sequence.shape[0]
    frame_interval = 1.0 / frame_rate
    
    print(f"\n播放 {sequence_name}:")
    print(f"  总帧数: {num_frames}")
    print(f"  帧率: {frame_rate} Hz")
    print(f"  预计时长: {num_frames * frame_interval:.2f} 秒")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        for frame_idx in range(num_frames):
            frame_start = time.time()
            
            # 获取当前帧的关节角度 (25维)
            pose = joint_sequence[frame_idx].astype(np.uint8)
            
            # 发送到灵巧手
            hand.finger_move(pose=pose.tolist())
            
            # 控制帧率
            elapsed = time.time() - frame_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # 打印进度
            if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
                total_elapsed = time.time() - start_time
                print(f"  已播放: {frame_idx + 1}/{num_frames} 帧 "
                      f"({(frame_idx + 1) / num_frames * 100:.1f}%) "
                      f"耗时: {total_elapsed:.2f}s")
    
    except KeyboardInterrupt:
        print(f"\n✗ 播放被中断")
    except Exception as e:
        print(f"\n✗ 播放过程中出错: {e}")
    
    total_time = time.time() - start_time
    print(f"✓ 播放完成，总耗时: {total_time:.2f}s\n")


# ======================== 主程序 ========================
if __name__ == "__main__":
    # 初始化 API（右手，L21，CAN 接口 can0）
    print("初始化灵巧手...")
    hand = LinkerHandApi(hand_joint="L21", hand_type="right", can="can0")
    
    # 设置速度和力矩
    hand.set_speed(speed=[120]*5)
    hand.set_torque(torque=[200]*5)
    print("✓ 灵巧手初始化完成\n")
    
    # 运行模式选择
    print("=" * 60)
    print("请选择运行模式:")
    print("1. 循环关节输入 (原始功能)")
    print("2. 播放解码关节序列 (decoded_joints_raw255.npy)")
    print("3. 播放输入关节序列 (input_joints_raw255.npy)")
    print("4. 依次播放两个序列")
    print("5. 自定义关节控制 (单帧)")
    print("=" * 60)
    
    try:
        mode = input("请输入选项 (1/2/3/4/5): ").strip()
        
        if mode == "1":
            # ======================== 原始循环关节输入功能 ========================
            print("\n启动循环关节输入模式...")
            print("按 Ctrl+C 停止\n")
            
            while True:
                for i in range(10):
                    start = time.time()
                    pose = [i*20] * 25
                    hand.finger_move(pose=pose)
                    send_time = time.time() - start
                    print(f"send_time: {send_time:.4f} s")
                    time.sleep(0.1)
        
        elif mode == "2":
            print("\n选择: 播放解码关节序列")
            decoded_joints, _ = load_joint_sequences()
            if decoded_joints is not None:
                play_joint_sequence(hand, decoded_joints, "解码关节序列", frame_rate=30)
            else:
                print("✗ 关节序列加载失败")
        
        elif mode == "3":
            print("\n选择: 播放输入关节序列")
            _, input_joints = load_joint_sequences()
            if input_joints is not None:
                play_joint_sequence(hand, input_joints, "输入关节序列", frame_rate=30)
            else:
                print("✗ 关节序列加载失败")
        
        elif mode == "4":
            print("\n选择: 依次播放两个序列")
            decoded_joints, input_joints = load_joint_sequences()
            
            if decoded_joints is not None and input_joints is not None:
                play_joint_sequence(hand, decoded_joints, "解码关节序列", frame_rate=30)
                
                print("等待3秒后播放输入序列...")
                time.sleep(3)
                
                play_joint_sequence(hand, input_joints, "输入关节序列", frame_rate=30)
            else:
                print("✗ 关节序列加载失败")
        
        elif mode == "5":
            # ======================== 自定义关节控制 (单帧) ========================
            print("\n启动自定义关节控制模式...")
            print("=" * 60)
            print(f"默认关节值: {DEFAULT_POSE[:5]}... (全 {DEFAULT_POSE[0]})")
            print("关节范围: 0-255 (每个关节独立设置)")
            print("输入格式选项:")
            print("  1. 直接按Enter使用默认关节值")
            print("  2. 输入25个数字，用逗号分隔，如 128,128,128,...,128")
            print("快速命令:")
            print("  all 100   - 所有关节设置为100")
            print("  zero      - 所有关节设置为0")
            print("  max       - 所有关节设置为255")
            print("  default   - 使用默认关节值")
            print("  quit/exit - 退出此模式")
            print("=" * 60 + "\n")
            
            while True:
                try:
                    user_input = input("输入关节值或命令 (按Enter使用默认值): ").strip().lower()
                    
                    # 按Enter直接使用默认值
                    if user_input == "":
                        pose = DEFAULT_POSE.copy()
                        print(f"使用默认关节值: {pose[:5]}...")
                    
                    elif user_input in ["quit", "exit"]:
                        print("✓ 退出自定义控制模式")
                        break
                    
                    # 快速命令处理
                    elif user_input.startswith("all "):
                        try:
                            value = int(user_input.split()[1])
                            if 0 <= value <= 255:
                                pose = [value] * 25
                            else:
                                print("✗ 数值必须在0-255之间")
                                continue
                        except (IndexError, ValueError):
                            print("✗ 格式错误: all <数值>")
                            continue
                    
                    elif user_input == "zero":
                        pose = [0] * 25
                    
                    elif user_input == "max":
                        pose = [255] * 25
                    
                    elif user_input == "default":
                        pose = DEFAULT_POSE.copy()
                        print(f"使用默认关节值: {pose[:5]}...")
                    
                    else:
                        # 解析用户输入的关节值
                        values = user_input.split(',')
                        if len(values) != 25:
                            print(f"✗ 需要25个关节值，您输入了 {len(values)} 个")
                            continue
                        
                        pose = []
                        valid = True
                        for i, val_str in enumerate(values):
                            try:
                                val = int(val_str.strip())
                                if 0 <= val <= 255:
                                    pose.append(val)
                                else:
                                    print(f"✗ 关节 {i} 的值 {val} 超出范围 (0-255)")
                                    valid = False
                                    break
                            except ValueError:
                                print(f"✗ 关节 {i} 的值 '{val_str}' 无法转换为整数")
                                valid = False
                                break
                        
                        if not valid:
                            continue
                    
                    # 发送关节值到灵巧手
                    print(f"发送关节值: {pose[:5]}... (共25个)")
                    start = time.time()
                    hand.finger_move(pose=pose)
                    send_time = time.time() - start
                    print(f"✓ 发送完成，耗时: {send_time:.4f} s\n")
                
                except KeyboardInterrupt:
                    print("\n✓ 自定义控制模式已退出")
                    break
                except Exception as e:
                    print(f"✗ 发送出错: {e}\n")
        
        else:
            print("✗ 无效的选项")
    
    except KeyboardInterrupt:
        print("\n程序被中断")
    
    except Exception as e:
        print(f"✗ 程序出错: {e}")
    
    finally:
        print("\n关闭灵巧手...")
        # hand.close_can()  # 可选
        print("✓ 程序结束")