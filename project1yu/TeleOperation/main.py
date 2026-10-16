# main.py
import multiprocessing as mp
import signal
import sys
from shared_memory import SharedData
from joystick_process import joystick_process
from control_process import control_process
from storage_process import storage_process
from config import QUEUE_MAXSIZE

def main():
    # 1. 创建共享内存（主进程创建）
    shared = SharedData(create=True)
    shm_name = shared.get_shm_name()
    
    # 2. 创建队列
    data_queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
    
    # 3. 启动子进程，传递 shm_name 而不是整个 SharedData 对象（避免序列化问题）
    p_joy = mp.Process(target=joystick_process, args=(shm_name,))
    p_ctrl = mp.Process(target=control_process, args=(shm_name, data_queue))
    p_store = mp.Process(target=storage_process, args=(data_queue,))
    
    p_joy.start()
    p_ctrl.start()
    p_store.start()
    
    # 4. 信号处理
    def signal_handler(*_):
        print("\n收到退出信号，通知所有进程结束...")
        with shared.lock:
            shared.exit_flag[0] = True
        # 等待几秒后强制结束
        p_joy.join(timeout=2)
        p_ctrl.join(timeout=2)
        p_store.join(timeout=2)
        for p in [p_joy, p_ctrl, p_store]:
            if p.is_alive():
                p.terminate()
        # 清理共享内存
        shared.close()
        shared.unlink()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 主进程等待子进程结束
    p_joy.join()
    p_ctrl.join()
    p_store.join()
    
    # 正常结束清理
    shared.close()
    shared.unlink()
    print("所有进程已结束，数据保存完成。")

if __name__ == "__main__":
    # 多进程在 Windows 上必须加这个保护
    main()
