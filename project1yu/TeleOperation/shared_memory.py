# shared_memory.py
import numpy as np
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory

class SharedData:
    def __init__(self, create=True, existing_shm_name=None):
        """
        :param create: True 表示由主进程创建新的共享内存块
        :param existing_shm_name: 如果 create=False，则传入已有共享内存名称（供子进程使用）
        """
        self.lock = Lock()
        if create:
            # 创建64字节的共享内存块：6 float64 (48) + 1 int32 (4) + 1 bool (1) = 53，补齐到64
            self.shm = SharedMemory(create=True, size=64)
            self._created = True
        else:
            self.shm = SharedMemory(name=existing_shm_name, create=False)
            self._created = False
        
        # 映射为 numpy 数组，方便读写
        self.joints = np.ndarray((6,), dtype=np.float64, buffer=self.shm.buf[0:48])
        self.grip = np.ndarray((1,), dtype=np.int32, buffer=self.shm.buf[48:52])
        self.exit_flag = np.ndarray((1,), dtype=np.bool_, buffer=self.shm.buf[52:53])
        
        # 初始化默认值（仅当创建时）
        if create:
            self.joints[:] = 0.0
            self.grip[0] = 1000   # 完全张开
            self.exit_flag[0] = False
    
    def get_shm_name(self):
        """返回共享内存名称，供子进程连接用"""
        return self.shm.name
    
    def close(self):
        """每个进程用完都应关闭自己的映射"""
        self.shm.close()
    
    def unlink(self):
        """只有主进程在最后调用，释放共享内存"""
        if self._created:
            self.shm.unlink()
    
    def __del__(self):
        self.close()