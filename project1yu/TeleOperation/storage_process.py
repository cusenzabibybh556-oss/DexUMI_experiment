# storage_process.py
import multiprocessing as mp
import h5py
import numpy as np
import os
from config import DATA_SAVE_DIR, EPISODE_IDX, HDF5_BUFFER_SIZE

def storage_process(data_queue: mp.Queue):
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(DATA_SAVE_DIR, f"episode_{EPISODE_IDX}.hdf5")
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    
    # 创建数据集（可扩展）
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('timestamps', (0,), maxshape=(None,), dtype='float64')
        f.create_dataset('target/joints', (0,6), maxshape=(None,6), dtype='float64')
        f.create_dataset('target/grip', (0,), maxshape=(None,), dtype='int32')
        f.create_dataset('actual/joints', (0,6), maxshape=(None,6), dtype='float64')
        f.create_dataset('actual/grip', (0,), maxshape=(None,), dtype='int32')
    
    buffer = []
    
    while True:
        try:
            data = data_queue.get(timeout=2.0)
        except:
            # 超时可能表示无新数据，检查是否需要退出（可增加退出标志共享内存）
            break
        
        buffer.append(data)
        if len(buffer) >= HDF5_BUFFER_SIZE:
            _flush_buffer(buffer, save_path)
            buffer.clear()
    
    # 最后写入剩余数据
    if buffer:
        _flush_buffer(buffer, save_path)

def _flush_buffer(buffer, save_path):
    timestamps, tgt_j, tgt_g, act_j, act_g = zip(*buffer)
    with h5py.File(save_path, 'a') as f:
        cur_len = f['timestamps'].shape[0]
        new_len = cur_len + len(buffer)
        f['timestamps'].resize((new_len,))
        f['target/joints'].resize((new_len,6))
        f['target/grip'].resize((new_len,))
        f['actual/joints'].resize((new_len,6))
        f['actual/grip'].resize((new_len,))
        
        f['timestamps'][cur_len:new_len] = timestamps
        f['target/joints'][cur_len:new_len] = np.array(tgt_j)
        f['target/grip'][cur_len:new_len] = tgt_g
        f['actual/joints'][cur_len:new_len] = np.array(act_j)
        f['actual/grip'][cur_len:new_len] = act_g