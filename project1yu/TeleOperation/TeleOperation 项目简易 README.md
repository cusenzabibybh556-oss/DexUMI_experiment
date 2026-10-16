## TeleOperation 项目简易 README

### 1. 项目简介
这是一个基于多进程的机械臂远程操控与数据采集系统。  
它通过操纵杆读取目标关节、夹爪状态，控制机械臂和灵巧手，并把控制/实际数据保存到文件中。

### 2. 关键组件

- teleop_collect_hdf5.py
  - 入口程序
  - 创建共享内存 `SharedData`
  - 启动三个子进程：
    - `joystick_process`：读取操纵杆输入，写入共享内存
    - `control_process`：从共享内存读取目标命令，控制机械臂/手，并把数据放入队列
    - `storage_process`：从队列读取数据并写入磁盘

- shared_memory.py
  - 负责跨进程共享：6 个关节目标、1 个夹爪目标、1 个退出标志

- config.py
  - 各类参数配置：机械臂 IP/端口、操纵杆串口、频率、数据保存目录等

---

### 3. 运行方式

直接运行：

```bash
python main.py
```

当前代码在 Windows 下有多进程保护，适合直接执行。

---

### 4. 数据保存说明

- 保存目录：`datas/`
- 保存文件：`episode_0.hdf5`
- 当前配置里 `EPISODE_IDX = 0`，所以每次启动同名文件会被重新创建/覆盖
- 数据格式：HDF5（二进制格式）

保存的数据字段包括：

- `timestamps`：时间戳
- `target/joints`：目标关节角度，6 维
- `target/grip`：目标夹爪值
- `actual/joints`：实际关节角度，6 维
- `actual/grip`：实际夹爪值（当前代码中实际夹爪写入的是目标夹爪）

---

### 5. 当前问题/改进建议

- `episode_0.hdf5` 会被覆盖
  - 如果想保留多次运行的数据，需要把 `EPISODE_IDX` 改成自动递增或按时间命名
- 如果需要更可读的格式，可以改成 `txt` / `csv`
  - 但目前代码写的是 HDF5，适合大数据批量保存

---

### 6. 备注

- storage_process.py 里面使用 `h5py.File(..., 'w')` 初始化文件，所以每次启动都会重新创建文件
- 写入逻辑是先缓存 `HDF5_BUFFER_SIZE` 条数据，再一次性写入，降低 I/O 频率

---