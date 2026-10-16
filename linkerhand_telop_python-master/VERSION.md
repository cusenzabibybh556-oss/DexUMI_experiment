# 版本更新说明

## v2.12.3 (2026-04-15)

- 优化进程退出机制：串口 close() 顺序调整，先关串口解除阻塞
- handretarget.py 移除 sys.exit(0)，让 ROS 清理流程完成
- retarget.py 添加 shutdown 回调和 runing 标志，确保串口线程正确停止
- .gitignore 更新：忽略 docs/ 和 scripts/ 目录

## v2.12.1 (2026-04-10)

- 修正所有手型 ROBOT_FIST_* 值使其与 URDF <limit upper> 一致
  - o6: 11元素修正
  - o7: 扩展到17元素
  - l10: 扩展到20元素
  - g20/l20: 21元素修正
- 改进 ROS1/ROS2 同步脚本：双向同步 + 删除多余文件功能
- 添加 Foxy/Jazzy ParameterDescriptor 兼容性（dynamic_typing 属性）
- 删除 ROS1/ROS2 多余文件（datamap.py, __init__.py, *.tmp, l10v6/l10v7/l25 handler）
- 修复 handretarget.py Foxy 兼容性问题
- L10 配置完善：添加 thumb_rotate/index_roll/ring_roll/pinky_roll 关节，更新 robot_idx 和 ROBOT_OPOSE/ORIGINAL 值
- L10 handler 添加 _resolve_version_config 函数支持 v1/v2 版本配置

## v2.12.0 (2026-04-07)

- 剔除 L10v6 手型支持（配置文件/实现文件/枚举/映射分支）
- 修复 L10 右手小指侧摆映射索引（arc_value[4] → arc_value[16]）
- 修复 L10 pinky_roll robot_idx（17 → 16）
- 修复 L10 ring_roll robot_idx（13 → 12）
- 添加 L10 ROBOT_ORIGINAL_LEFT[16] = -0.35 小指侧摆初始值
- 修复 G20 侧摆映射配置
- 完善 L10/L10v6/G20 手型配置，添加 ROBOT_ORIGINAL 和 ROBOT_FIST
- 恢复 MOTOR_CONSTRAINTS，使用配置替代 URDF limits
- 统一所有 hand 类 set_glove_version 方法（参考 L6 实现）
- 修复 G20 重复 LeftHand 类
- 删除所有 retarget.py 中的 l10v6 分支
- 串口连接状态日志改为 Warn
- 权限失败提示修复命令
- ROS1 run_calibration 连接检测
- 手套未连接时中止标定
- 修复 retarget.py 缩进错误
- ROS1/ROS2 代码全量同步（6个 hand 文件 + 配置文件）
- 支持手型更新：移除 l10v6
- 修复 G20 index_roll 延伸映射导致输出异常
- 修复 G20 LeftHand 使用 trans_to_motor_right 错误
- 添加 mapper 调试显示手套原始数据
- 调整 thumb_rotate/thumb_abduction 权重
- 同步 G20 配置到 L20（关闭侧摆延伸映射）
- 修复 linkerg/linkermcg process 方法缩进错误
- 修复 linkermcg runing 拼写错误
- udexreal/vtrdyn 添加 G20 手型支持
- 修复 udexreal/vtrdyn 左手型判断错误

## v2.11.8 (2026-03-26)

- 修复手套未连接时继续标定 bug
- 添加 tmp 目录 .gitkeep

## v2.11.7 (2026-03-26)

- 同步 ROS1 l6_config 到 ROS2
- 添加电机输出约束功能
- 优化延伸映射参数
- 更新标定样本数据

## v2.11.6 (2026-03-24)

- 修复标定结束后未正确加载历史数据到 hand 对象的问题
- 同步 ROS1/ROS2 修复

## v2.11.5 (2026-03-24)

- v1/v2 版本区分支持（weights, reverse_motion 字典格式）
- 延伸映射只在 ['original', 'opose'] 模式触发
- 移除自动拟合功能
- 添加 MULTI_SEGMENT_CONFIG_FROZEN 配置冻结
- 更新 README 标定配置建议

## v2.11.4 (2026-03-21)

- 修复 ROS1 标定保存时 ROS2 API 混入问题
- 修复 SerialConnection 缺少 isdebug 属性
- 串口读取错误日志改为受 serial_debug 控制

## v2.11.3 (2026-03-21)

- 串口读取错误日志改为受 serial_debug 控制
- 增加串口初始化等待时间提升连接稳定性

## v2.11.2 (2026-03-21)

- 删除废弃型号：l18, l16, t25, l21, l30
- 新增 l20lite 型号（使用 l10 配置）
- 清理 retarget.py 中的废弃代码引用
- 优化串口配置：波特率 2000000 优先

## v2.11.1 (2026-03-21)

- 全手型配置优化，统一配置结构
- 新增 l10v6 手型支持
- 修复 ROS1 兼容性问题
- 支持无设备模式使用标定数据

## v2.11.0 (2026-03-21)

- 映射器多段映射优化
- 话题参数扩展，支持动态调整映射参数
- 标定稳定性检测

## v2.10.2 (2026-03-20)

- 修复候选端口检测问题
- 添加串口连接使用说明

## v2.10.1 (2026-03-20)

- 支持命令行串口参数
- 优化设备未连接提示

## v2.10.0 (2026-03-20)

- 映射器调试开关
- 指定手指调试
- 优化 G20 扩展映射参数

## v2.9.0 (2026-03-18)

- 动态模式切换
- 状态反馈话题
- 样本标定数据
- 串口配置管理
- 串口调试开关
- G20 映射优化
- 架构重构

## 支持手型

o6, l6, o7, l7, l10, l20, g20, l25
