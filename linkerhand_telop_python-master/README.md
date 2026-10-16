# LinkerHand ROS Teleop SDK

ROS/ROS2 SDK for LinkerHand teleoperation and motion retargeting.

## Supported Devices

### Data Gloves and Robot Hand Models

| Data Glove | O6 | L7 | L10 | G20 | R20 | L25 | Module | Docs |
|------------|----|----|-----|-----|-----|-----|--------|------|
| LinkerFFG | Yes | Yes | Yes | Yes | Yes | Yes | `motion/linkerforce` | [Click here](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/linkerforce/README.md) |
| VTR-DYN | Yes | Yes | Yes | Yes | Yes | Yes | `motion/vtrdyn` | [Click here](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/vtrdyn/README.md) |
| LinkerTG | Yes | Yes | Yes | Yes | Yes | Yes | `motion/udexreal` | [Click here](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/udexreal/README.md) |
| LinkerMCG | Yes | Yes | Yes | Yes | Yes | Yes | `motion/linkermcg` | [Click here](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/linkermcg/README.md) |
| LinkerEG | Yes | Yes | Yes | Yes | Yes | Yes | `motion/linkereg` | [Click here](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/linkereg/README.md) |

### Data Glove ROS Version Support

| Data Glove | ROS1 Noetic Docs | ROS2 Docs |
|------------|------------------|-----------|
| LinkerFFG | [Click here](ros1/src/motion/linkerforce/README.md) | [Click here](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/linkerforce/README.md) |
| VTR-DYN | - | [Click here](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/vtrdyn/README.md) |
| LinkerTG | - | [Click here](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/udexreal/README.md) |
| LinkerMCG | - | [Click here](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/linkermcg/README.md) |
| LinkerEG | - | [Click here](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/linkereg/README.md) |

## ROS Version Compatibility

| ROS Version | Python | x86_64 | ARM64 | Windows |
|-------------|--------|--------|-------|---------|
| Noetic (ROS1) | 3.8 | Yes | Yes | - |
| Foxy (ROS2) | 3.8 | Yes | Yes | - |
| Humble (ROS2) | 3.10 | Yes | Yes | - |
| Jazzy (ROS2) | 3.12 | Yes | Yes | Yes |

## Directory Structure

```
linkerhand_telop_sdk/
├── ros1/src/              # ROS1 Noetic package
│   ├── linkerhand/        # Core library
│   ├── motion/            # Device drivers
│   ├── config/           # Configuration files
│   ├── scripts/          # Entry point scripts
│   └── assets/           # Robot model assets
│
├── ros2/src/              # ROS2 package
│   └── linkerhand_retarget/
│       └── linkerhand_retarget/
│           ├── handretarget.py  # Entry script
│           ├── linkerhand/      # Core library
│           ├── motion/          # Device drivers
│           ├── config/         # Configuration files
│           ├── launch/         # Launch files
│           ├── assets/          # Robot model assets
│           ├── resource/        # ROS2 resource
│           └── tests/          # Unit/integration tests
│
├── LICENSE                # MIT License
└── NOTICE                # Third-party dependencies notice
```

## Quick Start

### ROS2 Installation (git clone)

```bash
cd your_ros2_ws/src
git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git
mv linkerhand_telop_python/ros2/src/linkerhand_retarget ./
rm -rf linkerhand_telop_python
cd ..
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### ROS2 Installation (Download ZIP)

```bash
cd your_ros2_ws/src
wget -O linkerhand_telop_python.zip https://gitee.com/ericbrunt/linkerhand_telop_python/repository/archive/master.zip
unzip linkerhand_telop_python.zip
mv linkerhand_telop_python/ros2/src/linkerhand_retarget ./
rm -rf linkerhand_telop_python.zip linkerhand_telop_python/
cd ..
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### ROS1 Installation (git clone)

```bash
cd your_catkin_ws/src
git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git
mv linkerhand_telop_python/ros1/src ./
rm -rf linkerhand_telop_python
cd ..
catkin_make install
source install/setup.bash
```

### ROS1 Installation (Download ZIP)

```bash
cd your_catkin_ws/src
wget -O linkerhand_telop_python.zip https://gitee.com/ericbrunt/linkerhand_telop_python/repository/archive/master.zip
unzip linkerhand_telop_python.zip
mv linkerhand_telop_python/ros1/src ./
rm -rf linkerhand_telop_python.zip linkerhand_telop_python/
cd ..
catkin_make install
source install/setup.bash
```

## Configuration

Configuration file is located at `config/base_config.yml`, main options:

### system System Configuration

| Config | Description | Options |
|--------|-------------|---------|
| `motion_type` | Data glove type | `linkerforce`, `vtrdyn`, `udexreal`, `udexrealv2t`, `linkermcg`, `linkereg1`, `linkereg2` |
| `datasource_type` | Data source type | `motion` |
| `retargeting_type` | Retargeting type | `projection` |
| `robotname_r` | Right hand robot model | `o6`, `l6`, `l7`, `l10`, `l20`, `l21`, `l25`, `g20` |
| `robotname_l` | Left hand robot model | same as above |
| `motion_device` | Motion device ID (LinkerTG only) | e.g. `eric` |

### serial Serial Configuration (LinkerFFG)

| Config | Description | Default |
|--------|-------------|---------|
| `auto_scan` | Auto scan serial ports | `false` |
| `baudrates` | Candidate baudrate list | `[2000000, 460800, 1000000, 921600]` |
| `left.port` / `right.port` | Left/right hand serial path | e.g. `/dev/ttyUSB0` |
| `left.baudrate` / `right.baudrate` | Left/right hand baudrate | `460800` |
| `exclude_ports` | Ports to exclude | `[]` |
| `serial_debug` | Serial debug switch | `false` |

### udp Network Configuration (LinkerTG / LinkerMCG / VTR-DYN)

| Config | Description | Default |
|--------|-------------|---------|
| `ip` | Glove UDP server IP | `192.168.11.88` |
| `port` | Glove UDP port | `8888` |
| `serverport` | Local listening port | `5551` |

### calibration Calibration Configuration (LinkerFFG)

| Config | Description | Default |
|--------|-------------|---------|
| `show_fist` | Show fist calibration | `true` |
| `fist_extend_ratio` | Fist extend ratio | `0.5` |

### debug Debug Configuration

| Config | Description | Default |
|--------|-------------|---------|
| `mapper_debug` | Mapper debug switch | `false` |
| `joint_motor_debug_r` | Right hand joint debug | `false` |
| `joint_motor_debug_l` | Left hand joint debug | `false` |
| `joint_pub_debug` | Topic publish debug | `false` |

### humanset Human Pose Configuration (VTR-DYN)

| Config | Description |
|--------|-------------|
| `bodyfile` | Human pose file name (without extension) |
| `targethandfile` | Target hand info file name (without extension) |

### LinkerEG Specific Configuration

```yaml
linkereg:
  port: null        # Serial port path, null for auto scan
  password: "i"    # sudo password for auto fixing serial permissions
```

### Example Configuration

```yaml
system:
  motion_type: linkerforce    # Use LinkerFFG robot hand
  robotname_r: l25            # Right hand L25
  robotname_l: l25            # Left hand L25

serial:
  auto_scan: false
  baudrates: [2000000, 460800, 1000000, 921600]
  left:
    port: /dev/ttyUSB1
    baudrate: 460800
  right:
    port: /dev/ttyUSB0
    baudrate: 460800
```

## Repository

```bash
git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
