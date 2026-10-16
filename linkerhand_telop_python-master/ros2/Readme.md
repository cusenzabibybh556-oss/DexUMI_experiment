# 1. LinkerFFG Glove

## 1.1 Product Introduction
For specific product introduction, please refer to the calibration example description contained within
Attachment 1: Linker FFG (FFG01) Product Instruction Manual

## 1.2 Software Usage
This product provides ROS1 and ROS2 SDK packages under Ubuntu, working together with Linkerhand's ROS1 SDK and ROS2 SDK to achieve teleoperation actions
For specific usage, please jump to Chapter 4: Teleoperation Retargeting System

## 1.3 Communication Protocol
For specific product introduction, please refer to Attachment 2: Linker FFG (FFG01) Communication Protocol

---

# 2. Teleoperation SDK System

## 2.1 System Introduction
This system's retargeting program is developed based on the ROS1/ROS2 platform, consisting of one node, four publisher topics, and two subscription topics
The Linker FFG glove pushes data to the retargeting program via serial port protocol

### 2.1.1 Download Address
https://github.com/linker-bot/linkerhand-ros-teleo
Please contact customer service to obtain the latest SDK

### 2.1.2 Installing the Teleoperation Retargeting Program
The teleoperation retargeting program is provided in src form, therefore requiring users to compile it themselves for use on different platforms. The teleoperation retargeting program provides a ROS2 version; the ROS1 version is still under development

Copy src to the workspace, then compile
```bash
colcon build --symlink-install
```

There may be permission issues during the process preventing execution; use the following command
```bash
chmod 777 -R *
```

### 2.1.3 Program Configuration
Enter the ls command in this directory
```bash
cd src/linkerhand_retarget/linkerhand_retarget/config
ls -1
```
After entering the above command, you should normally enter the config directory. After `ls -1`, you should see the following list
```
base_config.yml
body_custom_pose.yml
body_unity_pose.yml
hand_config.yml
human_hand_info.yml
linker_hand_info.yml
model_config.yml
retarget_config.yml
speed_config.yml
```

The only configuration file we need to modify is `base_config.yml`; other configuration files must not be modified.

The following is an introduction to the configuration content in `base_config.yml`:

**1. Configure Robot Hand Type**
```yaml
robotname_r: l10
robotname_l: l10
```
- LinkerHand_L10 series hands should be configured as: `l10`
- LinkerHand_L20 series hands should be configured as: `l20`
- LinkerHand_O7 series hands should be configured as: `o7`
- LinkerHand_T25 series hands should be configured as: `t25`
- LinkerHand_L25 series hands should be configured as: `l25`
- LinkerHand_L6 series hands should be configured as: `l6`
- LinkerHand_O6 series hands should be configured as: `o6`

**Note:** All the above must be in lowercase; uppercase will not be correctly recognized.

**2. Configure Data Printing**
```yaml
debug:
  joint_pub_debug: false
  joint_motor_debug_r: true
  joint_motor_debug_l: false
```
- `joint_pub_debug`: Set to true to print the content after Topic Pub, refresh rate is slow, can display left and right hands simultaneously
- `joint_motor_debug_r`: Set to true to quickly refresh the output data of the right hand, range 255-0, representing the change from open to closed
- `joint_motor_debug_l`: Set to true to quickly refresh the output data of the left hand, range 255-0, representing the change from open to closed

**3. Configure Data Source**
```yaml
motion_type: linkerforce
```
- `motion_type`: When using the LinkerFFG glove, set to `linkerforce`, `motion_device` does not need to be set

### 2.1.4 Program Launch (ROS2 Version)
Next, execute the following command to confirm the USB device list on your device. Then insert the 2 force-feedback gloves, execute the command again, and confirm the refreshed USB device numbers. Take `/dev/ttyUSB0` and `/dev/ttyUSB1` as examples:
```bash
ls /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1
```

Then execute
```bash
# There are two launch modes,
# One with calibration. Must be executed for the first time without calibration
ros2 run linkerhand_retarget handretarget --ros-args -p calibration := True
# One normal launch, calibration not executed by default
ros2 run linkerhand_retarget handretarget 
```

**Notes:**
For customer convenience, a startup script is provided in the same directory as `src`, named `startup_linkerforce.sh`

```bash
#!/bin/bash
# LinkerForce Glove Teleoperation Startup Script

# Set serial port permissions
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1

# source ROS workspace (please enter the correct directory)
source ~/project/linkerhand_telop_ws/ros2/v2.8.7/install/setup.bash

# Launch program (forced calibration)
ros2 run linkerhand_retarget handretarget --ros-args -p calibration:=True

# Launch without calibration
# ros2 run linkerhand_retarget handretarget 
```

After editing, launch it like this
```bash
./startup_linkerforce.sh
# If you encounter Tab not recognized, grant permissions
sudo chmod a+x ./startup_linkerforce.sh
```

### 2.1.5 Program Launch (ROS1 Version)
Next, execute the following command to confirm the USB device list on your device. Then insert the 2 force-feedback gloves, execute the command again, and confirm the refreshed USB device numbers. Take `/dev/ttyUSB0` and `/dev/ttyUSB1` as examples:
```bash
ls /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1
```

Then execute
```bash
# There are two launch modes,
# One with calibration. Must be executed for the first time without calibration
rosrun ros_linkerhand_retarget handretarget.py _calibrate:=true
# One normal launch, calibration not executed by default
rosrun ros_linkerhand_retarget handretarget.py
```

**Notes:**
For customer convenience, a startup script is provided in the same directory as `src`, named `startup_linkerforce.sh`

```bash
#!/bin/bash
# LinkerForce Glove Teleoperation Startup Script

# Set serial port permissions
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1

# source ROS workspace (please enter the correct directory)
source ~/linkerhand_telop/ros1/v2.8.6/install/setup.bash

# Launch program (forced calibration)
rosrun ros_linkerhand_retarget handretarget.py _calibrate:=true

# Launch without calibration
# rosrun ros_linkerhand_retarget handretarget.py
```

After editing, launch it like this
```bash
./startup_linkerforce.sh
# If you encounter Tab not recognized, grant permissions
sudo chmod a+x ./startup_linkerforce.sh
```

## 2.2 Topic Description
The working principle of the SDK is to clean and organize the data transmitted from the upper computer, then distribute the data uniformly via Topics. Therefore, Topics are the data transfer stations and an important basis for judging whether the teleoperation is working properly.

- ROS1 command: `rostopic list`
- ROS2 command: `ros2 topic list`

**1. ROS2 command:**
```bash
ros2 topic echo /cb_left_hand_control_cmd
# or
ros2 topic echo /cb_right_hand_control_cmd
```

### Data Description

#### L25 Robotic Hand (21 DOF)
| Joint | Finger | Action | Default Value |
|-------|--------|--------|---------------|
| joint1 | Thumb | Proximal Flexion | 255 |
| joint2 | Index | Proximal Flexion | 255 |
| joint3 | Middle | Proximal Flexion | 255 |
| joint4 | Ring | Proximal Flexion | 255 |
| joint5 | Pinky | Proximal Flexion | 255 |
| joint6 | Thumb | Abduction | 255 |
| joint7 | Index | Abduction | 128 |
| joint8 | Middle | Abduction | 128 |
| joint9 | Ring | Abduction | 128 |
| joint10 | Pinky | Abduction | 128 |
| joint11 | Thumb | Rotation | 255 |
| joint12 | Reserved | - | - |
| joint13 | Reserved | - | - |
| joint14 | Reserved | - | - |
| joint15 | Reserved | - | - |
| joint16 | Thumb | Distal Flexion | 255 |
| joint17 | Index | Distal Flexion | 255 |
| joint18 | Middle | Distal Flexion | 255 |
| joint19 | Ring | Distal Flexion | 255 |
| joint20 | Pinky | Distal Flexion | 255 |
| joint21 | Thumb | Tip Flexion | 255 |
| joint22 | Index | Tip Flexion | 255 |
| joint23 | Middle | Tip Flexion | 255 |
| joint24 | Ring | Tip Flexion | 255 |
| joint25 | Pinky | Tip Flexion | 255 |

#### L20 Robotic Hand (16 DOF)
| Joint | Finger | Action | Default Value |
|-------|--------|--------|---------------|
| joint1 | Thumb | Proximal Flexion | 255 |
| joint2 | Index | Proximal Flexion | 255 |
| joint3 | Middle | Proximal Flexion | 255 |
| joint4 | Ring | Proximal Flexion | 255 |
| joint5 | Pinky | Proximal Flexion | 255 |
| joint6 | Thumb | Abduction | 255 |
| joint7 | Index | Abduction | 128 |
| joint8 | Middle | Abduction | 128 |
| joint9 | Ring | Abduction | 128 |
| joint10 | Pinky | Abduction | 128 |
| joint11 | Thumb | Rotation | 255 |
| joint12 | Reserved | - | - |
| joint13 | Reserved | - | - |
| joint14 | Reserved | - | - |
| joint15 | Reserved | - | - |
| joint16 | Thumb | Tip Flexion | 255 |
| joint17 | Index | Tip Flexion | 255 |
| joint18 | Middle | Tip Flexion | 255 |
| joint19 | Ring | Tip Flexion | 255 |
| joint20 | Pinky | Tip Flexion | 255 |

#### L10 Robotic Hand (10 DOF)
| Joint | Finger | Action | Default Value |
|-------|--------|--------|---------------|
| joint1 | Thumb | Proximal Flexion | 255 |
| joint2 | Thumb | Abduction | 128 |
| joint3 | Index | Proximal Flexion | 255 |
| joint4 | Middle | Proximal Flexion | 255 |
| joint5 | Ring | Proximal Flexion | 255 |
| joint6 | Pinky | Proximal Flexion | 255 |
| joint7 | Index | Abduction | 128 |
| joint8 | Ring | Abduction | 128 |
| joint9 | Pinky | Abduction | 128 |
| joint10 | Thumb | Rotation | 255 |

#### O7 Robotic Hand (7 DOF)
| Joint | Finger | Action | Default Value |
|-------|--------|--------|---------------|
| joint1 | Thumb | Proximal Flexion | 255 |
| joint2 | Thumb | Abduction | 128 |
| joint3 | Index | Proximal Flexion | 255 |
| joint4 | Middle | Proximal Flexion | 255 |
| joint5 | Ring | Proximal Flexion | 255 |
| joint6 | Pinky | Proximal Flexion | 255 |
| joint7 | Thumb | Rotation | 255 |

#### O6/L6 Robotic Hand (7 DOF)
| Joint | Finger | Action | Default Value |
|-------|--------|--------|---------------|
| joint1 | Thumb | Proximal Flexion | 255 |
| joint2 | Thumb | Abduction | 128 |
| joint3 | Index | Proximal Flexion | 255 |
| joint4 | Middle | Proximal Flexion | 255 |
| joint5 | Ring | Proximal Flexion | 255 |
| joint6 | Pinky | Proximal Flexion | 255 |

## 2.3 Troubleshooting Self-Check

### 2.3.1 Network Issues
The teleoperation and control devices are connected via the network. Network abnormalities can cause devices to malfunction. First, check if the network cable connections are correct, ensure that both IPs can ping each other, and confirm that the corresponding ports are open on the firewall.

### 2.3.2 Topic Anomalies
Under normal network conditions, you can confirm whether topics exist on other network devices using commands.
Refer to section 2.2 Topic Description for details.

## 2.4 Mapping Parameter Settings

### 2.4.1 Principle
The mapping system has 3 modes

| Mapping Mode | English Equivalent | Chinese Equivalent | Applicable Scenario |
|--------------|--------------------|--------------------|---------------------|
| Three-State Mapping | origin-opose-fist | Open Hand-O Pose-Fist | Aims to balance the maximum range of the robotic hand while satisfying pinch gestures. Suitable for most mapping requirements. |
| Dual-End Mapping | origin-fist | Open Hand-Fist | Maximizes the full range of the robotic hand to the greatest extent. |
| Pinch Mapping | origin-opose- | Open Hand-O Pose | Targets pinch gestures primarily, aiming to improve granularity for grasping actions. |

### 2.4.2 Configuration Files
This mapping system configures parameter binding for the robotic hand. The mapping configuration directory is located at:
```
~/src/linkerhand_retarget/linkerhand_retarget/motion/linkerforce/config
```

Currently provided mapping parameter packages are as follows:
```
g20_config.py
l10_config.py
l20_config.py
l6_config.py
o6_config.py
o7_config.py
```

### 2.4.3 Configuration Introduction
The following uses `l6_config.py` as an example.

#### 2.4.3.1 Finger Configuration Constant `FINGER_CONFIGS`

##### 2.4.3.1.1 Common Constant Table
(Content omitted)

##### 2.4.3.1.2 `reverse_motion` Attribute
Reverse signal: Default is `False`

##### 2.4.3.1.3 `dynamic_weight` Attribute
Dynamic weight: Used for detailed expression within intervals. Default is `None`, not enabled in this version.

##### 2.4.3.1.4 `extended_mapping` Attribute
Extended mapping configuration
```python
'extended_mapping': {    # New extended mapping configuration
    'enabled': True,
    'scale_factor': 1.2,  # Scaling factor, default 1.0
    'extended_exp_factor': 80  # New: Extension index parameter
}
```
- `enabled`: Enabled only for pinch mapping; `False` means disabled.
- `scale_factor`: Scaling factor, used to quickly lock onto the O-Pose position.
- `extended_exp_factor`: Extension parameter, used for continuation actions after the O-Pose. A smaller value makes it harder to reach the mechanical limits of the robotic hand, allowing for more detailed expression of grasping. A larger value makes it easier to quickly reach the drive joint limit positions.

#### 2.4.3.2 Mapping Order `MAPPING_ORDER`
Taking L6 as an example, indicates the priority order of processed joints
```python
MAPPING_ORDER = [
    'thumb_abduction', 'thumb_flexion', 
    'index', 'middle', 'ring', 'pinky'
]
```

#### 2.4.3.3 State Configuration `STATE_CONFIG` (Not Enabled)

#### 2.4.3.4 Robotic Hand OPOSE Calibration Position: `ROBOT_OPOSE_LEFT`
Taking the L6 configuration as an example, the following angles correspond to the OPOSE position of the left robotic hand. Note that the following order corresponds to the URDF sequence.
```python
ROBOT_OPOSE_LEFT = [
    1.4, 0.54, 0.0, 0.38, 0.0, 0.38, 0.0, 0.38, 0.0, 0.38, 0.0
]
```

#### 2.4.3.5 Robotic Hand OPOSE Calibration Position: `ROBOT_OPOSE_RIGHT`
Taking the L6 configuration as an example, the following angles correspond to the OPOSE position of the right robotic hand. Note that the following order corresponds to the URDF sequence.
```python
ROBOT_OPOSE_RIGHT = [
    1.4, 0.54, 0.0, 0.38, 0.0, 0.38, 0.0, 0.38, 0.0, 0.38, 0.0
]
```

### 2.4.4 Robotic Hand Calibration
Sections 2.4.3.4 and 2.4.3.5 introduced the OPOSE calibration positions for the left and right robotic hands respectively. These are used to handle discrepancies between the current robotic hand position and the URDF. When using for the first time, modifications need to be made in the following code.

Path: `./src/linkerhand_retarget/linkerhand_retarget/motion/linkerforce/hand`

Taking the L6 robotic hand as an example, after opening `linkerforce_l6.py`, around line 149:
```python
if self.calibrationoriginal is not None 
and self.calibrationfistpose is not None 
and self.calibrationopose is not None:
    # for i in range(20):
    #     self.multi_state_mapper.debug_value[i] = joint_arc[i]
    # arc_value = self.multi_state_mapper.map_glove_to_robot(joint_arc)   # Comment out
    arc_value = ROBOT_OPOSE_RIGHT          # Enable calibration
    qpos[17] = self.g_jointpositions_arc[1] = arc_value[0]
    qpos[20] = self.g_jointpositions_arc[0] = arc_value[1]
    qpos[1] = self.g_jointpositions_arc[2] = arc_value[3]
    qpos[9] = self.g_jointpositions_arc[3] = arc_value[5]
    qpos[13] = self.g_jointpositions_arc[4] = arc_value[7]
    qpos[5] = self.g_jointpositions_arc[5] = arc_value[9]
```

Rewrite it as shown in the example above to enable calibration for the right robotic hand. After launching, the robotic hand will be fixed at the current angle according to the mapped angle.

The corresponding section for the left hand is around line 360:
```python
if self.calibrationoriginal is not None and self.calibrationfistpose is not None and self.calibrationopose is not None:
    # for i in range(20):
    #     self.multi_state_mapper.debug_value[i] = joint_arc[i]
    # arc_value = self.multi_state_mapper.map_glove_to_robot(joint_arc) # Comment out
    # Enable calibration
    arc_value = ROBOT_OPOSE_LEFT
    qpos[17] = self.g_jointpositions_arc[1] = arc_value[0]
    qpos[20] = self.g_jointpositions_arc[0] = arc_value[1]
    qpos[1] = self.g_jointpositions_arc[2] = arc_value[3]
    qpos[9] = self.g_jointpositions_arc[3] = arc_value[5]
    qpos[13] = self.g_jointpositions_arc[4] = arc_value[7]
    qpos[5] = self.g_jointpositions_arc[5] = arc_value[9]
```

Rewrite it as shown in the example above to enable calibration for the right robotic hand. After launching, the robotic hand will be fixed at the current angle according to the mapped angle.

Once both left and right hands have reached the desired pinch position, you can restore the original state and use the teleoperation system normally.