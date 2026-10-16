# LinkerFFG Robot Hand Driver Module (ROS1)

LinkerFFG (O6/L7/L10/G20/R20/L25) robot hand ROS1 driver module, controls robot hands via serial port with real-time data glove mapping.

---

## Quick Start (Step-by-Step Guide)

### Step 1: Install

```bash
cd ~/catkin_ws/src
git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git
mv linkerhand_telop_python/ros1/src ./
rm -rf linkerhand_telop_python
cd ..
catkin_make install
source install/setup.bash
```

### Step 2: Connect Serial Port

Connect LinkerFFG robot hand to your PC via USB, then verify serial port permissions:

```bash
# Add current user to dialout group (requires re-login)
sudo usermod -a -G dialout $USER

# List serial devices
ls -l /dev/ttyUSB*
```

### Step 3: Configure Robot Hand Model

Edit `config/base_config.yml` to set robot hand model:

```yaml
system:
  motion_type: linkerforce    # Data glove type: linkerforce (required)
  robotname_r: l25            # Right hand model: o6 / l7 / l10 / g20 / r20 / l25
  robotname_l: l25            # Left hand model

serial:
  auto_scan: false            # Enable auto serial scan
  baudrates: [2000000, 460800, 1000000, 921600]  # 2000000 recommended
  left:
    port: /dev/ttyUSB1        # Left hand serial port
    baudrate: 460800          # Wireless: 460800, Wired: 2000000
  right:
    port: /dev/ttyUSB0        # Right hand serial port
    baudrate: 460800          # Wireless: 460800, Wired: 2000000
```

### Step 4: Launch

**Method 1: Run node directly**

```bash
rosrun linkerhand_retarget handretarget
```

**Method 2: Specify serial port (without modifying config file)**

```bash
rosrun linkerhand_retarget handretarget ports:="/dev/ttyUSB0,/dev/ttyUSB1"
```

### Step 5: Calibration (if needed)

```bash
rosrun linkerhand_retarget handretarget calibration:=auto_calibrate
```

During calibration, perform three gestures as prompted:
1. **Open hand** → hold for 5 seconds
2. **Make fist** → hold for 5 seconds
3. **O-pose** → hold for 5 seconds

---

## Robot Hand Models

| Model | DOF | Description |
|-------|-----|-------------|
| O6 | 6 | 6-DOF basic model |
| L7 | 7 | 7-DOF (thumb with roll) |
| L10 | 10 | 10-DOF industrial model |
| G20 | 20 | 20-DOF industrial model |
| R20 | 20 | 20-DOF research model |
| L25 | 25 | 25-DOF full-featured model |

`robotname_r` and `robotname_l` in config must match the actual connected robot hand models.

---

## Serial Connection Details

### Method 1: Command-line Port List (most common)

Suitable when multiple serial ports exist, system auto-detects left/right hand:

```bash
rosrun linkerhand_retarget handretarget ports:="/dev/ttyUSB0,/dev/ttyUSB1"
```

### Serial Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ports` | Candidate port list (comma-separated) | empty (use config file) |
| `baudrate` | Specified baudrate (overrides config) | use config file |
| `auto_scan` | Auto scan when preset fails | `false` |

---

## Calibration Details

### Calibration Config

In `config/base_config.yml`:

```yaml
calibration:
  show_fist: true       # Whether to show fist calibration step
  fist_extend_ratio: 0.5  # Fist extend ratio (only effective when show_fist=false)
```

### Start Calibration

```bash
rosrun linkerhand_retarget handretarget calibration:=auto_calibrate
```

### Calibration Process

1. **Open hand** → hold for 5 seconds (motor value 255)
2. **O-pose** → hold for 5 seconds (motor middle value)

If `show_fist: true`, a third step appears:

3. **Make fist** → hold for 5 seconds (motor value 0)

### Difference between show_fist=true and show_fist=false

| Item | `show_fist: true` | `show_fist: false` |
|------|-------------------|-------------------|
| Calibration steps | Open → O-pose → Fist (3 steps) | Open → O-pose (2 steps) |
| Fist data source | User actually performs fist gesture | Calculated from O-pose by ratio |
| Fist formula | N/A | `fist = opose + (opose - original) × fist_extend_ratio` |
| Mapping precision | Three-segment linear interpolation, most accurate | Two-segment interpolation, relies on extension |
| fist_extend_ratio | Not used | Controls extension ratio (default 0.5) |

### fist_extend_ratio Details

Only effective when `show_fist: false`, used to calculate fist value from O-pose:

- `fist_extend_ratio = 0.5` (default): O-pose extends 50% toward fist
- `fist_extend_ratio = 0.0`: fist value = O-pose value (no extension)
- `fist_extend_ratio = 1.0`: fist value = O-pose + full (O-pose - Open) extension
- Recommended range: `0.3 ~ 0.7`, adjust based on actual results

### Stability Detection

- Auto-detects gesture stability (variance < 0.03)
- Requires **5 seconds continuous stability** to complete
- Resets on instability, no timeout limit
- Terminal shows progress bar with real-time stability duration

### Calibration Data Storage

- Location: `motion/linkerforce/tmp/jointangle_data.tmp`
- Format: JSON
- Auto-loads preset sample data on first use
- Re-calibration overwrites old data

---

## ROS Topic Reference (ROS1)

### Published Topics

| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/cb_right_hand_control_cmd` | `sensor_msgs/JointState` | Right hand control data |
| `/cb_left_hand_control_cmd` | `sensor_msgs/JointState` | Left hand control data |

### Subscribed Topics

| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/hand_teleop_param` | `std_msgs/String` | Dynamic parameter control (JSON) |

### Topic Parameter Control (Runtime Dynamic Adjustment)

Dynamically adjust parameters via `/hand_teleop_param` topic without restarting the node.

```bash
# Enable mapper debug for all fingers
rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"mapper_debug\": true}"}' -1

# Use calibration data instead of glove data (for testing)
rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"force_glove_pose\": \"opose\"}"}' -1

# Restore real-time glove data
rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"force_glove_pose\": \"none\"}"}' -1
```

### Adjustable Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `mapper_debug` | Mapper debug switch | `true` / `false` / `["thumb_rotate"]` |
| `mapper_exp_factor` | Extrapolation factor | `2.0` (global) or `{"thumb_rotate": 2.0}` (per finger) |
| `mapper_scale_factor` | Scale factor | `1.5` (global) or `{"index_root_flexion": 1.5}` (per finger) |
| `force_glove_pose` | Force glove data source | `open` / `fist` / `opose` / `none` |

---

## Troubleshooting

### Cannot open serial port

```bash
# Check serial permissions
ls -l /dev/ttyUSB*
# Grant permissions
sudo chmod 666 /dev/ttyUSB0
```

### Robot hand not responding (Key Topic Monitoring)

Follow these steps in order:

**Step 1: Check if topics are published**

```bash
# List all related topics
rostopic list | grep cb_

# Check if data is being output (is the node publishing normally)
rostopic echo /cb_right_hand_control_cmd -n 1
rostopic echo /cb_left_hand_control_cmd -n 1
```

**Step 2: Check LinkerFFG joint control topics**

```bash
# Check if joint control commands have output
rostopic echo /cb_right_hand_control_cmd -n 1
rostopic echo /cb_left_hand_control_cmd -n 1

# Check topic frequency (should be around 50Hz)
rostopic hz /cb_right_hand_control_cmd
```

**Step 3: Enable debug output**

```bash
# Enable all debug
rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"mapper_debug\": true}"}' -1

# Observe terminal output, check if glove data is changing
# If data doesn't change, glove is not connected or topic is not published
```

**Step 4: Check calibration data**

```bash
# Check current calibration data in use
rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"force_glove_pose\": \"opose\"}"}' -1
# Observe if robot hand responds

rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"force_glove_pose\": \"fist\"}"}' -1
# Observe if robot hand grips

rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"force_glove_pose\": \"none\"}"}' -1
# Restore normal data source
```

**Step 5: Check serial connection**

1. Verify baudrate matches robot hand settings (wireless: 460800, wired: 2000000)
2. Try `auto_scan: true` for auto detection
3. Check robot hand power supply

**Step 6: Check logs**

```bash
# View node output
rosrun linkerhand_retarget handretarget
# Observe debug info in terminal output
```

**Quick Problem Identification**

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| No topic data | Glove not connected or topic name wrong | Check glove connection, verify topic name |
| Data always 0 or 255 | Calibration data abnormal | Re-calibrate or delete `tmp/jointangle_data.tmp` |
| Data fluctuates wildly | Serial signal interference | Check serial cable, use shielded cable |
| No terminal output | Node didn't start successfully | Check for errors, verify dependencies installed |
| Topic has data but robot hand doesn't move | Robot hand SDK not receiving commands | Check SDK connection, verify topic is subscribed |

---

## File Structure

```
motion/linkerforce/
├── config/              # Hand model configurations
│   ├── o6_config.py     # O6 config
│   ├── l6_config.py     # L6 config
│   ├── l7_config.py     # L7 config
│   ├── l10_config.py    # L10 config
│   ├── l20_config.py    # L20 config
│   ├── g20_config.py    # G20 config
│   └── o7_config.py     # O7 config
├── hand/                # Robot hand drivers
│   ├── linkerforce_o6.py
│   ├── linkerforce_l6.py
│   ├── linkerforce_l7.py
│   ├── linkerforce_l10.py
│   ├── linkerforce_l20.py
│   └── linkerforce_g20.py
├── tmp/                 # Temp files (calibration data, etc.)
├── retarget.py          # ROS1 integration
└── README.md            # English documentation
```