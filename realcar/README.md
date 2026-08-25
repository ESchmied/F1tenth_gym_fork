# F1TENTH Real Car - DreamerV3 Training & Inference

This folder contains ROS2 packages for training and deploying DreamerV3 agents on real F1TENTH cars, enabling seamless sim-to-real transfer.

## Docker Quick Start

### Build the Container

**Recommended: Build from repository root** (includes all training scripts):

```bash
cd /path/to/f1tenth_gym_fork
docker build -t f1tenth-dreamer:latest -f Dockerfile.realcar .
```

Alternative: Build from realcar folder (requires mounting scripts at runtime):

```bash
cd /path/to/f1tenth_gym_fork/realcar
docker build -t f1tenth-dreamer:latest .
```

### Run Training on Real Car

```bash
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  IMPORTANT: Must use --net=host and --ipc=host for ROS2 communication!  │
# └─────────────────────────────────────────────────────────────────────────┘

docker run --rm -it \
    --net=host \
    --ipc=host \
    --gpus all \
    -v ~/logdir:/logdir \
    -v /path/to/sim_checkpoint:/checkpoints \
    f1tenth-dreamer:latest \
    python3 train_dreamerv3.py --real \
        --from_checkpoint /checkpoints/ckpt \
        --max_speed 3.0 \
        --logdir /logdir
```

### Run Inference Only

```bash
docker run --rm -it \
    --net=host \
    --ipc=host \
    -v /path/to/model.onnx:/models/model.onnx \
    f1tenth-dreamer:latest \
    ros2 launch f1tenth_onnx_inference inference.launch.py \
        model_path:=/models/model.onnx
```

### Interactive Shell

```bash
docker run --rm -it \
    --net=host \
    --ipc=host \
    --gpus all \
    -v ~/logdir:/logdir \
    f1tenth-dreamer:latest \
    bash
```

### Network Configuration Warning

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  WARNING: ROS_LOCALHOST_ONLY                                            │
│                                                                         │
│  The F1TENTH car MUST have ROS_LOCALHOST_ONLY=0 set!                    │
│  Otherwise, the container cannot communicate with the car's ROS2 nodes. │
│                                                                         │
│  On the car, either:                                                    │
│    • Run before starting ROS2: export ROS_LOCALHOST_ONLY=0              │
│    • Or add to ~/.bashrc: export ROS_LOCALHOST_ONLY=0                   │
│                                                                         │
│  Docker run flags (REQUIRED):                                           │
│    • --net=host    (share host network namespace)                       │
│    • --ipc=host    (share host IPC namespace)                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Overview

The real car integration provides:
- **Same interface as simulation**: The `F1TenthReal` environment wrapper matches the `F1Tenth` simulation interface exactly
- **Sim-to-real transfer**: Train in simulation, then finetune on the real car with the same codebase
- **Safety features**: Configurable speed limits, collision detection, and emergency stop
- **Flexible deployment**: Train directly on-car or export to ONNX for inference-only deployment

## Packages

### 1. `f1tenth_dreamer_training`
For training DreamerV3 agents on the real car.

### 2. `f1tenth_onnx_inference`
For deploying trained models as ONNX for inference-only (no training).

## Quick Start

### Prerequisites

```bash
# ROS2 (Humble or later)
# Source your ROS2 installation
source /opt/ros/humble/setup.bash

# Python dependencies
pip install numpy transforms3d

# DreamerV3 (for training) - install from source to avoid PyPI issues
# IMPORTANT: Keep the directory for editable install to work
git clone https://github.com/danijar/dreamerv3.git ~/dreamerv3
cd ~/dreamerv3
pip install -e .
cd ~

# JAX (GPU support recommended for training)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# TensorFlow (required for TensorBoard logging in elements)
pip install tensorflow-cpu
```

### Sim-to-Real Training Workflow

#### Step 1: Train in Simulation

```bash
# Train on random tracks for variety
python train_dreamerv3.py --task None --steps 5_000_000

# Or train on a specific track
python train_dreamerv3.py --task Austin --steps 5_000_000 --render
```

This creates a checkpoint at `~/logdir/f1tenth/sim_<timestamp>/`.

#### Step 2: Transfer to Real Car

```bash
# SSH into the car or run from the car's computer
# Make sure ROS2 is running and the car's sensors are publishing

# Start training on real car, loading simulation weights
python3 train_dreamerv3.py --real \
    --from_checkpoint ~/logdir/docker/f1tenth/sim_20260324T085419/ckpt/20260324T103948F511868 \
    --max_speed 3.0 \
    --step_frequency 100.0 \
    --collision_threshold 0.3 \
    --steps 100_000 \
    --train_ratio 12.0



python3 train_dreamerv3.py --real \
    --from_checkpoint /logdir/real_20260310T120624/ckpt/20260310T225153F516088 \
    --max_speed 3.0 \
    --step_frequency 100.0 \
    --collision_threshold 0.3 \
    --logdir /logdir \
    --steps 200_000 \
    --train_ratio 12.0
```

[text](../../logdir/docker/f1tenth/sim_20260324T085419/ckpt/20260324T103948F511868)
[text](../../logdir/docker/f1tenth/sim_20260324T085419)
[text](../../logdir/real_20260310T120624)


#### Step 3: Monitor Training

```bash
# In a separate terminal, monitor training progress
tensorboard --logdir ~/logdir/f1tenth
```

### Real Car Training Options

```bash
python train_dreamerv3.py --real \
    --from_checkpoint /path/to/sim_checkpoint \  # Load simulation weights
    --max_speed 3.0 \                            # Max speed in m/s (safety!)
    --step_frequency 20.0 \                      # Control frequency in Hz
    --collision_threshold 0.3 \                  # Min distance for collision (m)
    --scan_beams 32 \                            # Number of LiDAR beams
    --scan_topic /scan \                         # LiDAR topic
    --odom_topic /odom \                         # Odometry topic  
    --drive_topic /drive \                       # Drive command topic
    --steps 100_000 \                            # Training steps
    --model_size size12m                         # Model size
```

### Manual Reset Mode (Default)

During training, the environment will pause after each episode (collision) and prompt you to:
1. Stop the car
2. Place it back at a safe starting position
3. Press ENTER to continue

### Automatic Reset Mode (Optional)

If you have external pose tracking (e.g., Vicon, OptiTrack):

```bash
python train_dreamerv3.py --real \
    --automatic_reset \
    --pose_topic /vrpn_client_node/car/pose \
    --from_checkpoint /path/to/checkpoint
```

## Environment Interface

The `F1TenthReal` environment has the same interface as `F1Tenth`:

### Observation Space
| Key | Shape | Description |
|-----|-------|-------------|
| `scan` | `(scan_beams,)` | Subsampled LiDAR scan |
| `linear_vel_x` | `()` | Forward velocity (m/s) |
| `ang_vel_z` | `()` | Angular velocity (rad/s) |
| `delta` | `()` | Current steering angle (rad) |

### Action Space
| Key | Shape | Description |
|-----|-------|-------------|
| `action` | `(2,)` | `[steering_angle, speed]` |

This matches the simulation interface exactly, enabling seamless transfer.

## ROS2 Topics

### Subscribed
- `/scan` (sensor_msgs/LaserScan): LiDAR scan data
- `/odom` (nav_msgs/Odometry): Odometry data

### Published
- `/drive` (ackermann_msgs/AckermannDriveStamped): Drive commands

## Building the ROS2 Packages

```bash
# Navigate to your ROS2 workspace
cd ~/f1tenth_ws/src

# Symlink or copy the realcar folder
ln -s /path/to/f1tenth_gym_fork/realcar/f1tenth_dreamer_training .

# Build
cd ~/f1tenth_ws
colcon build --packages-select f1tenth_dreamer_training

# Source
source install/setup.bash
```

## Safety Considerations

⚠️ **IMPORTANT**: When training on a real car:

1. **Start with low speeds**: Use `--max_speed 1.0` initially
2. **Clear the area**: Ensure no obstacles or people nearby
3. **Emergency stop**: Keep the hardware e-stop ready
4. **Supervision**: Never leave training unattended
5. **Test first**: Run inference mode to verify sensor data before training
6. **Battery monitoring**: Check battery levels regularly

## Debugging

### Check sensor data is flowing:
```bash
ros2 topic echo /scan --once
ros2 topic echo /odom --once
```

### Check drive commands:
```bash
ros2 topic echo /drive
```

### Test environment connection:
```python
from f1tenth_real import F1TenthReal

# This will initialize ROS2 and wait for sensor data
env = F1TenthReal(max_speed=1.0)
obs = env.step({'reset': True})
print(f"Observation keys: {obs.keys()}")
print(f"Scan shape: {obs['scan'].shape}")
print(f"Velocity: {obs['linear_vel_x']}")
env.close()
```

## Troubleshooting

### "ROS2 dependencies not found"
Make sure you've sourced your ROS2 workspace and installed the required packages.

### "Timeout waiting for sensor data"
Check that the car's sensors are publishing:
```bash
ros2 topic list
ros2 topic hz /scan
ros2 topic hz /odom
```

### "Training is too slow"
- Reduce `--train_ratio` (e.g., `--train_ratio 8`)
- Use a smaller model (`--model_size size1m`)
- Increase step frequency (`--step_frequency 30`)

### "Car doesn't respond"
- Check `/drive` topic is being published
- Verify no other nodes are publishing to `/drive`
- Check the car's drive controller is running

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     train_dreamerv3.py                       │
│                    (DreamerV3 Training Loop)                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      F1TenthReal                             │
│                  (embodied.Env interface)                    │
│                                                              │
│   ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   │
│   │  obs_space  │   │  act_space   │   │    step()     │   │
│   │  (same as   │   │  (same as    │   │  (ROS2 I/O)   │   │
│   │   sim)      │   │   sim)       │   │               │   │
│   └─────────────┘   └──────────────┘   └───────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │    /scan    │    │    /odom    │    │   /drive    │
    │  (LiDAR)    │    │ (Odometry)  │    │ (Commands)  │
    └─────────────┘    └─────────────┘    └─────────────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Real F1TENTH  │
                    │      Car        │
                    └─────────────────┘
```

## License

MIT License - See main repository for details.
