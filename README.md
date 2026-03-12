# My Changes
- **DreamerV3 Integration**: Full DreamerV3 training and evaluation pipeline
- **Sim-to-Real Transfer**: Seamless transfer from simulation to real F1TENTH car
- **Real Car Training**: Direct training on hardware via ROS2
- **Multi-track Support**: Random track selection for robust policy learning
- **Docker Support**: Containerized training for both simulation and real car

---

## Key Features

🏎️ **Seamless Sim-to-Real Transfer**
- Identical observation and action spaces between simulation and real car
- Automatic transferable observations (scan, velocity, steering) for sim-to-real compatibility
- Load simulation checkpoints directly on hardware

🚀 **DreamerV3 Integration**
- State-of-the-art model-based RL algorithm
- Efficient training with world models
- Support for all DreamerV3 model sizes (1M to 400M parameters)

🔄 **Multi-Track Training**
- Train on 22+ F1-scale tracks
- Smart map rotation: each environment sticks with one map for 20 resets before switching
- Random track selection for robust policies
- Track-specific or generalist agents

🐳 **Docker Support**
- Containerized training for reproducibility
- Separate containers for simulation and real car
- GPU acceleration support

🤖 **ROS2 Real Car Integration**
- Direct training on F1TENTH hardware
- Safety features: speed limits, collision detection
- Manual or automatic reset modes

## Table of Contents
1. [Simulation Training](#simulation-training)
2. [Real Car Training (Sim-to-Real)](#real-car-training-sim-to-real)
3. [Architecture](#architecture)
4. [Docker Setup](#docker-setup)

---

## Simulation Training

### Docker Setup for Simulation
```bash
# Build simulation container
docker build -f Dockerfile -t f1tenth_gymdock .

# Run with GPU support
docker run -it --rm \
  --gpus all \
  -v ~/logdir/docker:/root/logdir \
  -v ~/Downloads/f1tenth_gym:/home/f1tenth \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  f1tenth_gymdock:latest
```

### Training in Simulation
```bash
# Train on all tracks (random selection)
# All training uses transferable observations for sim-to-real compatibility
# Each environment sticks with one map for 20 resets before switching
python3 train_dreamerv3.py --model_size size12m --envs 100 --steps 1_000_000 --train_ratio 8.0 --replay.size 4000000

# Train on specific track
python3 train_dreamerv3.py --task Spielberg --model_size size12m --envs 100 --steps 1_000_000

# Longer training for better performance
python3 train_dreamerv3.py --model_size size12m --envs 50 --steps 5_000_000

# Adjust map switching frequency (default: 20 resets per map)
python3 train_dreamerv3.py --model_size size12m --envs 100 --resets_per_map 50
```

### Evaluation
```bash
# Evaluate on random tracks
python3 evaluate_dreamerv3.py /root/logdir/f1tenth/TIMESTAMP/ckpt/CHECKPOINT/ --render

# Evaluate on specific track
python3 evaluate_dreamerv3.py /root/logdir/f1tenth/TIMESTAMP/ckpt/CHECKPOINT/ --render --task Austin

python3 evaluate_dreamerv3.py /root/logdir/f1tenth/sim_20260311T141821/ckpt/20260311T144850F259730/ --render --task Austin

python3 evaluate_dreamerv3.py /logdir/real_20260310T120624/ckpt/20260310T225153F516088/ --render --task Austin
```

logdir/docker/f1tenth/sim_20260311T141821/ckpt/20260311T144850F259730
[text](../logdir/docker/f1tenth/sim_20260311T152533/ckpt/20260311T171056F733495)

---

## Real Car Training (Sim-to-Real)

### Prerequisites

**On the F1TENTH Car:**
```bash
# CRITICAL: Set ROS_LOCALHOST_ONLY=0 (add to ~/.bashrc)
export ROS_LOCALHOST_ONLY=0

# Start your car's ROS2 nodes (sensors, controllers, etc.)
# Ensure these topics are publishing:
#   /scan (sensor_msgs/LaserScan)
#   /odom (nav_msgs/Odometry)
#   /drive (ackermann_msgs/AckermannDriveStamped)
```

### Docker Setup for Real Car

```bash
# Build real car container from repository root
cd /path/to/f1tenth_gym_fork
docker build -t f1tenth-dreamer:latest -f Dockerfile.realcar .
```

### Training Workflow

#### Step 1: Train in Simulation (Sim Policy)

```bash
# Train with automatic sim-to-real compatible observations
python3 train_dreamerv3.py \
    --task None \
    --model_size size12m \
    --envs 50 \
    --steps 5_000_000

#trainig befehle für BA eval
#Train in sim with one map and env for comparability with MPC Supervisor. 
python3 train_dreamerv3.py \
    --task Austin \
    --model_size size12m \
    --envs 1 \
    --steps 3_000_000

#Train in real car sim with collision and reset through backup MPC couldn't make it work 
python3 train_dreamerv3.py --real \
    --max_speed 3.0 \
    --step_frequency 100.0 \
    --collision_threshold 0.1 \
    --logdir /logdir \
    --steps 3_000_000
    
    
#Train with BackupMPC
python3 train_dreamerv3.py --real \
    --max_speed 3.0 \
    --step_frequency 100.0 \
    --logdir /logdir \
    --steps 5_000_000 \
    --train_ratio 12.0
    

# Checkpoint saved to: ~/logdir/f1tenth/sim_TIMESTAMP/ckpt/
```

#### Step 2: Transfer to Real Car (Finetune on Hardware)
#für mich relevant

```bash

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  REQUIRED FLAGS: --net=host --ipc=host                                  │
# │  WARNING: Car must have ROS_LOCALHOST_ONLY=0                            │
# └─────────────────────────────────────────────────────────────────────────┘
#   
#ROS Topics dont get published outside of the container: 
#see https://github.com/eProsima/Fast-DDS/issues/5396
#The problem is due to the SHM transport. When you launch a Docker with --net=host and --ipc=host configuration, the publisher and subscriber detect
#that they are on the same host and try to use SHM. This is due to how the calculation of SHM usage was done until now. In addition, SHM uses the /dev
#shm directory, which in this case is being attempted to be shared by users who have different permissions: root in the case of Docker and $USER` in
#the case of the host. This causes communication to fail.
#-To fix this you have several quick solutions:
#-Have the same user on Docker as on the host.
#-Launch the ROS 2 application as root (equivalent to the above but without changing the Docker configuration).
#-Use UDP instead of SHM. This is as simple as running the following before launching the ROS 2 application in the Docker.

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

#-Do not set --ipc=host --net=host when running the Docker container.
```

```bash
docker run --rm -it \
    --net=host \
    --ipc=host \
    --gpus all \
    -v ~/logdir:/logdir \
    f1tenth-dreamer:latest bash
```
``` bash
#sim_ ... an aktuelle version anpassen bei bedarf

python3 train_dreamerv3.py --real \
    --from_checkpoint /logdir/real_20260310T120624/ckpt/20260310T225153F516088 \
    --max_speed 3.0 \
    --step_frequency 100.0 \
    --collision_threshold 0.3 \
    --logdir /logdir \
    --steps 200_000 \
    --train_ratio 12.0
```

#### Step 3: Monitor Training

```bash
# On your laptop/monitoring machine
tensorboard --logdir ~/logdir/f1tenth
```

### Real Car Safety

⚠️ **IMPORTANT SAFETY CONSIDERATIONS:**
- Start with low speeds: `--max_speed 1.0` initially
- Always have hardware e-stop ready
- Clear the training area of obstacles and people
- Never leave training unattended
- Test sensor connectivity before starting

### Direct Training (No Docker)

If you prefer running directly on the car:

```bash
# Install dependencies
pip3 install jax jaxlib elements embodied dreamerv3 gymnasium numpy transforms3d

# Train
python3 train_dreamerv3.py --real \
    --from_checkpoint /path/to/sim_checkpoint \
    --max_speed 3.0 \
    --steps 100_000
```

### Troubleshooting

**Cannot see ROS2 topics:**
```bash
# Verify topics are publishing
ros2 topic list
ros2 topic hz /scan
ros2 topic hz /odom

# Check ROS_LOCALHOST_ONLY
echo $ROS_LOCALHOST_ONLY  # Should be 0
```

**Training too slow:**
- Reduce `--train_ratio` (e.g., `--train_ratio 8`)
- Use smaller model (`--model_size size1m`)
- Increase `--step_frequency`

For more details, see [realcar/README.md](realcar/README.md).

---

## Architecture

### Training Features

**Random Starting Positions**
- Each episode, the car resets to a random position ANYWHERE along the raceline
- Not just the starting grid - can spawn at any point on the track
- Promotes better exploration and more robust policies
- Prevents overfitting to specific starting locations

**Episode Time Limit**
- 2000 steps per episode maximum
- Prevents infinite loops and ensures training progress
- Episodes end early on collision or time limit

**Map Rotation**
- When training on multiple tracks, each environment sticks with one map for 20 resets
- Then switches to a new random map
- Configurable via `--resets_per_map` argument

### Observation Space (Transferable - Default)
All training uses transferable observations for sim-to-real compatibility:

| Feature | Shape | Description | Source (Sim) | Source (Real) |
|---------|-------|-------------|--------------|---------------|
| `scan` | `(32,)` | LiDAR scan (subsampled from 1080) | Simulator | `/scan` topic |
| `linear_vel_x` | `()` | Forward velocity (m/s) | Simulator | `/odom` topic |
| `ang_vel_z` | `()` | Angular velocity (rad/s) | Simulator | `/odom` topic |
| `delta` | `()` | Steering angle (rad) | Tracked internally | Tracked internally |

### Action Space
| Action | Shape | Range | Target (Sim) | Target (Real) |
|--------|-------|-------|--------------|---------------|
| `action` | `(2,)` | `[[-0.42, -1.0], [0.42, max_speed]]` | Simulator | `/drive` topic |

**Format:** `[steering_angle, speed]`

### File Structure
```
f1tenth_gym_fork/
├── train_dreamerv3.py          # Main training script (sim & real)
├── f1tenth.py                  # Simulation environment wrapper
├── f1tenth_real.py             # Real car environment wrapper (ROS2)
├── evaluate_dreamerv3.py       # Evaluation script
├── Dockerfile                  # Simulation Docker image
├── Dockerfile.realcar          # Real car Docker image
├── f1tenth_gym/                # Core simulation package
└── realcar/                    # Real car packages
    ├── README.md               # Detailed real car documentation
    ├── Dockerfile              # Alternative lightweight Dockerfile
    ├── f1tenth_dreamer_training/   # ROS2 training package
    └── f1tenth_onnx_inference/     # ROS2 inference package
```

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMULATION TRAINING                          │
│                                                                 │
│  train_dreamerv3.py --steps 5M  (transferable obs automatic)    │
│         │                                                       │
│         ├─► F1Tenth (f1tenth.py)                                │
│         │      └─► f1tenth_gym (Gymnasium)                      │
│         │             └─► Multiple tracks, random selection     │
│         │                                                       │
│         └─► DreamerV3 Agent                                     │
│                └─► World Model + Policy                         │
│                                                                 │
│  Output: ~/logdir/f1tenth/sim_TIMESTAMP/ckpt/                   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ Transfer checkpoint
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REAL CAR TRAINING                            │
│                                                                 │
│  train_dreamerv3.py --real --from_checkpoint ...                │
│         │                                                       │
│         ├─► F1TenthReal (f1tenth_real.py)                       │
│         │      └─► ROS2 (--net=host --ipc=host)                │
│         │             ├─► Subscribe: /scan, /odom               │
│         │             └─► Publish: /drive                       │
│         │                                                       │
│         └─► DreamerV3 Agent (same architecture)                 │
│                └─► Loads sim weights, finetunes on real data    │
│                                                                 │
│  Output: ~/logdir/f1tenth/real_TIMESTAMP/ckpt/                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Docker Setup




![Python 3.8 3.9](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)
![Code Style](https://github.com/f1tenth/f1tenth_gym/actions/workflows/lint.yml/badge.svg)

# The F1TENTH Gym environment

This is a **fork** of the F1TENTH Gym environment with **DreamerV3 integration** and **sim-to-real transfer** capabilities.

✨ **New in this fork:**
- 🤖 DreamerV3 model-based RL training
- 🔄 Seamless sim-to-real transfer to physical F1TENTH cars
- 🐳 Docker containers for both simulation and real car deployment
- 🏁 Multi-track training on 22+ race circuits
- 📡 ROS2 integration for real hardware

This project is still under heavy development.

**Original F1TENTH Gym:** [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym)  
**Documentation:** [f1tenth-gym.readthedocs.io](https://f1tenth-gym.readthedocs.io/en/latest/)

## Quickstart
We recommend installing the simulation inside a virtualenv. You can install the environment by running:

```bash
virtualenv gym_env
source gym_env/bin/activate
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
pip install -e .
```

Then you can run a quick waypoint follow example by:
```bash
cd examples
python3 waypoint_follow.py
```

A Dockerfile is also provided with support for the GUI with nvidia-docker (nvidia GPU required):
```bash
docker build -t f1tenth_gym_container -f Dockerfile .
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_container
````
Then the same example can be ran.

## Known issues
- Library support issues on Windows. You must use Python 3.8 as of 10-2021
- On MacOS Big Sur and above, when rendering is turned on, you might encounter the error:
```
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```
You can fix the error by installing a newer version of pyglet:
```bash
$ pip3 install pyglet==1.5.11
```
And you might see an error similar to
```
gym 0.17.3 requires pyglet<=1.5.0,>=1.4.0, but you'll have pyglet 1.5.11 which is incompatible.
```
which could be ignored. The environment should still work without error.

## Citing
If you find this Gym environment useful, please consider citing:

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```

---

## Additional Citations

**DreamerV3 (Model-Based RL):**
```bibtex
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}
```

## License

This project maintains the original F1TENTH Gym license. See LICENSE file for details.

## Acknowledgments

- **F1TENTH Team** for the original gym environment
- **Danijar Hafner et al.** for DreamerV3
- **ROS2 Community** for robotic middleware
- **JAX Team** for high-performance computing framework

## Contributing

This fork adds sim-to-real capabilities. For contributions to the base F1TENTH Gym, please see the [original repository](https://github.com/f1tenth/f1tenth_gym).

## Support

For issues related to:
- **Real car training**: Open an issue with `[realcar]` tag
- **DreamerV3 integration**: Open an issue with `[dreamerv3]` tag  
- **Base simulation**: Refer to [original F1TENTH Gym](https://github.com/f1tenth/f1tenth_gym)
