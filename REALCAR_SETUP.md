# Real Car Setup Guide - Quick Reference

This document provides a quick reference for setting up and using the F1TENTH real car training system.

## 🚀 Quick Start (3 Steps)

### 1. Train in Simulation

```bash
# All training automatically uses transferable observations (sim-to-real compatible)
python3 train_dreamerv3.py \
    --task Austin \
    --model_size size12m \
    --envs 100 \
    --max_speed 10.0 \
    --steps 1_000_000
```

**Output:** `~/logdir/f1tenth/sim_TIMESTAMP/ckpt/`

### 2. Build Docker Container

```bash
cd /path/to/f1tenth_gym_fork
docker build -t f1tenth-dreamer:latest -f Dockerfile.realcar .
```

**Note:** The build installs DreamerV3 from GitHub (not PyPI) to avoid package issues. This takes a few minutes but ensures everything works correctly.

### 3. Deploy on Real Car

**On the F1TENTH car, set:**
```bash
export ROS_LOCALHOST_ONLY=0  # Add to ~/.bashrc
```

**Run training:**
```bash
docker run --rm -it \
    --net=host \
    --ipc=host \
    --gpus all \
    -v ~/logdir:/logdir \
    -v ~/logdir/f1tenth/sim_TIMESTAMP:/checkpoints \
    f1tenth-dreamer:latest \
    python3 train_dreamerv3.py --real \
        --from_checkpoint /checkpoints/ckpt \
        --max_speed 3.0 \
        --steps 100_000
```

---

## 🎬 Evaluation (Simulation & Real Car)

The evaluation script works identically for both simulation and real car, with diagnostic plots.

### Simulation Evaluation
```bash
# From simulation checkpoint
python3 evaluate_dreamerv3.py ~/logdir/f1tenth/sim_TIMESTAMP/ckpt \
    --render \
    --episodes 5

# Specific track
python3 evaluate_dreamerv3.py ~/logdir/f1tenth/sim_TIMESTAMP/ckpt \
    --task Spielberg \
    --render \
    --episodes 3
```

### Real Car Evaluation
```bash
docker run --rm -it \
    --net=host \
    --ipc=host \
    --gpus all \
    -v ~/logdir:/logdir \
    f1tenth-dreamer:latest \
    python3 evaluate_dreamerv3.py /logdir/f1tenth/sim_TIMESTAMP/ckpt \
        --real \
        --max_speed 3.0 \
        --episodes 5 \
        --plot_dir /logdir/eval_plots
```

**Output:**
- Episode statistics (reward, length)
- Diagnostic plots for each episode:
  - Reward over time
  - Actions (steering, speed commands)
  - Observations (velocity, yaw rate, steering angle)
  - Command vs actual (steering, speed)
  - LiDAR scan heatmap
  - Polar scan snapshots
- Summary statistics and plots

---

## 📋 Checklist Before Real Car Training

### Prerequisites
- [ ] F1TENTH car with ROS2 running
- [ ] Topics publishing: `/scan`, `/odom`
- [ ] Topic subscribing: `/drive`
- [ ] `ROS_LOCALHOST_ONLY=0` set on car
- [ ] Checkpoint from simulation training available
- [ ] Training area cleared of obstacles
- [ ] Hardware e-stop accessible

### Safety Checks
- [ ] Start with low speed (`--max_speed 1.0`)
- [ ] Test sensor connectivity first
- [ ] Monitor first episode closely
- [ ] Have emergency stop ready

---

## 🔍 Common Issues & Solutions

### Issue: Cannot see ROS2 topics

**Solution:**
```bash
# Check topics are publishing
ros2 topic list
ros2 topic hz /scan

# Verify ROS_LOCALHOST_ONLY
echo $ROS_LOCALHOST_ONLY  # Should output: 0

# Ensure container uses --net=host --ipc=host
```

### Issue: Training is too slow

**Solution:**
```bash
# Reduce training ratio
--train_ratio 8

# Use smaller model
--model_size size1m

# Increase step frequency
--step_frequency 30
```

### Issue: Car moves erratically

**Possible causes:**
- Speed too high (reduce `--max_speed`)
- Sensor data quality issues (check `/scan` topic)
- Time synchronization issues (check `--step_frequency`)
- Checkpoint trained on different observation space (ensure both sim and real use same version)

---

## 🎯 Recommended Training Parameters

### Conservative (First Run)
```bash
--max_speed 1.0 \
--step_frequency 10.0 \
--collision_threshold 0.5
```

### Standard
```bash
--max_speed 3.0 \
--step_frequency 20.0 \
--collision_threshold 0.3
```

### Aggressive (After Testing)
```bash
--max_speed 5.0 \
--step_frequency 30.0 \
--collision_threshold 0.2
```

---

## 📁 File Locations

### Key Files
- **Simulation wrapper:** `f1tenth.py`
- **Real car wrapper:** `f1tenth_real.py`
- **Training script:** `train_dreamerv3.py`
- **Evaluation script:** `evaluate_dreamerv3.py` (works for both sim & real)
- **Real car Docker:** `Dockerfile.realcar`
- **Documentation:** `realcar/README.md`

### Docker Images
- **Simulation:** `f1tenth_gymdock:latest`
- **Real car:** `f1tenth-dreamer:latest`

### Checkpoints
- **Simulation:** `~/logdir/f1tenth/sim_TIMESTAMP/`
- **Real car:** `~/logdir/f1tenth/real_TIMESTAMP/`

---

## 🔧 ROS2 Topics

### Subscribed (Input)
| Topic | Type | Description |
|-------|------|-------------|
| `/scan` | `sensor_msgs/LaserScan` | LiDAR scan (1080 beams → 32 subsampled) |
| `/odom` | `nav_msgs/Odometry` | Velocity and pose |

### Published (Output)
| Topic | Type | Description |
|-------|------|-------------|
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | Steering & speed commands |

---

## 💡 Tips & Best Practices

1. **Start with simulation**: Always train a baseline policy in simulation first
2. **Automatic sim-to-real**: All training uses transferable observations automatically
3. **Test incrementally**: Start with low speeds and increase gradually
4. **Monitor TensorBoard**: Watch training progress in real-time
5. **Save frequently**: Checkpoints are saved automatically every N steps
6. **Backup checkpoints**: Copy simulation checkpoints before real car training
7. **Log everything**: The container logs sensor data and actions automatically

---

## 📞 Support

For detailed documentation:
- **Main README:** [README.md](README.md)
- **Real car details:** [realcar/README.md](realcar/README.md)
- **F1TENTH Gym docs:** https://f1tenth-gym.readthedocs.io/

---

## ⚠️ Critical Reminders

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. Always use: --net=host --ipc=host                                  │
│  2. Car must have: ROS_LOCALHOST_ONLY=0                                │
│  3. Start with low speeds for safety                                    │
│  4. Never leave training unattended                                     │
│  5. Hardware e-stop should always be accessible                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
