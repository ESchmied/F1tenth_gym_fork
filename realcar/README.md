# F1TENTH ONNX Inference - Docker Deployment

This directory contains everything needed to deploy your trained F1TENTH model to a real RC car using ROS2 Foxy.

## Structure

```
realcar/
├── Dockerfile                          # Docker container setup
├── entrypoint.sh                       # Container entrypoint script
├── f1tenth_onnx_inference/            # ROS2 package
│   ├── f1tenth_onnx_inference/
│   │   ├── __init__.py
│   │   └── inference_node.py          # Main inference node
│   ├── launch/
│   │   └── inference.launch.py        # Launch file
│   ├── resource/
│   ├── package.xml                     # ROS2 package manifest
│   └── setup.py                        # Python package setup
└── README.md                           # This file
```

## Prerequisites

1. **Export your trained model to ONNX:**
   ```bash
   python export_onnx.py models/your_model.zip --deterministic --verify
   ```

2. **Copy the ONNX model to this directory:**
   ```bash
   cp models/your_model.onnx realcar/final_model.onnx
   ```

## Building the Docker Image

```bash
cd realcar
docker build -t f1tenth-onnx-inference:latest .
```

## Running on the F1TENTH Car

### Option 1: Docker Run (Basic)

```bash
docker run --rm -it \
  --network host \
  -v $(pwd)/final_model.onnx:/models/final_model.onnx:ro \
  f1tenth-onnx-inference:latest
```

### Option 2: Docker Run with Custom Parameters

```bash
docker run --rm -it \
  --network host \
  -v $(pwd)/final_model.onnx:/models/final_model.onnx:ro \
  f1tenth-onnx-inference:latest \
  ros2 launch f1tenth_onnx_inference inference.launch.py \
    model_path:=/models/final_model.onnx \
    inference_rate:=50.0 \
    max_speed:=5.0 \
    max_steering_angle:=0.4189
```

### Option 3: Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  f1tenth_inference:
    image: f1tenth-onnx-inference:latest
    network_mode: host
    volumes:
      - ./final_model.onnx:/models/final_model.onnx:ro
    environment:
      - ROS_DOMAIN_ID=0
    command: >
      ros2 launch f1tenth_onnx_inference inference.launch.py
        model_path:=/models/final_model.onnx
        inference_rate:=50.0
        max_speed:=5.0
        max_steering_angle:=0.4189
```

Then run:
```bash
docker-compose up
```

## ROS2 Topics

### Subscribed Topics:
- `/scan` (sensor_msgs/LaserScan): LiDAR data (1080 beams → subsampled to 12)
- `/odom` (nav_msgs/Odometry): Vehicle odometry (position, velocity)

### Published Topics:
- `/drive` (ackermann_msgs/AckermannDriveStamped): Autonomous driving commands

## Configuration Parameters

- `model_path`: Path to ONNX model file (default: `/models/final_model.onnx`)
- `inference_rate`: Inference frequency in Hz (default: `50.0`)
- `max_speed`: Maximum speed in m/s (default: `20.0`)
- `max_steering_angle`: Maximum steering angle in radians (default: `0.4189`)
- `scan_subsample_size`: Number of LiDAR beams after subsampling (default: `12`)
- `scan_original_size`: Original number of LiDAR beams (default: `1080`)

## Testing Without Hardware

You can test the node with simulated topics:

```bash
# Terminal 1: Run the inference node
docker run --rm -it --network host \
  -v $(pwd)/final_model.onnx:/models/final_model.onnx:ro \
  f1tenth-onnx-inference:latest

# Terminal 2: Publish fake scan data
ros2 topic pub /scan sensor_msgs/msg/LaserScan \
  "{header: {frame_id: 'laser'}, \
    angle_min: -2.35, angle_max: 2.35, \
    angle_increment: 0.00436, \
    range_min: 0.1, range_max: 30.0, \
    ranges: [5.0] * 1080}"

# Terminal 3: Publish fake odometry
ros2 topic pub /odom nav_msgs/msg/Odometry \
  "{pose: {pose: {position: {x: 0, y: 0, z: 0}, \
                  orientation: {w: 1, x: 0, y: 0, z: 0}}}, \
    twist: {twist: {linear: {x: 2.0, y: 0, z: 0}}}}"

# Terminal 4: Monitor drive commands
ros2 topic echo /drive
```

## Safety Notes

⚠️ **IMPORTANT SAFETY CONSIDERATIONS:**

1. **Start with low speeds**: Use `max_speed:=2.0` for initial testing
2. **Emergency stop**: Have a way to immediately stop the car (e.g., RC override)
3. **Test in safe environment**: Use a controlled space with safety barriers
4. **Monitor behavior**: Watch the car's behavior closely during initial runs
5. **Gradual increase**: Slowly increase max_speed as you verify safe operation

## Troubleshooting

### Model not loading
- Verify ONNX model path is correct
- Check model was exported with `--verify` flag
- Ensure model file is mounted in container

### No sensor data
- Check ROS2 topics: `ros2 topic list`
- Verify ROS_DOMAIN_ID matches: `echo $ROS_DOMAIN_ID`
- Check network connectivity in Docker

### Poor performance
- Reduce `inference_rate` if CPU usage is high
- Check sensor data quality with `ros2 topic echo /scan`
- Verify observation normalization matches training

### Steering oscillations
- Reduce `inference_rate` to smooth control
- Consider adding exponential smoothing to actions
- Check that `delta` (current steering) is being tracked correctly

## Development

To modify the inference node:

1. Edit `f1tenth_onnx_inference/f1tenth_onnx_inference/inference_node.py`
2. Rebuild the Docker image
3. Test changes

For local development without Docker:
```bash
cd f1tenth_onnx_inference
colcon build
source install/setup.bash
ros2 run f1tenth_onnx_inference inference_node
```

## License

MIT License - See main repository for details.
