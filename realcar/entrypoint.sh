#!/bin/bash
set -e

# =============================================================================
# F1TENTH DreamerV3 Container Entrypoint
# =============================================================================

# Source ROS2 setup
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

# Add f1tenth_gym to Python path
export PYTHONPATH=/f1tenth_gym:$PYTHONPATH

# Print warning about network configuration
echo ""
echo "============================================================"
echo "  F1TENTH DreamerV3 Training Container"
echo "============================================================"
echo ""
echo "  IMPORTANT: This container requires host networking!"
echo ""
echo "  If ROS2 topics are not visible, ensure:"
echo "    1. Container was started with: --net=host --ipc=host"
echo "    2. Car has: export ROS_LOCALHOST_ONLY=0"
echo ""
echo "============================================================"
echo ""

# Check if we can see ROS2 topics (quick sanity check)
if command -v ros2 &> /dev/null; then
    echo "Checking ROS2 connectivity..."
    timeout 3 ros2 topic list > /dev/null 2>&1 && echo "ROS2 topics accessible!" || echo "Warning: Cannot list ROS2 topics. Check network configuration."
    echo ""
fi

# Execute the command
exec "$@"
