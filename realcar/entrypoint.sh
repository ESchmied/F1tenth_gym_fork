#!/bin/bash
set -e

# Source ROS2 setup
source /opt/ros/foxy/setup.bash
source /ros2_ws/install/setup.bash

# Execute the command
exec "$@"
