# Dockerfile for F1TENTH ONNX inference on ROS2 Foxy
FROM ros:foxy-ros-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=foxy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-foxy-ackermann-msgs \
    ros-foxy-sensor-msgs \
    ros-foxy-nav-msgs \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for ONNX inference
# Note: Using standard onnxruntime from pip (supports ONNX IR v9)
RUN pip3 install --upgrade pip && \
    pip3 install \
    onnxruntime==1.16.3 \
    numpy==1.24.3 \
    transforms3d==0.4.1

# Create workspace
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# Copy the ROS2 package
COPY f1tenth_onnx_inference /ros2_ws/src/f1tenth_onnx_inference

# Build the workspace
RUN /bin/bash -c "source /opt/ros/foxy/setup.bash && \
    cd /ros2_ws && \
    colcon build --symlink-install && \
    ls -la /ros2_ws/install/f1tenth_onnx_inference/lib/f1tenth_onnx_inference/"

# Source the workspace in bashrc
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Set the entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["ros2", "launch", "f1tenth_onnx_inference", "inference.launch.py"]
