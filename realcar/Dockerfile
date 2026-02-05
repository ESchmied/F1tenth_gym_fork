# =============================================================================
# Dockerfile for F1TENTH DreamerV3 Training & Inference
# =============================================================================
#
# RECOMMENDED: Use Dockerfile.realcar in the repository root instead!
#              It includes all training scripts pre-built.
#
#   cd /path/to/f1tenth_gym_fork
#   docker build -t f1tenth-dreamer:latest -f Dockerfile.realcar .
#
# This Dockerfile can be used standalone but requires mounting the training
# scripts at runtime:
#
#   cd /path/to/f1tenth_gym_fork/realcar
#   docker build -t f1tenth-dreamer:latest .
#
#   docker run --rm -it \
#       --net=host \
#       --ipc=host \
#       --gpus all \
#       -v /path/to/f1tenth_gym_fork:/f1tenth_gym \
#       -v /path/to/checkpoints:/checkpoints \
#       f1tenth-dreamer:latest \
#       python3 /f1tenth_gym/train_dreamerv3.py --real --from_checkpoint /checkpoints/ckpt
#
# =============================================================================
# IMPORTANT: ROS2 NETWORK CONFIGURATION
# =============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  This container MUST be run with:                                       │
# │    --net=host                                                           │
# │    --ipc=host                                                           │
# │                                                                         │
# │  WARNING: The F1TENTH car must have ROS_LOCALHOST_ONLY=0 set!           │
# │           Otherwise, the container cannot communicate with ROS2 nodes.  │
# │                                                                         │
# │  On the car: export ROS_LOCALHOST_ONLY=0                                │
# └─────────────────────────────────────────────────────────────────────────┘
#
# =============================================================================

# Use ROS2 Humble with CUDA support for JAX training
# For inference-only, you can use a lighter base image
ARG BASE_IMAGE=ros:humble-ros-base
FROM ${BASE_IMAGE}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV PYTHONUNBUFFERED=1

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-ackermann-msgs \
    ros-humble-sensor-msgs \
    ros-humble-nav-msgs \
    ros-humble-geometry-msgs \
    wget \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python Dependencies for Training
# =============================================================================

# Upgrade pip
RUN pip3 install --upgrade pip

# Install JAX with CUDA support (for GPU training)
# Note: Adjust cuda version if needed (cuda11 or cuda12)
RUN pip3 install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || \
    pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || \
    pip3 install --upgrade jax jaxlib

# Install DreamerV3 dependencies
RUN pip3 install \
    numpy==1.24.3 \
    transforms3d==0.4.1 \
    gymnasium \
    ruamel.yaml \
    tensorboard \
    matplotlib \
    pandas \
    tensorflow-cpu

# Clone and install DreamerV3 from source (includes embodied and elements)
# This avoids PyPI package issues
# Keep the source directory for editable install to work
RUN git clone https://github.com/danijar/dreamerv3.git /opt/dreamerv3 && \
    cd /opt/dreamerv3 && \
    pip3 install -e .

# Install ONNX runtime for inference mode
RUN pip3 install onnxruntime==1.16.3

# =============================================================================
# Create Workspace
# =============================================================================
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# Copy the ROS2 packages
COPY f1tenth_onnx_inference /ros2_ws/src/f1tenth_onnx_inference
COPY f1tenth_dreamer_training /ros2_ws/src/f1tenth_dreamer_training

# Build the ROS2 workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    cd /ros2_ws && \
    colcon build --symlink-install"

# =============================================================================
# F1TENTH Gym and Training Scripts
# =============================================================================
#
# The training scripts (f1tenth.py, f1tenth_real.py, train_dreamerv3.py) 
# should be mounted at runtime since they are in the parent directory.
#
# When building, use the repository root as build context:
#   cd /path/to/f1tenth_gym_fork
#   docker build -t f1tenth-dreamer:latest -f realcar/Dockerfile .
#
# Or mount at runtime:
#   docker run -v /path/to/f1tenth_gym_fork:/f1tenth_gym ...
#

# Create directory for f1tenth gym
RUN mkdir -p /f1tenth_gym

# Create a placeholder README
RUN echo "Mount f1tenth_gym_fork here or build with repo root as context" > /f1tenth_gym/README.txt

# =============================================================================
# Setup Environment
# =============================================================================

# Source ROS2 and workspace in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc && \
    echo "export PYTHONPATH=/f1tenth_gym:\$PYTHONPATH" >> ~/.bashrc && \
    echo "" >> ~/.bashrc && \
    echo "# Reminder: Car must have ROS_LOCALHOST_ONLY=0" >> ~/.bashrc && \
    echo "echo ''" >> ~/.bashrc && \
    echo "echo '================================================='" >> ~/.bashrc && \
    echo "echo 'F1TENTH DreamerV3 Training Container'" >> ~/.bashrc && \
    echo "echo '================================================='" >> ~/.bashrc && \
    echo "echo 'WARNING: Ensure ROS_LOCALHOST_ONLY=0 on the car!'" >> ~/.bashrc && \
    echo "echo ''" >> ~/.bashrc && \
    echo "echo 'Training example:'" >> ~/.bashrc && \
    echo "echo '  python3 /f1tenth_gym/train_dreamerv3.py --real --max_speed 3.0'" >> ~/.bashrc && \
    echo "echo ''" >> ~/.bashrc && \
    echo "echo 'Inference example:'" >> ~/.bashrc && \
    echo "echo '  ros2 launch f1tenth_onnx_inference inference.launch.py'" >> ~/.bashrc && \
    echo "echo '================================================='" >> ~/.bashrc && \
    echo "echo ''" >> ~/.bashrc

# Set the entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default command shows help
CMD ["bash", "-c", "echo 'Use: python3 /f1tenth_gym/train_dreamerv3.py --help'"]
