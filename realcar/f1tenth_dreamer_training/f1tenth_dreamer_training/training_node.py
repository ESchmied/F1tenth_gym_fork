#!/usr/bin/env python3
"""
F1TENTH DreamerV3 Training Node

This ROS2 node enables training DreamerV3 agents directly on the real F1TENTH car.
It provides a wrapper that integrates with the main training script while handling
ROS2 communication.

Usage:
    # Build the package
    cd ~/f1tenth_ws
    colcon build --packages-select f1tenth_dreamer_training
    source install/setup.bash
    
    # Run training (ROS2 node approach - not recommended, use script instead)
    ros2 run f1tenth_dreamer_training training_node --ros-args -p checkpoint_path:=/path/to/checkpoint
    
    # Recommended: Use the main training script directly
    cd /path/to/f1tenth_gym_fork
    python train_dreamerv3.py --real --from_checkpoint /path/to/sim_checkpoint

Note: For training, it's recommended to use the main train_dreamerv3.py script
with the --real flag, as it provides more flexibility and better integration
with DreamerV3's training infrastructure.
"""

import os
import sys
import signal
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool, String


class DreamerTrainingNode(Node):
    """
    ROS2 Node that orchestrates DreamerV3 training on the real car.
    
    This node is primarily for monitoring and control. The actual training
    is handled by the f1tenth_real.py environment wrapper which runs in
    a separate thread and communicates via ROS2 topics.
    """
    
    def __init__(self):
        super().__init__('dreamer_training_node')
        
        # Declare parameters
        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('logdir', '~/logdir/f1tenth/real')
        self.declare_parameter('max_speed', 3.0)
        self.declare_parameter('scan_beams', 32)
        self.declare_parameter('step_frequency', 20.0)
        self.declare_parameter('collision_threshold', 0.3)
        self.declare_parameter('model_size', 'size12m')
        self.declare_parameter('steps', 100000)
        
        # Get parameters
        self.checkpoint_path = self.get_parameter('checkpoint_path').value
        self.logdir = self.get_parameter('logdir').value
        self.max_speed = self.get_parameter('max_speed').value
        self.scan_beams = self.get_parameter('scan_beams').value
        self.step_frequency = self.get_parameter('step_frequency').value
        self.collision_threshold = self.get_parameter('collision_threshold').value
        self.model_size = self.get_parameter('model_size').value
        self.steps = self.get_parameter('steps').value
        
        # Status publisher
        self.status_pub = self.create_publisher(String, '/dreamer/status', 10)
        
        # Emergency stop subscriber
        self.estop_sub = self.create_subscription(
            Bool,
            '/dreamer/emergency_stop',
            self.emergency_stop_callback,
            10
        )
        
        # Drive command publisher (for emergency stop)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )
        
        self.training_active = False
        self.emergency_stopped = False
        
        # Timer for status updates
        self.create_timer(1.0, self.publish_status)
        
        self.get_logger().info('DreamerV3 Training Node initialized')
        self.get_logger().info(f'  Checkpoint: {self.checkpoint_path}')
        self.get_logger().info(f'  Max speed: {self.max_speed} m/s')
        self.get_logger().info(f'  Scan beams: {self.scan_beams}')
        self.get_logger().info(f'  Step frequency: {self.step_frequency} Hz')
        
        # Print instructions
        self.get_logger().info('')
        self.get_logger().info('='*60)
        self.get_logger().info('RECOMMENDED: Use the training script directly:')
        self.get_logger().info('')
        self.get_logger().info('  python train_dreamerv3.py --real \\')
        self.get_logger().info(f'    --max_speed {self.max_speed} \\')
        self.get_logger().info(f'    --scan_beams {self.scan_beams} \\')
        if self.checkpoint_path:
            self.get_logger().info(f'    --from_checkpoint {self.checkpoint_path}')
        self.get_logger().info('')
        self.get_logger().info('='*60)
    
    def emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop commands."""
        if msg.data:
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
            self.emergency_stopped = True
            self.stop_car()
        else:
            self.get_logger().info('Emergency stop released')
            self.emergency_stopped = False
    
    def stop_car(self):
        """Send stop command to the car."""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = 0.0
        msg.drive.speed = 0.0
        self.drive_pub.publish(msg)
    
    def publish_status(self):
        """Publish training status."""
        status = String()
        if self.emergency_stopped:
            status.data = 'EMERGENCY_STOPPED'
        elif self.training_active:
            status.data = 'TRAINING'
        else:
            status.data = 'IDLE'
        self.status_pub.publish(status)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = DreamerTrainingNode()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        node.get_logger().info('Shutting down...')
        node.stop_car()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_car()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
