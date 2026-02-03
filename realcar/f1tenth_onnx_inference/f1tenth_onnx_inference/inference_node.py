#!/usr/bin/env python3
"""
F1TENTH ONNX Inference Node

This ROS2 node performs real-time inference using a trained ONNX model
for autonomous racing on the F1TENTH platform.
"""

import numpy as np
import onnxruntime as ort
from collections import deque
import csv
import datetime
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from transforms3d.euler import quat2euler


class F1TenthONNXInference(Node):
    def __init__(self):
        super().__init__('f1tenth_onnx_inference')
        
        # Declare parameters
        self.declare_parameter('model_path', '/models/final_model.onnx')
        self.declare_parameter('scan_subsample_size', 12)
        self.declare_parameter('scan_original_size', 1080)
        self.declare_parameter('max_speed', 20.0)
        self.declare_parameter('max_steering_angle', 0.4189)
        self.declare_parameter('inference_rate', 20.0)  # Hz
        self.declare_parameter('window_size', 5)
        self.declare_parameter('log_data', True)
        self.declare_parameter('log_path', 'inference_logs')
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        self.scan_subsample_size = self.get_parameter('scan_subsample_size').value
        self.scan_original_size = self.get_parameter('scan_original_size').value
        self.max_speed = self.get_parameter('max_speed').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        inference_rate = self.get_parameter('inference_rate').value
        self.window_size = self.get_parameter('window_size').value
        self.log_data = self.get_parameter('log_data').value
        self.log_path = self.get_parameter('log_path').value
        
        # Load ONNX model
        self.get_logger().info(f'Loading ONNX model from: {model_path}')
        self.ort_session = ort.InferenceSession(model_path)
        self.get_logger().info('ONNX model loaded successfully')

        # Observation history for sliding window
        self.obs_history = deque(maxlen=self.window_size)

        # CSV Logging Setup
        if self.log_data:
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_file_path = os.path.join(self.log_path, f'inference_log_{timestamp}.csv')
            self.csv_file = open(self.csv_file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Header
            header = [
                'timestamp', 'linear_vel_x', 'ang_vel_z', 'delta'
            ]
            for i in range(self.scan_subsample_size):
                header.append(f'scan_{i}')
            header.extend(['output_steering', 'output_speed'])
            self.csv_writer.writerow(header)
            self.get_logger().info(f'Logging data to {self.csv_file_path}')
        
        # State variables
        self.current_scan = None
        self.current_odom = None
        self.current_steering = 0.0  # Track current steering angle (delta)
        self.ready = False
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.get_logger().info('Created subscription to /scan')
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.get_logger().info('Created subscription to /odom')
        
        # Publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )
        
        # Timer for inference
        self.timer = self.create_timer(
            1.0 / inference_rate,
            self.inference_callback
        )
        
        # Timer to check connections
        self.create_timer(5.0, self.check_connections)
        
        self.get_logger().info('F1TENTH ONNX Inference Node initialized')
    
    def scan_callback(self, msg: LaserScan):
        """Process incoming laser scan data."""
        self.get_logger().info('Received scan data', once=True)
        # Convert scan to numpy array
        scan_array = np.array(msg.ranges, dtype=np.float32)
        
        # Handle inf and nan values
        scan_array = np.nan_to_num(scan_array, nan=30.0, posinf=30.0, neginf=0.0)
        
        # Subsample the scan to match training (1080 -> 12 beams)
        if len(scan_array) == self.scan_original_size:
            indices = np.linspace(0, self.scan_original_size - 1, 
                                self.scan_subsample_size, dtype=int)
            self.current_scan = scan_array[indices]
        elif len(scan_array) == self.scan_subsample_size:
            self.current_scan = scan_array
        else:
            # Resample to desired size
            indices = np.linspace(0, len(scan_array) - 1, 
                                self.scan_subsample_size, dtype=int)
            self.current_scan = scan_array[indices]
        
        # Clip to reasonable range
        self.current_scan = np.clip(self.current_scan, 0.0, 30.0)
    
    def odom_callback(self, msg: Odometry):
        """Process incoming odometry data."""
        self.get_logger().info('Received odom data', once=True)
        # Extract position
        pose_x = msg.pose.pose.position.x
        pose_y = msg.pose.pose.position.y
        
        # Extract orientation (quaternion to euler)
        quat = msg.pose.pose.orientation
        _, _, pose_theta = quat2euler([quat.w, quat.x, quat.y, quat.z])
        
        # Extract velocities
        linear_vel_x = msg.twist.twist.linear.x
        ang_vel_z = msg.twist.twist.angular.z
        
        self.current_odom = {
            'linear_vel_x': np.array([linear_vel_x], dtype=np.float32),
            'ang_vel_z': np.array([ang_vel_z], dtype=np.float32),
            'delta': np.array([self.current_steering], dtype=np.float32)
        }
        
        if self.current_scan is not None:
            self.ready = True
    
    def check_connections(self):
        """Check subscription connections."""
        scan_info = self.count_publishers('/scan')
        odom_info = self.count_publishers('/odom')
        self.get_logger().info(
            f'Topic publishers - /scan: {scan_info}, /odom: {odom_info}'
        )
    
    def inference_callback(self):
        """Run inference and publish drive commands."""
        if not self.ready or self.current_scan is None or self.current_odom is None:
            status = f'Waiting for sensor data... (scan: {self.current_scan is not None}, odom: {self.current_odom is not None}, ready: {self.ready})'
            self.get_logger().warn(status, throttle_duration_sec=2.0)
            return
        
        try:
            # Prepare observation dict
            current_obs = {
                'linear_vel_x': -5*self.current_odom['linear_vel_x'].reshape(1, 1),
                'ang_vel_z': self.current_odom['ang_vel_z'].reshape(1, 1),
                'delta': self.current_odom['delta'].reshape(1, 1),
                'scan': self.current_scan.reshape(1, -1)
            }
            
            # Add to sliding window
            self.obs_history.append(current_obs)
            
            # Calculate sliding window average
            obs = {}
            for key in current_obs.keys():
                obs[key] = np.mean([o[key] for o in self.obs_history], axis=0)
            
            # Run inference
            outputs = self.ort_session.run(None, obs)
            action = outputs[0][0][0]  # Shape: (2,)
            
            # Extract steering and speed directly from model output
            # Action space: steering [-0.4189, 0.4189], speed [-5, 20]
            steering_angle = float(action[0])
            speed = min(float(action[1]), self.max_speed)
            
            # Update tracked steering for next observation
            self.current_steering = steering_angle
            
            # Create and publish Ackermann drive message
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.steering_angle = steering_angle
            drive_msg.drive.speed = speed
            
            self.drive_pub.publish(drive_msg)
            
            # Log to CSV
            if self.log_data:
                try:
                    now = self.get_clock().now().to_msg()
                    timestamp = now.sec + now.nanosec * 1e-9
                    row = [
                        timestamp,
                        float(obs['linear_vel_x'][0, 0]),
                        float(obs['ang_vel_z'][0, 0]),
                        float(obs['delta'][0, 0])
                    ]
                    row.extend(obs['scan'][0].tolist())
                    row.extend([steering_angle, speed])
                    self.csv_writer.writerow(row)
                    self.csv_file.flush()
                except Exception as log_e:
                    self.get_logger().error(f'Logging error: {str(log_e)}')
            
            # Log periodically
            self.get_logger().info(
                f'Action: steering={steering_angle:.3f} rad, speed={speed:.2f} m/s',
                throttle_duration_sec=1.0
            )
            
        except Exception as e:
            self.get_logger().error(f'Inference error: {str(e)}')

    def destroy_node(self):
        """Close the CSV file on shutdown."""
        if hasattr(self, 'csv_file') and self.csv_file:
            self.get_logger().info(f'Closing log file: {self.csv_file_path}')
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = F1TenthONNXInference()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
