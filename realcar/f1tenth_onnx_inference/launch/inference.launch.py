#!/usr/bin/env python3
"""Launch file for F1TENTH ONNX inference node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='/models/final_model.onnx',
            description='Path to ONNX model file'
        ),
        
        DeclareLaunchArgument(
            'inference_rate',
            default_value='50.0',
            description='Inference rate in Hz'
        ),
        
        DeclareLaunchArgument(
            'max_speed',
            default_value='20.0',
            description='Maximum speed in m/s'
        ),
        
        DeclareLaunchArgument(
            'max_steering_angle',
            default_value='0.4189',
            description='Maximum steering angle in radians'
        ),
        
        Node(
            package='f1tenth_onnx_inference',
            executable='inference_node',
            name='f1tenth_onnx_inference',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'inference_rate': LaunchConfiguration('inference_rate'),
                'max_speed': LaunchConfiguration('max_speed'),
                'max_steering_angle': LaunchConfiguration('max_steering_angle'),
                'scan_subsample_size': 12,
                'scan_original_size': 1080,
            }]
        ),
    ])
