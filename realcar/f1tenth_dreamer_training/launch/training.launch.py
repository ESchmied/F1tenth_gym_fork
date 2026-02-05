"""
Launch file for DreamerV3 training on real F1TENTH car.

This launch file is mainly for reference. For actual training, it's recommended
to use the train_dreamerv3.py script directly with the --real flag.

Usage:
    ros2 launch f1tenth_dreamer_training training.launch.py
    
    # With custom parameters:
    ros2 launch f1tenth_dreamer_training training.launch.py \
        checkpoint_path:=/path/to/checkpoint \
        max_speed:=3.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    checkpoint_path_arg = DeclareLaunchArgument(
        'checkpoint_path',
        default_value='',
        description='Path to checkpoint to load weights from (sim-to-real transfer)'
    )
    
    logdir_arg = DeclareLaunchArgument(
        'logdir',
        default_value='~/logdir/f1tenth/real',
        description='Directory for logs and checkpoints'
    )
    
    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='3.0',
        description='Maximum speed for safety (m/s)'
    )
    
    scan_beams_arg = DeclareLaunchArgument(
        'scan_beams',
        default_value='32',
        description='Number of LiDAR beams (subsampled)'
    )
    
    step_frequency_arg = DeclareLaunchArgument(
        'step_frequency',
        default_value='20.0',
        description='Training step frequency (Hz)'
    )
    
    collision_threshold_arg = DeclareLaunchArgument(
        'collision_threshold',
        default_value='0.3',
        description='Minimum distance to consider collision (m)'
    )
    
    model_size_arg = DeclareLaunchArgument(
        'model_size',
        default_value='size12m',
        description='DreamerV3 model size'
    )
    
    steps_arg = DeclareLaunchArgument(
        'steps',
        default_value='100000',
        description='Total training steps'
    )
    
    # Training node
    training_node = Node(
        package='f1tenth_dreamer_training',
        executable='training_node',
        name='dreamer_training',
        output='screen',
        parameters=[{
            'checkpoint_path': LaunchConfiguration('checkpoint_path'),
            'logdir': LaunchConfiguration('logdir'),
            'max_speed': LaunchConfiguration('max_speed'),
            'scan_beams': LaunchConfiguration('scan_beams'),
            'step_frequency': LaunchConfiguration('step_frequency'),
            'collision_threshold': LaunchConfiguration('collision_threshold'),
            'model_size': LaunchConfiguration('model_size'),
            'steps': LaunchConfiguration('steps'),
        }]
    )
    
    return LaunchDescription([
        checkpoint_path_arg,
        logdir_arg,
        max_speed_arg,
        scan_beams_arg,
        step_frequency_arg,
        collision_threshold_arg,
        model_size_arg,
        steps_arg,
        training_node,
    ])
