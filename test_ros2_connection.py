#!/usr/bin/env python3
"""
Test ROS2 connection for F1TENTH real car training.

This script helps diagnose ROS2 connectivity issues before starting training.

Usage:
    python3 test_ros2_connection.py
    
    # Or in Docker:
    docker run --rm -it --net=host --ipc=host f1tenth-dreamer:latest \
        python3 /f1tenth_gym/test_ros2_connection.py
"""

import sys
import time
import numpy as np

def test_ros2_topics():
    """Test if ROS2 topics are accessible."""
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import LaserScan
        from nav_msgs.msg import Odometry
        from ackermann_msgs.msg import AckermannDriveStamped
    except ImportError as e:
        print(f"❌ Failed to import ROS2 modules: {e}")
        print("Make sure ROS2 is installed and sourced.")
        return False
    
    print("✅ ROS2 Python modules imported successfully")
    
    # Initialize ROS2
    try:
        if not rclpy.ok():
            rclpy.init()
        print("✅ ROS2 initialized")
    except Exception as e:
        print(f"❌ Failed to initialize ROS2: {e}")
        return False
    
    # Create test node
    class TestNode(Node):
        def __init__(self):
            super().__init__('f1tenth_test_node')
            self.scan_received = False
            self.odom_received = False
            self.scan_data = None
            self.odom_data = None
            
            # Create subscriptions
            self.scan_sub = self.create_subscription(
                LaserScan, '/scan', self.scan_callback, 10
            )
            self.odom_sub = self.create_subscription(
                Odometry, '/odom', self.odom_callback, 10
            )
            
            # Create publisher
            self.drive_pub = self.create_publisher(
                AckermannDriveStamped, '/drive', 10
            )
            
            print("✅ Test node created")
            print(f"   Subscribed to: /scan, /odom")
            print(f"   Publishing to: /drive")
        
        def scan_callback(self, msg):
            if not self.scan_received:
                self.scan_received = True
                self.scan_data = np.array(msg.ranges)
                print(f"✅ Received scan data! ({len(msg.ranges)} beams)")
                print(f"   Min distance: {np.min(self.scan_data[np.isfinite(self.scan_data)]):.2f}m")
                print(f"   Max distance: {np.max(self.scan_data[np.isfinite(self.scan_data)]):.2f}m")
        
        def odom_callback(self, msg):
            if not self.odom_received:
                self.odom_received = True
                vel = msg.twist.twist.linear.x
                ang_vel = msg.twist.twist.angular.z
                print(f"✅ Received odom data!")
                print(f"   Linear velocity: {vel:.2f} m/s")
                print(f"   Angular velocity: {ang_vel:.2f} rad/s")
        
        def test_publish(self):
            """Test publishing a drive command."""
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.drive.steering_angle = 0.0
            msg.drive.speed = 0.0
            
            try:
                self.drive_pub.publish(msg)
                print("✅ Successfully published test drive command")
                return True
            except Exception as e:
                print(f"❌ Failed to publish drive command: {e}")
                return False
    
    node = TestNode()
    
    print("\n" + "="*60)
    print("Waiting for sensor data... (5 seconds)")
    print("="*60)
    
    # Spin for 5 seconds to receive data
    start_time = time.time()
    while time.time() - start_time < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    
    success = True
    
    if not node.scan_received:
        print("❌ Did NOT receive scan data on /scan")
        print("   Troubleshooting:")
        print("   1. Check if topic exists: ros2 topic list | grep /scan")
        print("   2. Check if publishing: ros2 topic hz /scan")
        print("   3. Check car has: export ROS_LOCALHOST_ONLY=0")
        print("   4. Check container started with: --net=host --ipc=host")
        success = False
    
    if not node.odom_received:
        print("❌ Did NOT receive odom data on /odom")
        print("   Troubleshooting:")
        print("   1. Check if topic exists: ros2 topic list | grep /odom")
        print("   2. Check if publishing: ros2 topic hz /odom")
        print("   3. Check car has: export ROS_LOCALHOST_ONLY=0")
        print("   4. Check container started with: --net=host --ipc=host")
        success = False
    
    # Test publishing
    print("\nTesting drive command publishing...")
    if not node.test_publish():
        success = False
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("You can proceed with training.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Fix the issues above before starting training.")
    print("="*60)
    
    return success

if __name__ == '__main__':
    try:
        success = test_ros2_topics()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
