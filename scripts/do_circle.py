#!/usr/bin/env python3
"""
Circle Action Script
Publishes circle command to ROS topic
"""

import sys
import json

def publish_ros_message(topic, message):
    """
    Publish message to ROS topic
    
    In a real ROS environment, this would use:
    - rospy.Publisher for ROS1
    - rclpy for ROS2
    
    For now, this simulates the publish
    """
    print(f"[ROS PUBLISH] Topic: {topic}")
    print(f"[ROS PUBLISH] Message: {message}")
    
    # Example ROS1 code (commented):
    # import rospy
    # from std_msgs.msg import String
    # pub = rospy.Publisher(topic, String, queue_size=10)
    # pub.publish(message)
    
    # Example ROS2 code (commented):
    # import rclpy
    # from std_msgs.msg import String
    # node.get_logger().info(f'Publishing: {message}')
    # publisher.publish(String(data=message))
    
    return True

if __name__ == "__main__":
    # Get topic and message from command line args
    topic = sys.argv[1] if len(sys.argv) > 1 else "/demo/shape/circle"
    message = sys.argv[2] if len(sys.argv) > 2 else "circle"
    
    print("="*50)
    print("Circle Action - ROS Publisher")
    print("="*50)
    
    success = publish_ros_message(topic, message)
    
    if success:
        print("[SUCCESS] Circle command published")
        print("="*50)
        sys.exit(0)
    else:
        print("[ERROR] Failed to publish")
        sys.exit(1)
