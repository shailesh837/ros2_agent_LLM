#!/usr/bin/env python3
"""
Dot Action Script
Publishes dot command to ROS topic
"""

import sys

def publish_ros_message(topic, message):
    """Publish message to ROS topic"""
    print(f"[ROS PUBLISH] Topic: {topic}")
    print(f"[ROS PUBLISH] Message: {message}")
    return True

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "/demo/shape/dot"
    message = sys.argv[2] if len(sys.argv) > 2 else "dot"
    
    print("="*50)
    print("Dot Action - ROS Publisher")
    print("="*50)
    
    success = publish_ros_message(topic, message)
    
    if success:
        print("[SUCCESS] Dot command published")
        print("="*50)
        sys.exit(0)
    else:
        print("[ERROR] Failed to publish")
        sys.exit(1)
