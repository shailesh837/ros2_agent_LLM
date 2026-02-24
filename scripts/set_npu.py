#!/usr/bin/env python3
"""Set processing mode to NPU"""
import sys

def publish_ros_message(topic, message):
    print(f"[ROS] Publishing to {topic}: {message}")
    return True

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "/demo/device/mode"
    message = sys.argv[2] if len(sys.argv) > 2 else "npu"
    
    print("="*50)
    print("NPU Mode - ROS Publisher")
    print("="*50)
    
    success = publish_ros_message(topic, message)
    
    if success:
        print("[SUCCESS] NPU mode set")
        print("="*50)
        sys.exit(0)
    else:
        print("[ERROR] Failed to set NPU mode")
        sys.exit(1)
