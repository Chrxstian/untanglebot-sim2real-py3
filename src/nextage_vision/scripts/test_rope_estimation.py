#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import traceback
import time
import roslibpy
import rospkg 

from nextage_vision.image_subscriber import RGBDSubscriber, ImageSubscriber
from nextage_vision.image_processor import ImageProcessor
from libRopeEstimator.RopePoseEstimator import RopePoseEstimator

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('nextage_vision')
CONFIG_DIR = os.path.join(pkg_path, "src/nextage_vision/config") + "/"

ROS_HOST = 'localhost'
ROS_PORT = 9090
RGB_TOPIC = '/camera/color/image_rect_color/compressed'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw/compressedDepth'
USE_DEPTH = True

def main():
    sub = None
    ros_client = None
    
    try:
        ros_client = roslibpy.Ros(host=ROS_HOST, port=ROS_PORT)
        print(f"Connecting to rosbridge at {ROS_HOST}:{ROS_PORT}...")
        ros_client.run()
        
        # Wait for connection
        start_time = time.time()
        while not ros_client.is_connected:
            time.sleep(0.1)
            if time.time() - start_time > 10:
                raise Exception("Connection to rosbridge timed out.")
        print("Successfully connected.")
        
        # 1. Initialize Subscriber
        if USE_DEPTH:
            print("Initializing RGB-D Subscriber...")
            sub = RGBDSubscriber(ros_client, RGB_TOPIC, DEPTH_TOPIC)
        else:
            print("Initializing RGB-only Subscriber...")
            sub = ImageSubscriber(ros_client, RGB_TOPIC)
        
        # 2. Initialize Logic & Model
        image_processor = ImageProcessor(CONFIG_DIR, use_depth=USE_DEPTH)
        rope_estimator_model = RopePoseEstimator(save_visualizations=True, num_balls=30, ros_client=ros_client, use_depth=USE_DEPTH)
        
        print("\n" + "="*50)
        print(f"Test script ready. USE_DEPTH: {USE_DEPTH}")
        print("="*50)
        
        while True:
            key = input("\nPress 's' and ENTER to estimate/publish (or 'q' to quit): ")
            
            if key.lower() == 'q':
                break
            
            if key.lower() == 's':
                print("\nCapturing frames...")
                
                # Fetch frames
                if isinstance(sub, RGBDSubscriber):
                    img, depth = sub.get_frames(timeout=3.0)
                else:
                    img = sub.get_frame()
                    depth = None

                if img is None:
                    print("ERROR: Could not get RGB image. Retrying...")
                    continue

                if USE_DEPTH and depth is None:
                     print("ERROR: Depth enabled but timed out waiting for depth frame.")
                     continue
                
                print(f"Frames Acquired. RGB: {img.shape} " + (f"Depth: {depth.shape}" if depth is not None else ""))
                
                # Setup Vis
                timestamp = "latest_manual"
                vis_dir = os.path.join(pkg_path, "vis")
                os.makedirs(os.path.join(vis_dir, timestamp), exist_ok=True)
                cv2.imwrite(f"{vis_dir}/{timestamp}/image_raw.png", img)

                tensor, depth_map_mm, seg_bitmap = image_processor.process_image(
                    img, 
                    timestamp=timestamp, 
                    depth_img=depth,
                    vis_dir=vis_dir
                )
                
                print(f"Processed Tensor Shape: {tensor.shape}")
            
                rope_poses = rope_estimator_model.estimate_rope_pose(tensor, timestamp=timestamp)
                
                if rope_poses is not None:
                    print("Rope poses estimated successfully.")
                    
                    # Show example depth
                    center_xy = (128, 128)
                    dist_m = image_processor.get_depth_at_point(center_xy, depth_map_mm)
                    print(f"Real-world Depth at center pixel (128,128): {dist_m:.3f} meters")
                else:
                    print("No pose returned.")
            else:
                print("Invalid key.")
            
    except KeyboardInterrupt:
        print("\nShutting down via KeyboardInterrupt.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if sub: sub.close()
        if ros_client: ros_client.terminate()

if __name__ == '__main__':
    main()