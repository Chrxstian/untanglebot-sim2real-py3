#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import cv2
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import rospy
from nextage_vision.image_subscriber import RGBDSubscriber

RGB_TOPIC = '/camera/color/image_rect_color/compressed'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw/compressedDepth'

def main():
    print("Initializing ROS Node...")
    rospy.init_node('test_depth_sensor', anonymous=True)
            
    print("Connected. Initializing Subscriber...")
    cam = RGBDSubscriber(RGB_TOPIC, DEPTH_TOPIC)
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    print(f"Listening to:\n - {RGB_TOPIC}\n - {DEPTH_TOPIC}")
    print("Updating every 2 seconds. Close window or Ctrl+C to quit.")

    try:
        while plt.fignum_exists(fig.number) and not rospy.is_shutdown():
            
            rgb, depth = cam.get_frames(timeout=3.0) 
            
            if not plt.fignum_exists(fig.number):
                break

            # 1. Update RGB
            ax1.clear()
            if rgb is not None:
                ax1.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                ax1.set_title(f"RGB ({rgb.shape})")
            else:
                ax1.text(0.5, 0.5, "No RGB Data", ha='center')
                ax1.set_title("RGB Feed (Timeout)")

            # 2. Update Depth
            ax2.clear()
            if depth is not None:
                ax2.imshow(depth, cmap='jet', vmin=0, vmax=2000)
                cy, cx = depth.shape[0]//2, depth.shape[1]//2
                dist_mm = depth[cy, cx]
                ax2.set_title(f"Depth Aligned (Center: {dist_mm}mm)")
            else:
                ax2.text(0.5, 0.5, "No Depth Data", ha='center')
                ax2.set_title("Depth Feed (Timeout)")

            plt.draw()
            plt.pause(2.0)

    except KeyboardInterrupt:
        print("\nStopping via Ctrl+C...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("Closing connections...")
        plt.close('all')
        print("Done.")

if __name__ == '__main__':
    main()