#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np
import argparse
import roslibpy
import rospkg
import json

from nextage_vision.image_subscriber import ImageSubscriber

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('nextage_vision')
CONFIG_DIR = os.path.join(pkg_path, "src/nextage_vision/config")

ROS_HOST = 'localhost'
ROS_PORT = 9090
CAMERA_TOPIC = '/camera/color/image_rect_color/compressed'
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'default_mask.json')

def nothing(x):
    pass

def load_config(config_path):
    """
    Loads mask values from a JSON config file.
    """
    # Default values if file doesn't exist
    default_values = {
        "h_low": 0, "h_high": 179,
        "s_low": 0, "s_high": 255,
        "v_low": 0, "v_high": 255,
        "erosion": 0, "dilation": 0
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                values = json.load(f)
                print(f"Successfully loaded config from: {config_path}")
                # Ensure all keys are present, fall back if not
                for key in default_values:
                    if key not in values:
                        print(f"Warning: Key '{key}' missing in config, using default.")
                        values[key] = default_values[key]
                return values
        except Exception as e:
            print(f"Error loading {config_path}: {e}. Using default values.")
            return default_values
    else:
        print(f"No config file found at {config_path}. Using default values.")
        return default_values

def save_config(config_path, window_name):
    """
    Saves the current trackbar positions to a JSON config file.
    """
    # 1. Read all current values from trackbars
    current_values = {
        "h_low":   cv2.getTrackbarPos('H_low',   window_name),
        "h_high": cv2.getTrackbarPos('H_high', window_name),
        "s_low":   cv2.getTrackbarPos('S_low',   window_name),
        "s_high": cv2.getTrackbarPos('S_high', window_name),
        "v_low":   cv2.getTrackbarPos('V_low',   window_name),
        "v_high": cv2.getTrackbarPos('V_high', window_name),
        "erosion":  cv2.getTrackbarPos('Erosion',  window_name),
        "dilation": cv2.getTrackbarPos('Dilation', window_name)
    }
    
    # 2. Ensure the directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # 3. Save to the file
    try:
        with open(config_path, 'w') as f:
            json.dump(current_values, f, indent=4)
        print(f"\n--- Saved configuration to {config_path} ---\n")
    except Exception as e:
        print(f"\n--- Error saving config: {e} ---\n")


def main():
    parser = argparse.ArgumentParser(description="HSV Mask Tuning Tool")
    parser.add_argument(
        '-c', '--config', 
        type=str, 
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the mask config JSON file. Default: {DEFAULT_CONFIG_PATH}"
    )
    args = parser.parse_args()
    config_path = args.config
    
    # --- Load initial values ---
    os.makedirs(CONFIG_DIR, exist_ok=True)
    initial_values = load_config(config_path)

    # --- Create client ---
    ros_client = None
    try:
        ros_client = roslibpy.Ros(host=ROS_HOST, port=ROS_PORT)
        print(f"Connecting to rosbridge at {ROS_HOST}:{ROS_PORT}...")
        ros_client.run()
        
        connect_timeout = 10
        start_time = time.time()
        while not ros_client.is_connected:
            time.sleep(0.1)
            if time.time() - start_time > connect_timeout:
                raise Exception("Connection to rosbridge timed out.")
        print("Successfully connected.")
    except Exception as e:
        print(f"Error connecting to rosbridge: {e}")
        return

    # --- Setup ROS Connection ---
    sub_img = ImageSubscriber(client=ros_client, topic_name=CAMERA_TOPIC)
    time.sleep(0.2) # give it a moment to get first frame

    # --- Setup Window and Trackbars ---
    window_name = 'mask_image'
    cv2.namedWindow(window_name)

    # trackbars (H_max auf 179 ist korrekt für OpenCV HSV)
    cv2.createTrackbar('H_low',   window_name, 0,   179, nothing)
    cv2.createTrackbar('H_high', window_name, 179, 179, nothing)
    cv2.createTrackbar('S_low',   window_name, 0,   255, nothing)
    cv2.createTrackbar('S_high', window_name, 255, 255, nothing)
    cv2.createTrackbar('V_low',   window_name, 0,   255, nothing)
    cv2.createTrackbar('V_high', window_name, 255, 255, nothing)
    
    cv2.createTrackbar('Erosion', window_name, 0, 10, nothing)
    cv2.createTrackbar('Dilation',window_name, 0, 10, nothing)

    switch = '0 : Color \n1 : Mask'
    cv2.createTrackbar(switch, window_name, 0, 1, nothing)

    # --- Set trackbars to loaded values ---
    cv2.setTrackbarPos('H_low',   window_name, initial_values['h_low'])
    cv2.setTrackbarPos('H_high', window_name, initial_values['h_high'])
    cv2.setTrackbarPos('S_low',   window_name, initial_values['s_low'])
    cv2.setTrackbarPos('S_high', window_name, initial_values['s_high'])
    cv2.setTrackbarPos('V_low',   window_name, initial_values['v_low'])
    cv2.setTrackbarPos('V_high', window_name, initial_values['v_high'])
    cv2.setTrackbarPos('Erosion', window_name, initial_values['erosion'])
    cv2.setTrackbarPos('Dilation',window_name, initial_values['dilation'])

    kernel = np.ones((5, 5), np.uint8)

    print("\n--- Controls ---")
    print("  's' key: Save current settings to config file")
    print("  'ESC' key: Quit")
    print("----------------\n")

    try:
        while True:
            img = sub_img.get_frame()
            if img is None:
                cv2.waitKey(1)
                continue

            cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # read trackbars
            h_low   = cv2.getTrackbarPos('H_low',   window_name)
            h_high = cv2.getTrackbarPos('H_high', window_name)
            s_low   = cv2.getTrackbarPos('S_low',   window_name)
            s_high = cv2.getTrackbarPos('S_high', window_name)
            v_low   = cv2.getTrackbarPos('V_low',   window_name)
            v_high = cv2.getTrackbarPos('V_high', window_name)
            erosion = cv2.getTrackbarPos('Erosion', window_name)
            dilation = cv2.getTrackbarPos('Dilation',window_name)
            s       = cv2.getTrackbarPos(switch,    window_name)

            # Create arrays for cv2.inRange
            lower_bound = np.array([h_low, s_low, v_low])
            upper_bound = np.array([h_high, s_high, v_high])

            mask_img = cv2.inRange(cvt_image, lower_bound, upper_bound)

            if erosion != 0:
                mask_img = cv2.erode(mask_img, kernel, iterations=erosion)
            if dilation != 0:
                mask_img = cv2.dilate(mask_img, kernel, iterations=dilation)

            img_color = cv2.bitwise_and(img, img, mask=mask_img)

            if s == 0:
                cv2.imshow(window_name, img_color)
            else:
                cv2.imshow(window_name, mask_img)

            k = cv2.waitKey(1) & 0xFF
            if k == 27: # ESC
                print("ESC pressed. Exiting...")
                break
            elif k == ord('s'): # 's' key
                save_config(config_path, window_name)

    finally:
        sub_img.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()