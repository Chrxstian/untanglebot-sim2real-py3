#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import json
import time
import roslibpy
import roslibpy.core
from datetime import datetime

from nextage_vision.image_subscriber import ImageSubscriber, RGBDSubscriber
from nextage_vision.image_processor import ImageProcessor
from nextage_vision.mask_utils import create_hsv_mask, create_intersection_mask
from libRopeEstimator.RopePoseEstimator import RopePoseEstimator

import rospkg
rospack = rospkg.RosPack()
vision_pkg_path = rospack.get_path('nextage_vision')
CONFIG_DIR = os.path.join(vision_pkg_path, "src/nextage_vision/config") + "/"

ROS_HOST = 'localhost'
ROS_PORT = 9090
CAMERA_TOPIC = '/camera/color/image_rect_color/compressed'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw/compressedDepth'
USE_DEPTH = True

MESSAGE_TYPE = 'sensor_msgs/CompressedImage'

def send_service_request(client, request_data):
    try:
        print("\n" + "="*50)
        if not client.is_connected:
            raise Exception("ROS Client is not connected. Cannot send service request.")

        service = roslibpy.Service(client, '/policy_action_service', 'untanglebot_sim2real/PolicyAction')
        
        print("Sending service request:")
        print(json.dumps(request_data, indent=2))
        
        request = roslibpy.ServiceRequest(request_data)
        result = service.call(request)
        
        print("\nReceived service response:")
        print(result)
        print("="*50 + "\n")
        return result['success']

    except Exception as e:
        print(f"An error occurred during service call: {e}")
        return False

def main():
    
    sub = None
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
        
        if USE_DEPTH:
            sub = RGBDSubscriber(ros_client, CAMERA_TOPIC, DEPTH_TOPIC)
        else:
            sub = ImageSubscriber(ros_client, CAMERA_TOPIC)
            
        image_processor = ImageProcessor(CONFIG_DIR, use_depth=USE_DEPTH)
        rope_estimator_model = RopePoseEstimator(save_visualizations=True, num_balls=30, ros_client=ros_client, use_depth=USE_DEPTH)
        
        # Wait for the first frame
        print("Waiting for first image frame from rosbridge...")
        
        if isinstance(sub, RGBDSubscriber):
            img, depth_img = sub.get_frames(timeout=5.0)
        else:
            img = sub.get_frame()
            depth_img = None
            while img is None:
                cv2.waitKey(1)
                img = sub.get_frame()

        if img is None:
             raise Exception("No image received.")

        print("Image received. Processing...")
        
        # Save the image
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timestamp = "latest_IL"
        vis_dir = os.path.join(vision_pkg_path, "vis")
        os.makedirs(os.path.join(vis_dir, timestamp), exist_ok=True)
        cv2.imwrite(f"{vis_dir}/{timestamp}/image_raw.png", img)
            
        segmentation_tensor, depth_map_mm, segmentation_bitmap_np = image_processor.process_image(
            img, 
            timestamp=timestamp, 
            depth_img=depth_img, 
            vis_dir=vis_dir
        )
        
        rope_poses = rope_estimator_model.estimate_rope_pose(segmentation_tensor, timestamp=timestamp) 

        pred_coordinates_yx = rope_estimator_model.last_model_output["coordinates"].squeeze(0).cpu().numpy()
        pred_types = rope_estimator_model.last_model_output["types"]
        pred_types = (torch.sigmoid(pred_types) > 0.5).squeeze(0).int().cpu().numpy()
        
        print(f"Received rope_poses dict of shape {rope_poses.shape}")
        
        # 1. Ask user for 'solving_rope'    
        action_map = {0: "Pull Under", 1: "Pass Over"}
        print("\n--- Human-in-the-Loop Policy ---")
        print(f"Detected Action for Rope Red: {action_map.get(pred_types[0])}")
        print(f"Detected Action for Rope Pink: {action_map.get(pred_types[1])}")
        
        # Ask user which rope to solve
        # solving_rope = int(input("Which rope to solve? (0 or 1): "))
        # assert solving_rope in [0, 1], "Invalid input! Please enter 0 or 1."
        
        solving_rope = rope_estimator_model.choose_solving_rope(rope_poses)
        
        # 2. Get intersection type from the model
        # pred_types[0] corresponds to rope 0, pred_types[1] to rope 1
        intersection_type = pred_types[solving_rope]
        action_name = action_map.get(intersection_type)
        
        # 3. Create masks
        rope_l_mask = segmentation_bitmap_np[:, :, 0] > 0
        rope_r_mask = segmentation_bitmap_np[:, :, 1] > 0
        intersection_mask = create_intersection_mask(rope_l_mask, rope_r_mask)

        # 4. Set NN output points based on solving_rope
        if solving_rope == 0: # solving rope_l (red)
            endpoint_yx = pred_coordinates_yx[0]
            intersec_yx = pred_coordinates_yx[1]
            rope_mask = rope_l_mask
        else: # solving rope_r (pink)
            endpoint_yx = pred_coordinates_yx[2]
            intersec_yx = pred_coordinates_yx[3]
            rope_mask = rope_r_mask
            
        # 5. Snap NN points to masks
        # Note: We swap (x,y) from model to (y,x) for snap_to_mask
        snapped_endpoint_yx = image_processor.snap_to_mask(endpoint_yx, rope_mask)
        snapped_intersec_yx = image_processor.snap_to_mask(intersec_yx, intersection_mask)
        print("snapped intersec yx:", snapped_intersec_yx)
        
        # 6. Find closest ball IDs to the snapped intersection point
        # We must swap back (y,x) -> (x,y) to query rope_poses
        snapped_intersec_xy = np.array([snapped_intersec_yx[1], snapped_intersec_yx[0]])
        
        intersec_l_idx = image_processor.snap_to_rope_id(snapped_intersec_xy, rope_poses[0])
        intersec_r_idx = image_processor.snap_to_rope_id(snapped_intersec_xy, rope_poses[1])
        
        # Endpoints are hardcoded to the end of the rope
        endpoint_l_idx = 29
        endpoint_r_idx = 29
        print(f"Calculated IDs: intersec_l_idx={intersec_l_idx}, intersec_r_idx={intersec_r_idx}")

        # 7. Define default "no-op" service message
        force_left = [0.0, 0.0, 0.0]
        grasp_idx_left = -1
        
        force_right = [0.0, 0.0, 0.0]
        grasp_idx_right = -1

        # --- Action 1: LIFT AND PULL UNDER (Type 0) ---
        if intersection_type == 0:
            if solving_rope == 0: # solving rope_l
                # LIFT rope_r
                grasp_idx_right = intersec_r_idx
                force_right = [0.0, 0.0, 1.0] # LIFT flag
                grasp_pos_r_256 = rope_poses[1, grasp_idx_right]

                # PULL rope_l
                grasp_idx_left = max(intersec_l_idx - 1, 0)
                
                ahead_of_intersection = min(intersec_l_idx + 1, 29)
                behind_of_intersection = max(intersec_l_idx - 1, 0)
                direction_2d = rope_poses[0, behind_of_intersection] - rope_poses[0, ahead_of_intersection]
                
                # target_idx_left = max(intersec_l_idx - 3, 0)
                # grasp_pos_l_256 = rope_poses[0, grasp_idx_left]
                # target_pos_l_256 = rope_poses[0, target_idx_left]
                # print("pulling from idx:", grasp_idx_left, "to idx:", target_idx_left)
                # print("grasp_pos_l_256:", grasp_pos_l_256, "target_pos_l_256:", target_pos_l_256)
                
                # --- CALCULATE DIRECTION VECTOR ---
                # direction_2d = target_pos_l_256 - grasp_pos_l_256
                norm_direction_2d = direction_2d / (np.linalg.norm(direction_2d) + 1e-6)
                direction_3d = np.array([norm_direction_2d[0], norm_direction_2d[1], 0.0]) # Z=0 for straight pull
                force_left = direction_3d.tolist()
                print("resulting in force direction", force_left)
                
            else: # solving rope_r
                # LIFT rope_l
                grasp_idx_left = intersec_l_idx
                force_left = [0.0, 0.0, 1.0] # LIFT flag
                grasp_pos_l_256 = rope_poses[0, grasp_idx_left]
                
                # PULL rope_r
                grasp_idx_right = max(intersec_r_idx - 1, 0)
                
                ahead_of_intersection = min(intersec_r_idx + 1, 29)
                behind_of_intersection = max(intersec_r_idx - 1, 0)
                direction_2d = rope_poses[1, behind_of_intersection] - rope_poses[1, ahead_of_intersection]
                                
                # target_idx_right = max(intersec_r_idx - 3, 0)
                # grasp_pos_r_256 = rope_poses[1, grasp_idx_right]
                # target_pos_r_256 = rope_poses[1, target_idx_right]
                # print("pulling from idx:", grasp_idx_right, "to idx:", target_idx_right)
                # print("grasp_pos_r_256:", grasp_pos_r_256, "target_pos_r_256:", target_pos_r_256)
                
                # --- CALCULATE DIRECTION VECTOR ---
                # direction_2d = target_pos_r_256 - grasp_pos_r_256
                norm_direction_2d = direction_2d / (np.linalg.norm(direction_2d) + 1e-6)
                direction_3d = np.array([norm_direction_2d[0], norm_direction_2d[1], 0.0]) # Z=0 for straight pull
                force_right = direction_3d.tolist()
                print("resulting in force direction", force_right)


        # --- Action 2: PIN AND PASS OVER (Type 1) ---
        elif intersection_type == 1:
            if solving_rope == 0: # solving rope_l
                # PIN rope_r (the "other" rope)
                grasp_idx_right = -1 # PIN (no movement for now)
                grasp_pos_r_256 = rope_poses[1, intersec_r_idx]
                force_right = [0.0, 0.0, 0.0]

                # PASS rope_ls
                grasp_idx_left = min(intersec_l_idx + 2, 29)
                target_idx_left = max(intersec_l_idx - 1, 1)
                grasp_pos_l_256 = rope_poses[0, grasp_idx_left]
                target_pos_l_256 = rope_poses[0, target_idx_left]
                print("grasping from idx:", grasp_idx_left, "to idx:", target_idx_left)
                print("grasp_pos_l_256:", grasp_pos_l_256, "target_pos_l_256:", target_pos_l_256)

                # --- CALCULATE DIRECTION VECTOR (from sim) ---
                direction_2d = target_pos_l_256 - grasp_pos_l_256
                norm_direction_2d = direction_2d / (np.linalg.norm(direction_2d) + 1e-6)
                direction_3d = np.array([norm_direction_2d[0], norm_direction_2d[1], 0.0]) # TODO maybe add a Z height for pass over
                # final_direction = direction_3d / (np.linalg.norm(direction_3d) + 1e-6)
                force_left = direction_3d.tolist()
                print("resulting in force direction", force_left)

            else: # solving rope_r
                # PIN rope_l (the "other" rope)
                grasp_idx_left = -1 # PIN (no movement for now)
                grasp_pos_l_256 = rope_poses[0, intersec_l_idx]
                force_left = [0.0, 0.0, 0.0]

                # PASS rope_r
                grasp_idx_right = min(intersec_l_idx + 2, 29)
                target_idx_right = max(intersec_r_idx - 1, 1)
                grasp_pos_r_256 = rope_poses[1, grasp_idx_right]
                target_pos_r_256 = rope_poses[1, target_idx_right]
                print("grasping from idx:", grasp_idx_right, "to idx:", target_idx_right)
                print("grasp_pos_r_256:", grasp_pos_r_256, "target_pos_r_256:", target_pos_r_256)

                # --- CALCULATE DIRECTION VECTOR (from sim) ---
                direction_2d = target_pos_r_256 - grasp_pos_r_256
                norm_direction_2d = direction_2d / (np.linalg.norm(direction_2d) + 1e-6)
                direction_3d = np.array([norm_direction_2d[0], norm_direction_2d[1], 0.0]) # TODO maybe add a Z height for pass over
                # final_direction = direction_3d / (np.linalg.norm(direction_3d) + 1e-6)
                force_right = direction_3d.tolist()
                print("resulting in force direction", force_right)
        
        else:
            raise ValueError(f"Unknown intersection_type: {intersection_type}")
        
        if np.isnan(force_left).any():
            print(f"NaN in force_left detected! Original value: {force_left}. Overwriting with zeros.")
            force_left = np.zeros_like(force_left)

        if np.isnan(force_right).any():
            print(f"NaN in force_right detected! Original value: {force_right}. Overwriting with zeros.")
            force_right = np.zeros_like(force_right)

        # Convert Force data from cv2 image coordinate system to robot waist frame
        force_left = [-force_left[1], -force_left[0], force_left[2]]
        force_right = [-force_right[1], -force_right[0], force_right[2]]

        # 8. Assemble and send the final request
        request_data = {
            'force_left': {'x': force_left[0], 'y': force_left[1], 'z': force_left[2]},
            'grasp_idx_left': grasp_idx_left,

            'force_right': {'x': force_right[0], 'y': force_right[1], 'z': force_right[2]},
            'grasp_idx_right': grasp_idx_right,

            'frame_id': 'WAIST'
        }
        
        send_service_request(ros_client, request_data)
            
    except KeyboardInterrupt:
        print("Shutting down via KeyboardInterrupt.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if sub:
            try:
                sub.close()
            except AttributeError as e:
                print(f"roslibpy internal error during ImageSubscriber.close() (ignoring): {e}")
        if ros_client:
            try:
                ros_client.terminate()
                print("Connection terminated.")
            except AttributeError as e:
                print(f"roslibpy internal error during main terminate (ignoring): {e}")
            
if __name__ == "__main__":
    main()