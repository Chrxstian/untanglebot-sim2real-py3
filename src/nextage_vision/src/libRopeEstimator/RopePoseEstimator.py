#!/usr/bin/env python3

from datetime import datetime
import os
import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.ops
from . import config
from .model import TangledRopeKeypointModel
import numpy as np
import roslibpy
import rospkg

class RopePoseEstimator():
    def __init__(self, save_visualizations=False, num_balls=30, ros_client=None, use_depth=False):
        
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('nextage_vision')
        checkpoints_dir = os.path.join(package_path, "src/libRopeEstimator/checkpoints")
        vis_dir_path = os.path.join(package_path, "vis")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_width = 256
        self.num_balls = num_balls

        checkpoint_files = os.listdir(checkpoints_dir)
        assert len(checkpoint_files) == 1, f"Please ensure there is exactly one checkpoint in {checkpoints_dir}. Found: {checkpoint_files}"
        checkpoint_name = checkpoint_files[0]
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)

        # Initialize ROS client
        self.ros_client = ros_client

        # Initialize the visualization directory
        self.save_visualizations = save_visualizations
        if self.save_visualizations:
            self.vis_dir = vis_dir_path
            os.makedirs(self.vis_dir, exist_ok=True)

        # Flag if we are using the depth channel together with segmentation (256, 256, 3) instead of (256, 256, 2)
        self.use_depth = use_depth

        # Initialize the model
        self.model = TangledRopeKeypointModel()
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Saving the last model output
        self.last_model_output = None
        
        # Initialize ROS publisher for publishing the rope pose estimations
        self.publisher = None
        if ros_client and ros_client.is_connected:
            try:
                self.publisher = roslibpy.Topic(ros_client, '/rope_pose_estimations', 'std_msgs/Float64MultiArray')
                print("RopePoseEstimator: Publisher for /rope_pose_estimations initialized.")
            except Exception as e:
                print(f"RopePoseEstimator: Failed to create publisher: {e}")
        else:
            print("RopePoseEstimator: ros_client not provided or not connected. Publisher will not be created.")

        # Load the model weights
        print("Loaded RopePoseEstimator model from {}".format(checkpoint_path))

    def choose_solving_rope(self, rope_poses):
        # For now, we chose the solving rope by the quality of the rope id estimations.
        # Return the rope with the smaller average distance between consecutive balls.
        avg_distances = []
        for i in range(2):
            rope = rope_poses[i]
            diffs = np.linalg.norm(rope[1:] - rope[:-1], axis=1)
            avg_distance = np.mean(diffs)
            avg_distances.append(avg_distance)
        best_rope_index = np.argmin(avg_distances)
        print("solving rope index:", best_rope_index, "with avg distance:", avg_distances[best_rope_index], "vs", avg_distances[1 - best_rope_index])
        return best_rope_index

    def get_pos_for_id(self, target_id, length_map_channel_np, value_tolerance=0.05):
        # Estimates the (y, x) pixel coordinate for a target ball ID
        
        # --- Step 1: Convert the target ID back to a target float value ---
        target_float_value = target_id / (self.num_balls - 1.0)

        # --- Step 2: Get all valid pixels and their values ---
        valid_mask = (length_map_channel_np > -1.0)
        valid_coords_yx = np.argwhere(valid_mask)
        
        if valid_coords_yx.size == 0:
            print("Warning: No rope pixels found in length map. Returning [0, 0].")
            return np.array([0, 0], dtype=np.float32)

        valid_values_on_map = length_map_channel_np[valid_mask]

        # --- Step 3: Find ALL pixels within the tolerance ---
        value_errors = np.abs(valid_values_on_map - target_float_value)
        matching_indices = np.where(value_errors <= value_tolerance)[0]

        # --- Step 4: Calculate centroid and find closest real pixel ---
        
        if matching_indices.size > 0:
            # Get all matching (y, x) coordinates
            coords_to_average_yx = valid_coords_yx[matching_indices]
            
            # Calculate the mean position (centroid)
            mean_coord_yx = np.mean(coords_to_average_yx, axis=0)
            
            # Find the pixel in our cluster that is spatially
            # closest to this calculated mean coordinate.
            distances = np.linalg.norm(coords_to_average_yx - mean_coord_yx, axis=1)
            
            # Find the index of the pixel with the minimum distance
            closest_index_in_cluster = np.argmin(distances)
            
            # Get the coordinate of that pixel
            final_coord_yx = coords_to_average_yx[closest_index_in_cluster]
        else:
            # Fallback: No pixels were inside the tolerance... return the single closest pixel (original behavior).
            print("Warning: No pixels found within tolerance {} for ID {}. Returning single closest pixel.".format(value_tolerance, target_id))
            best_pixel_index = np.argmin(value_errors)
            final_coord_yx = valid_coords_yx[best_pixel_index]

        return final_coord_yx.astype(np.float32)
    
    def estimate_rope_pose(self, segmentation_tensor, timestamp=None):
    # Input image tensor bitmap of shape (256, 256, 2) or (256, 256, 3) with dtype=float32

        print("segmentation tensor of dtype:", segmentation_tensor.dtype, "and shape:", segmentation_tensor.shape)
        print("unique values of depth channel", np.unique(segmentation_tensor[:, :, 2]))

        if timestamp is None and self.save_visualizations:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(self.vis_dir + "/" + timestamp, exist_ok=True)

        mask_channels = segmentation_tensor[:, :, :2] 
        rope_l_mask = mask_channels[:, :, 0] == 1
        rope_r_mask = mask_channels[:, :, 1] == 1

        model_input_tensor = segmentation_tensor.permute(2, 0, 1).to(self.device).float().unsqueeze(0) # (1, 2, 256, 256)
        
        with torch.no_grad():
            model_output = self.model(model_input_tensor.to(config.DEVICE))
            self.last_model_output = model_output
            
            pred_length_map = model_output['length_maps'].squeeze(0).permute(1, 2, 0).cpu() # (256, 256, 2)
            pred_length_map = torch.where((mask_channels == 1), pred_length_map, -1.0)
            pred_special_points_yx = model_output['coordinates'].squeeze(0).cpu().numpy().astype(np.int32)
            
        cmap_red = plt.get_cmap('Reds')
        cmap_green = plt.get_cmap('Greens')
        seg_image_np = np.zeros((self.image_width, self.image_width, 3), dtype=np.float32)
        values_l = pred_length_map[:, :, 0].clamp(0, 1).numpy()
        values_r = pred_length_map[:, :, 1].clamp(0, 1).numpy()
        colors_l = cmap_red(values_l)[:, :, :3]
        colors_r = cmap_green(values_r)[:, :, :3]
        seg_image_np[rope_l_mask] = colors_l[rope_l_mask]
        seg_image_np[rope_r_mask] = colors_r[rope_r_mask]
        seg_image_np = (seg_image_np * 255).astype(np.uint8)

        pose_estimations = np.zeros((2, self.num_balls, 2), dtype=np.float32)
        for i in range(self.num_balls):
            pose_estimations[0, i, :] = self.get_pos_for_id(i, pred_length_map[:, :, 0].numpy(), value_tolerance=0.05)
            pose_estimations[1, i, :] = self.get_pos_for_id(i, pred_length_map[:, :, 1].numpy(), value_tolerance=0.05)

        if self.save_visualizations:
            
            # 1. Extract points for polylines
            l_points_yx = pose_estimations[0, :, :].astype(np.int32)
            r_points_yx = pose_estimations[1, :, :].astype(np.int32)
            l_points_xy = l_points_yx[:, ::-1]
            r_points_xy = r_points_yx[:, ::-1]

            # 2. Define bgr colors (to match cv2's drawing)
            color_l_poly = (0, 0, 255)
            color_r_poly = (255, 255, 0)
            
            color_l_end = (0, 0, 255)
            color_l_int = (0, 255, 0)
            color_r_end = (255, 0, 0)
            color_r_int = (0, 255, 255)

            # 3. Draw polylines
            cv2.polylines(seg_image_np, [l_points_xy], isClosed=False, color=color_l_poly, thickness=1)
            cv2.polylines(seg_image_np, [r_points_xy], isClosed=False, color=color_r_poly, thickness=1)

            # 4. Draw small circles for all points
            for pt in l_points_xy:
                cv2.circle(seg_image_np, tuple(pt), radius=1, color=color_l_poly, thickness=-1)
            for pt in r_points_xy:
                cv2.circle(seg_image_np, tuple(pt), radius=1, color=color_r_poly, thickness=-1)

            # Get points (y, x)
            endpoint_l_yx = pred_special_points_yx[0]
            intersec_l_yx = pred_special_points_yx[1]
            endpoint_r_yx = pred_special_points_yx[2]
            intersec_r_yx = pred_special_points_yx[3]
            
            # Draw (x, y)
            cv2.circle(seg_image_np, (endpoint_l_yx[1], endpoint_l_yx[0]), radius=3, color=color_l_end, thickness=-1)
            cv2.circle(seg_image_np, (intersec_l_yx[1], intersec_l_yx[0]), radius=3, color=color_l_int, thickness=-1)
            cv2.circle(seg_image_np, (endpoint_r_yx[1], endpoint_r_yx[0]), radius=3, color=color_r_end, thickness=-1)
            cv2.circle(seg_image_np, (intersec_r_yx[1], intersec_r_yx[0]), radius=3, color=color_r_int, thickness=-1)

            # 6. Create the Legend
            legend_width = 150
            legend_canvas = np.zeros((self.image_width, legend_width, 3), dtype=np.uint8)
            text_color = (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            
            # Legend items
            items = [
                ("L-End (NN)", color_l_end),
                ("L-Int (NN)", color_l_int),
                ("R-End (NN)", color_r_end),
                ("R-Int (NN)", color_r_int),
                ("L-Rope (Est)", color_l_poly),
                ("R-Rope (Est)", color_r_poly)
            ]
            
            y_pos = 20
            for (text, color) in items:
                # Draw color circle
                cv2.circle(legend_canvas, (15, y_pos), radius=3, color=color, thickness=-1)
                # Draw text
                cv2.putText(legend_canvas, text, (30, y_pos + 4), font, font_scale, text_color, 1, cv2.LINE_AA)
                y_pos += 20 # Move down for next item

            # 7. Combine image and legend
            final_vis_image = np.hstack((seg_image_np, legend_canvas))

            # 8. Save with current timestamp
            vis_path = os.path.join(self.vis_dir, timestamp, "length_map_vis_with_legend.png")
            print("saving image at", vis_path)
            cv2.imwrite(vis_path, cv2.cvtColor(final_vis_image, cv2.COLOR_RGB2BGR))
            
        pose_estimations = pose_estimations[:, :, ::-1]  # Convert (y, x) to (x, y)
            
        if self.publisher:
            try:
                data_list = pose_estimations.flatten().tolist()
                msg = {'data': data_list}
                self.publisher.publish(msg)
                print("Successfully published (2, 30, 2) pose estimations to /rope_pose_estimations")
            except Exception as e:
                print(f"Failed to publish pose estimations: {e}")
        else:
            print("Pose estimation publisher not initialized. Skipping publish.")
            
        return pose_estimations
