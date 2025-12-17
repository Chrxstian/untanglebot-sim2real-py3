#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import torch
from datetime import datetime
from .mask_utils import create_hsv_mask

class ImageProcessor:
    def __init__(self, config_dir, use_depth=False):
        self.use_depth = use_depth
        self.mask_1_config_path = os.path.join(config_dir, "red_towel.json")
        self.mask_2_config_path = os.path.join(config_dir, "pink_towel.json")

        # Depth Params (MM)
        self.min_depth_mm = 600.0
        self.max_depth_mm = 800.0

        with open(self.mask_1_config_path, 'r') as f:
            self.mask_1_parameters = json.load(f) 
        with open(self.mask_2_config_path, 'r') as f:
            self.mask_2_parameters = json.load(f)

        self.img_dimensions = (256, 256)
        
        # Padding Config
        self.pads = {'top': 280, 'bottom': 280, 'left': 200, 'right': 200}
        
    def snap_to_mask(self, pixel_yx, mask):
        mask_coords_yx = np.argwhere(mask)
        if mask_coords_yx.shape[0] == 0:
            return pixel_yx
        distances = np.linalg.norm(mask_coords_yx - pixel_yx, axis=1)
        return mask_coords_yx[np.argmin(distances)]

    def snap_to_rope_id(self, pixel_xy, rope_poses_xy):
        if rope_poses_xy.shape[0] == 0:
            raise ValueError("rope_poses array is empty!")
        distances = np.linalg.norm(rope_poses_xy - pixel_xy, axis=1)
        return int(np.argmin(distances))

    def get_depth_at_point(self, pixel_xy, depth_map_mm):
        if depth_map_mm is None:
            return 0.0
        x, y = int(pixel_xy[0]), int(pixel_xy[1])
        h, w = depth_map_mm.shape
        if 0 <= x < w and 0 <= y < h:
            depth_mm = depth_map_mm[y, x]
            return float(depth_mm) / 1000.0
        return 0.0

    def process_image(self, image_bgr, timestamp=None, depth_img=None, vis_dir=None):
        """
        Returns:
            tensor: (256, 256, C) for NN
            resized_depth_mm: (256, 256) raw depth in mm (or None)
            segmentation_bitmap: (256, 256, 2) numpy array of masks
        """
        # --- 1. Resize & Pad RGB ---
        pad_top, pad_bot = 200+80, 200+80
        pad_left, pad_right = 200, 200
        
        image_enlarged = cv2.copyMakeBorder(
            image_bgr, pad_top, pad_bot, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        image_resized = cv2.resize(image_enlarged, self.img_dimensions, interpolation=cv2.INTER_AREA)
        
        # Capture precise timestamp for dataset pairing
        real_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

        if timestamp and vis_dir:
            # 1. Standard Visualization
            cv2.imwrite(os.path.join(vis_dir, timestamp, "resized_image_input.png"), image_resized)
            
            # 2. Cumulative Dataset Save (RGB Image)
            test_rgbs_dir = os.path.join(vis_dir, "test_rgbs")
            os.makedirs(test_rgbs_dir, exist_ok=True)
            cv2.imwrite(os.path.join(test_rgbs_dir, f"{real_timestamp}.png"), image_resized)

        # --- 2. Generate Masks (HSV) ---
        image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        mask_1 = create_hsv_mask(image_hsv, self.mask_1_parameters)
        mask_2 = create_hsv_mask(image_hsv, self.mask_2_parameters)
        
        # Combined mask
        combined_mask = cv2.bitwise_or(mask_1, mask_2)

        if timestamp and vis_dir:
             masked_1 = cv2.bitwise_and(image_resized, image_resized, mask=mask_1)
             cv2.imwrite(os.path.join(vis_dir, timestamp, "mask_1.png"), masked_1)

        # --- 3. Resize & Pad Depth (if enabled) ---
        depth_tensor_channel = None
        resized_depth_mm = None
        
        if self.use_depth:
            if depth_img is not None:
                depth_enlarged = cv2.copyMakeBorder(
                    depth_img, pad_top, pad_bot, pad_left, pad_right, 
                    cv2.BORDER_REPLICATE
                )
                resized_depth_mm = cv2.resize(depth_enlarged, self.img_dimensions, interpolation=cv2.INTER_NEAREST)
                depth_flattened = resized_depth_mm.copy()
                depth_flattened[combined_mask == 0] = self.max_depth_mm
                depth_float = depth_flattened.astype(np.float32)
                depth_normalized = np.zeros_like(depth_float)
                depth_normalized = (depth_float - self.min_depth_mm) / (self.max_depth_mm - self.min_depth_mm)
                depth_tensor_channel = np.clip(depth_normalized, 0.0, 1.0)
                if timestamp and vis_dir:
                    vis_depth_u8 = (depth_tensor_channel * 255.0).astype(np.uint8)
                    vis_depth_color = cv2.applyColorMap(vis_depth_u8, cv2.COLORMAP_TURBO)
                    cv2.imwrite(os.path.join(vis_dir, timestamp, "depth_input_vis.png"), vis_depth_color)
            else:
                depth_tensor_channel = np.zeros(self.img_dimensions, dtype=np.float32)
                resized_depth_mm = np.zeros(self.img_dimensions, dtype=np.uint16)

        # --- 4. Stack & Return ---
        mask_1_norm = (mask_1 / 255.0).astype(np.float32)
        mask_2_norm = (mask_2 / 255.0).astype(np.float32)
        
        segmentation_bitmap = np.dstack((mask_1_norm, mask_2_norm))

        if self.use_depth:
            final_stack = np.dstack((mask_1_norm, mask_2_norm, depth_tensor_channel))
        else:
            final_stack = segmentation_bitmap

        final_tensor = torch.from_numpy(final_stack)

        # --- 5. Save Final Tensor ---
        if timestamp and vis_dir:
            test_rgbs_dir = os.path.join(vis_dir, "test_rgbs")
            os.makedirs(test_rgbs_dir, exist_ok=True)
            
            # Save using the exact same timestamp as the PNG above
            tensor_path = os.path.join(test_rgbs_dir, f"{real_timestamp}.pth")
            torch.save(final_tensor, tensor_path)

        return final_tensor, resized_depth_mm, segmentation_bitmap