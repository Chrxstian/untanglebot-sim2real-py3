#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def create_hsv_mask(hsv_image, mask_params_dict):
        """
        Creates a mask from an HSV image using a parameter dictionary
        loaded from the tuner's JSON config.
        
        :param hsv_image: The source image in HSV format.
        :param mask_params_dict: A dictionary from the JSON config, e.g.,
                                 {"h_low": 0, "s_low": 113, ...}
        :return: The final binary mask.
        """
        
        # 1. Unpack parameters from the dictionary
        try:
            lower_bound = np.array([
                mask_params_dict['h_low'],
                mask_params_dict['s_low'],
                mask_params_dict['v_low']
            ], dtype=np.uint8)
            
            upper_bound = np.array([
                mask_params_dict['h_high'],
                mask_params_dict['s_high'],
                mask_params_dict['v_high']
            ], dtype=np.uint8)
            
            erosion_iter = mask_params_dict['erosion']
            dilation_iter = mask_params_dict['dilation']
            
        except KeyError as e:
            print(f"ERROR: create_hsv_mask missing key {e} in parameters.")
            # Return an empty mask to avoid crashing
            return np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        # 2. Check for Hue wrap-around (e.g., for Red)
        # This happens if H_low (e.g., 170) > H_high (e.g., 10)
        if lower_bound[0] > upper_bound[0]:
            # Create two masks and combine them
            
            # Mask 1: from H_low to 179
            lower1 = lower_bound
            upper1 = np.array([179, upper_bound[1], upper_bound[2]])
            mask1 = cv2.inRange(hsv_image, lower1, upper1)
            
            # Mask 2: from 0 to H_high
            lower2 = np.array([0, lower_bound[1], lower_bound[2]])
            upper2 = upper_bound
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            
            mask_img = cv2.bitwise_or(mask1, mask2)
        else:
            # Normal case, no wrap-around
            mask_img = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # 3. Apply Erosion and Dilation
        kernel = np.ones((5, 5), np.uint8) # Use a consistent kernel
        
        if erosion_iter > 0:
            mask_img = cv2.erode(mask_img, kernel, iterations=erosion_iter)
        
        if dilation_iter > 0:
            mask_img = cv2.dilate(mask_img, kernel, iterations=dilation_iter)

        return mask_img
    
def create_intersection_mask(rope_l_mask, rope_r_mask, dilation_kernel_size=5):
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_l_mask = cv2.dilate(rope_l_mask.astype(np.uint8), kernel, iterations=1)
    dilated_r_mask = cv2.dilate(rope_r_mask.astype(np.uint8), kernel, iterations=1)
    interaction_mask = np.logical_and(dilated_l_mask, dilated_r_mask)
    return interaction_mask