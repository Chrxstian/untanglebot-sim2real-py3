#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import threading
import cv2
import numpy as np
import time
from sensor_msgs.msg import CompressedImage

class ImageSubscriber:
    """
    Standard subscriber for RGB CompressedImages (JPEG).
    """
    def __init__(self, topic_name):
        self.lock = threading.Lock()
        self.frame = None
        self.new_frame_event = threading.Event() # Event to signal new data
        self.sub = rospy.Subscriber(topic_name, CompressedImage, self._callback)

    def _callback(self, message):
        try:
            np_arr = np.frombuffer(message.data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is not None:
                with self.lock:
                    self.frame = img
                self.new_frame_event.set() # Signal that a new frame arrived
        except Exception as e:
            rospy.logerr(f"[ImageSubscriber] Error: {e}")

    def get_frame(self):
        """Non-blocking get."""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def wait_for_new_frame(self, timeout=2.0):
        """Blocking wait for the next frame."""
        # 1. Clear the event so we wait for a *future* callback
        self.new_frame_event.clear() 
        
        print("[ImageSubscriber] Waiting for fresh RGB frame...", end='', flush=True)
        
        # 2. Wait
        start = time.time()
        flag = self.new_frame_event.wait(timeout)
        elapsed = time.time() - start

        if flag:
            print(f" Arrived ({elapsed:.3f}s)")
            return self.get_frame()
        else:
            print(" Timeout!")
            return None

    def close(self):
        self.sub.unregister()


class DepthSubscriber:
    """
    Specialized subscriber for 'compressedDepth' topics.
    """
    def __init__(self, topic_name):
        self.lock = threading.Lock()
        self.frame = None
        self.new_frame_event = threading.Event()
        self.sub = rospy.Subscriber(topic_name, CompressedImage, self._callback)

    def _callback(self, message):
        try:
            data_bytes = message.data
            
            fmt = message.format
            if 'compressedDepth' in fmt and len(data_bytes) > 12:
                data_bytes = data_bytes[12:]
            
            np_arr = np.frombuffer(data_bytes, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            if img is not None:
                with self.lock:
                    self.frame = img
                self.new_frame_event.set()
        except Exception as e:
            rospy.logerr(f"[DepthSubscriber] Error: {e}")

    def get_frame(self):
        """Non-blocking get."""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()
    
    def wait_for_new_frame(self, timeout=2.0):
        """Blocking wait for the NEXT frame."""
        self.new_frame_event.clear()
        
        print("[DepthSubscriber] Waiting for fresh Depth frame...", end='', flush=True)
        
        start = time.time()
        flag = self.new_frame_event.wait(timeout)
        elapsed = time.time() - start
        
        if flag:
            print(f" Arrived ({elapsed:.3f}s)")
            return self.get_frame()
        else:
            print(" Timeout!")
            return None

    def close(self):
        self.sub.unregister()


class RGBDSubscriber:
    """
    Wrapper class to subscribe to both RGB and Depth topics.
    """
    def __init__(self, rgb_topic, depth_topic):
        self.rgb_sub = ImageSubscriber(rgb_topic)
        self.depth_sub = DepthSubscriber(depth_topic)

    def get_frames(self, timeout=2.0):
        print("\n[RGBDSubscriber] Syncing frames...")
        
        # 1. Wait for RGB
        rgb = self.rgb_sub.wait_for_new_frame(timeout)
        if rgb is None:
            print("[RGBDSubscriber] RGB Failed. Aborting.")
            return None, None
            
        # 2. Wait for Depth
        depth = self.depth_sub.wait_for_new_frame(timeout)
        if depth is None:
            print("[RGBDSubscriber] Depth Failed. Aborting.")
            return None, None
            
        return rgb, depth

    def close(self):
        self.rgb_sub.close()
        self.depth_sub.close()