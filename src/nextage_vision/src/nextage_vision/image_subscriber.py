#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import roslibpy
import threading
import base64
import cv2
import numpy as np
import time

class ImageSubscriber:
    """
    Standard subscriber for RGB CompressedImages (JPEG).
    """
    def __init__(self, client, topic_name):
        self.client = client
        self.lock = threading.Lock()
        self.frame = None
        self.new_frame_event = threading.Event() # Event to signal new data

        self.topic = roslibpy.Topic(client, topic_name, 'sensor_msgs/CompressedImage')
        self.topic.subscribe(self._callback)

    def _callback(self, message):
        try:
            data_str = message['data']
            data_bytes = base64.b64decode(data_str)
            np_arr = np.frombuffer(data_bytes, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is not None:
                with self.lock:
                    self.frame = img
                self.new_frame_event.set() # Signal that a new frame arrived
        except Exception as e:
            print(f"[ImageSubscriber] Error: {e}")

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
        self.topic.unsubscribe()


class DepthSubscriber:
    """
    Specialized subscriber for 'compressedDepth' topics.
    """
    def __init__(self, client, topic_name):
        self.client = client
        self.lock = threading.Lock()
        self.frame = None
        self.new_frame_event = threading.Event()

        self.topic = roslibpy.Topic(client, topic_name, 'sensor_msgs/CompressedImage')
        self.topic.subscribe(self._callback)

    def _callback(self, message):
        try:
            data_str = message['data']
            data_bytes = base64.b64decode(data_str)
            
            fmt = message.get('format', '')
            if 'compressedDepth' in fmt and len(data_bytes) > 12:
                data_bytes = data_bytes[12:]
            
            np_arr = np.frombuffer(data_bytes, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            if img is not None:
                with self.lock:
                    self.frame = img
                self.new_frame_event.set()
        except Exception as e:
            print(f"[DepthSubscriber] Error: {e}")

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
        self.topic.unsubscribe()


class RGBDSubscriber:
    """
    Wrapper class to subscribe to both RGB and Depth topics.
    """
    def __init__(self, client, rgb_topic, depth_topic):
        self.rgb_sub = ImageSubscriber(client, rgb_topic)
        self.depth_sub = DepthSubscriber(client, depth_topic)

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