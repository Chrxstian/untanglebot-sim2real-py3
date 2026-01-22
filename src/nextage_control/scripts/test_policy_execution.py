#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import random
import numpy as np
from nextage_control.srv import ExecutePolicyAction, ExecutePolicyActionRequest

def normalize_vector(vec):
    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return 0.0, 1.0, 0.0
    vec = vec / norm
    return vec[0], vec[1], vec[2]

def generate_random_direction():
    vec = np.array([random.uniform(-1, 1), random.uniform(0, 1), random.uniform(0, 1)])
    return normalize_vector(vec)

def test_policy_execution():
    rospy.init_node('test_policy_execution_node')
    
    USE_RANDOM = False
    # Predefined sequence of directions to compare different garments
    PREDEFINED_DIRECTIONS = [
        (-1, -1, 0),
        (0, -1, 0),
        (-1, 0, 0),
        (-1, 0, 1),
        (-1, 0, 1),
        (1, 0, 0),
        (1, 0, 0),
        (0, 0, -1),
        (0, 0, -1)
    ]
    
    service_name = '/execute_policy_action'
    rospy.loginfo("Waiting for service %s...", service_name)
    
    try:
        rospy.wait_for_service(service_name, timeout=5)
        execute_policy = rospy.ServiceProxy(service_name, ExecutePolicyAction)
    except rospy.ROSException:
        rospy.logerr("Service not available!")
        return

    if USE_RANDOM:
        iterations = range(5)
        rospy.loginfo("Running in RANDOM mode (5 steps).")
    else:
        iterations = range(len(PREDEFINED_DIRECTIONS))
        rospy.loginfo("Running in PREDEFINED sequence mode (%d steps).", len(PREDEFINED_DIRECTIONS))

    for i in iterations:
        if rospy.is_shutdown():
            break

        # 1. Get direction based on mode
        if USE_RANDOM:
            rx, ry, rz = generate_random_direction()
        else:
            raw_dir = PREDEFINED_DIRECTIONS[i]
            rx, ry, rz = normalize_vector(raw_dir)
        
        # 2. Construct the request
        req = ExecutePolicyActionRequest()
        req.Force_left.x = 0.0
        req.Force_left.y = 0.0
        req.Force_left.z = 0.0
        req.grasp_idx_left = 13
        
        req.Force_right.x = rx
        req.Force_right.y = ry
        req.Force_right.z = rz
        req.grasp_idx_right = 18
        
        req.frame = 'GYM'
        req.step_m = 0.0
        req.sleep_dt = 0.0
        req.max_steps = 0
        req.force_threshold = 30.0
        req.workspace_limits = False

        rospy.loginfo("--- Call #%d ---", i + 1)
        rospy.loginfo("Direction Right (Normalized): [x: %.2f, y: %.2f, z: %.2f]", rx, ry, rz)
        rospy.loginfo("Frame: %s", req.frame)

        # 3. Call Service
        try:
            resp = execute_policy(req)
            
            if resp.success:
                rospy.loginfo("Result: SUCCESS. Waiting 0.1 seconds...")
                rospy.sleep(0.1)
            else:
                rospy.logerr("Result: FAILURE. Stopping script.")
                break
                
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
            break

if __name__ == "__main__":
    test_policy_execution()