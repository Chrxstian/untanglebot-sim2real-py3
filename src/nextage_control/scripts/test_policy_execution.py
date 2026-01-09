#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import rospy
import random
import numpy as np
from nextage_control.srv import ExecutePolicyAction, ExecutePolicyActionRequest

def generate_random_direction():

    # return np.array([0, 1, 0])

    vec = np.array([random.uniform(-1, 1), random.uniform(0, 1), random.uniform(0, 1)])
    norm = np.linalg.norm(vec)
    
    if norm < 1e-6: # Prevent division by zero
        return 0.0, 1.0, 0.0
    
    # Return normalized vector
    vec = vec / norm
    return vec[0], vec[1], vec[2]

def test_policy_execution():
    rospy.init_node('test_policy_execution_node')
    
    service_name = '/execute_policy_action'
    rospy.loginfo("Waiting for service %s...", service_name)
    
    try:
        rospy.wait_for_service(service_name, timeout=5)
        execute_policy = rospy.ServiceProxy(service_name, ExecutePolicyAction)
    except rospy.ROSException:
        rospy.logerr("Service not available!")
        return

    count = 1
    
    # while not rospy.is_shutdown():
    while count <= 5:
    
        # 1. Generate arbitrary direction
        rx, ry, rz = generate_random_direction()
        
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
        
        req.frame = 'WAIST'
        req.step_m = 0.0
        req.sleep_dt = 0.0
        req.max_steps = 0
        req.force_threshold = 50.0
        req.workspace_limits = False

        rospy.loginfo("--- Call #%d ---", count)
        rospy.loginfo("Direction Right: [x: %.2f, y: %.2f, z: %.2f]", rx, ry, rz)

        # 3. Call Service
        try:
            resp = execute_policy(req)
            
            if resp.success:
                rospy.loginfo("Result: SUCCESS. Waiting 0.1 seconds...")
                rospy.sleep(0.1)
                count += 1
            else:
                rospy.logerr("Result: FAILURE. Stopping script.")
                break
                
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
            break

if __name__ == "__main__":
    test_policy_execution()