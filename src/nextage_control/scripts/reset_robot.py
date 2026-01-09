#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import rospy
import random
import numpy as np
from nextage_control.srv import ExecutePolicyAction, ExecutePolicyActionRequest

def reset_robot():
    rospy.init_node('reset_robot_node')
    
    service_name = '/execute_policy_action'
    rospy.loginfo("Waiting for service %s...", service_name)
    
    try:
        rospy.wait_for_service(service_name, timeout=5)
        execute_policy = rospy.ServiceProxy(service_name, ExecutePolicyAction)
    except rospy.ROSException:
        rospy.logerr("Service not available!")
        return
    
    # 2. Construct the request
    req = ExecutePolicyActionRequest()
    req.Force_left.x = 0.0
    req.Force_left.y = 0.0
    req.Force_left.z = 0.0
    req.grasp_idx_left = -1
    
    req.Force_right.x = 0.0
    req.Force_right.y = 0.0
    req.Force_right.z = 0.0
    req.grasp_idx_right = -1
    
    req.frame = 'WAIST'
    req.step_m = 0.0
    req.sleep_dt = 0.0
    req.max_steps = 0
    req.force_threshold = 50.0
    req.workspace_limits = False

    # 3. Call Service
    try:
        resp = execute_policy(req)
        
        if resp.success:
            rospy.loginfo("Result: SUCCESS. Waiting 0.1 seconds...")
            rospy.sleep(0.1)
            count += 1
        else:
            rospy.logerr("Result: FAILURE (%s). Stopping script.", resp.message)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

if __name__ == "__main__":
    reset_robot()