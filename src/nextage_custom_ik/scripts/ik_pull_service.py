#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import rospy
import numpy as np
import torch
import time
import math
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from nextage_custom_ik.srv import ExecuteForcePull, ExecuteForcePullResponse
from nextage_custom_ik.nextage_kinematic import NextageKinematic
from nextage_custom_ik.vis_utils import NextageVisualizer

class IKPullService:
    def __init__(self):
        rospy.init_node('ik_pull_service_node')
        self.device = "cpu"

        # 1. Initialize Kinematics Solvers and Visualizers
        rospy.loginfo("[IK-Service] Loading Kinematics Solvers...")
        self.solvers = {
            'left':  NextageKinematic(side='left', device=self.device),
            'right': NextageKinematic(side='right', device=self.device)
        }

        self.visualizers = {
            'left':  NextageVisualizer(self.solvers['left'], device=self.device),
            'right': NextageVisualizer(self.solvers['right'], device=self.device)
        }

        # 2. Setup ROS Interface
        self.current_joints = {}
        self.monitoring_active = False
        self.force_log_l = []
        self.force_log_r = []
        self.current_force_mag = {'left': 0.0, 'right': 0.0}

        self.joint_names_map = {
            'left':  ['LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5'],
            'right': ['RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5']
        }
        
        self.pubs = {
            'left':  rospy.Publisher('/larm_controller/command', JointTrajectory, queue_size=1),
            'right': rospy.Publisher('/rarm_controller/command', JointTrajectory, queue_size=1)
        }

        rospy.Subscriber('/joint_states', JointState, self.cb_joints)
        rospy.Subscriber('/l_wrist_ft/wrench_gym', WrenchStamped, self.cb_force_l)
        rospy.Subscriber('/r_wrist_ft/wrench_gym', WrenchStamped, self.cb_force_r)

        self.srv = rospy.Service('/execute_force_pull', ExecuteForcePull, self.handle_pull)
        rospy.loginfo("[IK-Service] Service Ready: /execute_force_pull")

    def cb_joints(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos

    def cb_force_l(self, msg):
        if not self.monitoring_active:
            return
        mag = math.sqrt(msg.wrench.force.x**2 +
                        msg.wrench.force.y**2 +
                        msg.wrench.force.z**2)
        self.current_force_mag['left'] = mag
        self.force_log_l.append([msg.header.stamp.to_sec(), msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

    def cb_force_r(self, msg):
        if not self.monitoring_active:
            return
        mag = math.sqrt(msg.wrench.force.x**2 +
                        msg.wrench.force.y**2 +
                        msg.wrench.force.z**2)
        self.current_force_mag['right'] = mag
        self.force_log_r.append([msg.header.stamp.to_sec(), msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

    def clear_force_history(self):
        self.force_log_l = []
        self.force_log_r = []

    def get_arm_joints(self, arm):
        names = self.joint_names_map[arm]
        try:
            return [self.current_joints[n] for n in names]
        except KeyError:
            return None

    def handle_pull(self, req):
        arm = req.arm.lower()
        if arm not in self.solvers:
            return ExecuteForcePullResponse(success=False, message="Invalid Arm ID", force_limit_met=False)

        solver = self.solvers[arm]
        q_curr_list = self.get_arm_joints(arm)
        joint_log = []

        if q_curr_list is None:
            return ExecuteForcePullResponse(success=False, message="No Joint States Received", force_limit_met=False)

        q_curr = torch.tensor(q_curr_list, dtype=torch.float32).unsqueeze(0)
        
        # Normalize Direction
        dir_vec = np.array(req.direction)
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-6:
             return ExecuteForcePullResponse(success=False, message="Zero Direction Vector", force_limit_met=False)
        dir_vec = dir_vec / norm

        # Loop Timing
        dt = 0.05 # 20Hz
        total_time = req.distance / req.speed
        steps = int(total_time / dt)
        
        rospy.loginfo("[IK-Service] Pulling {} for {:.2f}m at {:.2f}m/s ({} steps)".format(arm, req.distance, req.speed, steps))

        # Clear force log and start monitoring
        self.monitoring_active = True
        self.clear_force_history()
        force_triggered = False
        
        # --- Control Loop ---
        for _ in range(steps):
            if rospy.is_shutdown(): break

            # 1. Force Check
            current_force = self.current_force_mag[arm]

            if current_force > req.force_threshold:
                force_triggered = True
                rospy.logwarn("[IK-Service] FORCE LIMIT TRIGGERED ({:.2f}N > {:.2f}N)".format(current_force, req.force_threshold))
                break

            # 2. Predict Next Target (Cartesian)
            current_pos_hom, _ = solver.batch_FK(q_curr)
            
            next_pos_hom = current_pos_hom.clone()
            move_step = torch.tensor(dir_vec * req.speed * dt, dtype=torch.float32, device=self.device)
            
            # Add translation to position part of matrix
            next_pos_hom[0, :3, 3] += move_step
            
            # 3. Solve IK (Newton)
            q_next = solver.batch_ik_newton(next_pos_hom, q_curr, limits=solver.joint_limits, max_iter=3)
            
            # Verify the hand position after IK, if too far from target, abort (unreachable goal)
            achieved_pos_hom, _ = solver.batch_FK(q_next)
            target_xyz = next_pos_hom[0, :3, 3]
            actual_xyz = achieved_pos_hom[0, :3, 3]
            error_dist = torch.norm(target_xyz - actual_xyz).item()
            if error_dist > 0.005:
                rospy.logwarn(f"[IK-Service] Workspace Limit Reached - Planning failed! Tracking Error: {error_dist*1000:.2f} mm")
                rospy.logwarn(f"[IK-Service] Check ik_pull_{arm}_fail.gif for details.")

                output_file = f"./src/nextage_custom_ik/vis/ik_pull_{arm}_fail.gif"
                self.visualizers[arm].generate_gif(joint_log, filename=output_file, fps=int(1/dt))

                self.monitoring_active = False
                flat_log_l = [val for entry in self.force_log_l for val in entry]
                flat_log_r = [val for entry in self.force_log_r for val in entry]
                return ExecuteForcePullResponse(
                    success=False, 
                    message=f"Workspace Limit Reached (Error: {error_dist*1000:.1f}mm)", 
                    force_limit_met=False,
                    force_log_left=flat_log_l,
                    force_log_right=flat_log_r
                )

            # 4. Publish Command
            self.publish_traj(arm, q_next[0].tolist(), dt)
            
            # 5. Update State
            q_curr = q_next
            joint_log.append(q_curr.clone().detach().cpu())
            rospy.sleep(dt)

        # --- C. Finish ---
        msg = "Force Limit Exceeded" if force_triggered else "Goal Reached"
        
        # Visualize Trajectory
        # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        # output_file = "./src/nextage_custom_ik/vis/ik_pull_{}_{}.gif".format(arm, timestamp)
        # self.visualizers[arm].generate_gif(trajectory_list=joint_log, filename=output_file, fps=int(1/dt))

        # Send Response
        self.monitoring_active = False
        flat_log_l = [val for entry in self.force_log_l for val in entry]
        flat_log_r = [val for entry in self.force_log_r for val in entry]
        rospy.loginfo("Sending Pull Service Response")
        return ExecuteForcePullResponse(
            success=True, 
            message=msg, 
            force_limit_met=force_triggered,
            force_log_left=flat_log_l,
            force_log_right=flat_log_r
        )

    def publish_traj(self, arm, positions, duration):
        traj = JointTrajectory()
        traj.joint_names = self.joint_names_map[arm]
        traj.header.stamp = rospy.Time.now() + rospy.Duration(0.01)
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(duration)
        traj.points = [point]
        
        self.pubs[arm].publish(traj)

if __name__ == "__main__":
    IKPullService()
    rospy.spin()