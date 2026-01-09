#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import rospy
import numpy as np
import torch
import time
import os
import rospkg
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

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('nextage_custom_ik')
        
        self.vis_dir = os.path.join(self.pkg_path, "vis")
        os.makedirs(self.vis_dir, exist_ok=True)

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
        self.current_force = {'left': np.zeros(3), 'right': np.zeros(3)}

        self.joint_names_map = {
            'left':  ['LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5'],
            'right': ['RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5']
        }
        
        self.pubs = {
            'left':  rospy.Publisher('/larm_controller/command', JointTrajectory, queue_size=1),
            'right': rospy.Publisher('/rarm_controller/command', JointTrajectory, queue_size=1)
        }

        rospy.Subscriber('/joint_states', JointState, self.cb_joints)
        rospy.Subscriber('/left/ft_gym_tared', WrenchStamped, self.cb_force_l)
        rospy.Subscriber('/right/ft_gym_tared', WrenchStamped, self.cb_force_r)

        self.srv = rospy.Service('/execute_force_pull', ExecuteForcePull, self.handle_pull)
        rospy.loginfo("[IK-Service] Service Ready: /execute_force_pull")

    def cb_joints(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos

    def cb_force_l(self, msg):
        if not self.monitoring_active:
            return
        current_force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        self.current_force['left'] = current_force
        self.current_force_mag['left'] = np.linalg.norm(current_force)
        
    def cb_force_r(self, msg):
        if not self.monitoring_active:
            return
        current_force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        self.current_force['right'] = current_force
        self.current_force_mag['right'] = np.linalg.norm(current_force)

    def clear_force_history(self):
        self.current_force = {'left': np.zeros(3), 'right': np.zeros(3)}
        self.current_force_mag = {'left': 0.0, 'right': 0.0}
        self.force_log_l = []
        self.force_log_r = []
        
    def get_arm_joints(self, arm):
        names = self.joint_names_map[arm]
        try:
            return [self.current_joints[n] for n in names]
        except KeyError:
            return None

    def handle_pull(self, req):
        self.clear_force_history()

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

        dt = 0.005
        pull_distance = 0.03
        pull_speed = 0.03
        
        rate = rospy.Rate(1.0 / dt)
        total_time = pull_distance / pull_speed
        steps = int(total_time / dt)
        
        mode_str = "DRY RUN PREVIEW" if req.dry_run else "REAL EXECUTION"
        rospy.loginfo(f"[IK-Service] {mode_str}: Pulling {arm} for {pull_distance:.2f}m ({steps} steps)")

        # State Reset
        if not req.dry_run:
            self.monitoring_active = True
        
        force_triggered = False
        ik_failed = False
        fail_reason = ""
        
        # --- Trajectory Generation Loop ---
        for i in range(steps):
            if rospy.is_shutdown(): break
                
            # 1. Force Check
            current_force_mag = self.current_force_mag[arm]
            if i % 20 == 0:
                self.force_log_l.append(self.current_force["left"].copy())
                self.force_log_r.append(self.current_force["right"].copy())

            if current_force_mag > req.force_threshold:
                force_triggered = True
                force_history_size = int(req.force_history_size)
                rospy.logwarn("[IK-Service] FORCE LIMIT TRIGGERED ({:.2f}N > {:.2f}N)".format(current_force_mag, req.force_threshold))
                rospy.logwarn("[IK-Service] Stopping Pull Action, padding force log to size {}.".format(req.force_history_size))
                if len(self.force_log_l) > 0:
                    last_force_l, last_force_r = self.force_log_l[-1].copy(), self.force_log_r[-1].copy()
                else:
                    last_force_l, last_force_r = np.zeros(3), np.zeros(3)
                # Padding of force log with last force value
                while len(self.force_log_l) < force_history_size:
                    self.force_log_l.append(last_force_l.copy())
                while len(self.force_log_r) < force_history_size:
                    self.force_log_r.append(last_force_r.copy())
                time.sleep(0.2)
                break

            # 2. FK & Target Generation
            current_pos_hom, _ = solver.batch_FK(q_curr)
            next_pos_hom = current_pos_hom.clone()
            
            # Move along vector
            move_step = torch.tensor(dir_vec * pull_speed * dt, dtype=torch.float32, device=self.device)
            next_pos_hom[0, :3, 3] += move_step
            
            # 3. IK Solve
            q_next = solver.batch_ik_newton(next_pos_hom, q_curr, limits=solver.joint_limits, max_iter=3)
            
            diff = q_next - q_curr
            diff_list = diff.squeeze().tolist()
            calculated_velocities = [d / dt for d in diff_list]
            calculated_velocities = [max(min(v, 2.0), -2.0) for v in calculated_velocities]

            # 4. Workspace Safety Check
            achieved_pos_hom, _ = solver.batch_FK(q_next)
            target_xyz = next_pos_hom[0, :3, 3]
            actual_xyz = achieved_pos_hom[0, :3, 3]
            error_dist = torch.norm(target_xyz - actual_xyz).item()
            
            if error_dist > 0.005:
                ik_failed = True
                fail_reason = f"Workspace Limit (Error: {error_dist*1000:.1f}mm)"
                rospy.logwarn(f"[IK-Service] {fail_reason}")
                break

            # 5. Execution vs Logging
            q_curr = q_next
            joint_log.append(q_curr.clone().detach().cpu())
            
            if not req.dry_run:
                # Pass calculated velocities to new publish_traj signature
                duration_padding = dt * 1.2
                self.publish_traj(arm, q_next[0].tolist(), calculated_velocities, duration_padding)
                rate.sleep()

        # Ensures robot doesn't drift after the loop finishes
        if not req.dry_run:
            final_pos = q_curr[0].tolist()
            zero_vel = [0.0] * len(final_pos)
            self.publish_traj(arm, final_pos, zero_vel, 0.5)
            rospy.loginfo("[IK-Service] Movement finished. Holding position.")

        # --- Finish & Response ---
        self.monitoring_active = False

        # Construct Message
        if ik_failed:
            final_msg = f"Failed: {fail_reason}"
            success = False
        elif force_triggered:
            final_msg = "Force Limit Exceeded"
            success = True
        else:
            final_msg = "Dry Run Success" if req.dry_run else "Goal Reached"
            success = True

        assert len(self.force_log_l) == req.force_history_size, "Left force log size mismatch"
        assert len(self.force_log_r) == req.force_history_size, "Right force log size mismatch"
        rospy.loginfo("returning force logs of shape L:{} R:{}".format(len(self.force_log_l), len(self.force_log_r)))

        return ExecuteForcePullResponse(
            success=success, 
            message=final_msg, 
            force_limit_met=force_triggered,
            force_log_left=[v for e in self.force_log_l for v in e],
            force_log_right=[v for e in self.force_log_r for v in e]
        )

    def publish_traj(self, arm, positions, velocities, duration):
        traj = JointTrajectory()
        traj.joint_names = self.joint_names_map[arm]
        traj.header.stamp = rospy.Time(0) # Execute ASAP
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = velocities
        point.time_from_start = rospy.Duration(duration)
        traj.points = [point]
        
        self.pubs[arm].publish(traj)

if __name__ == "__main__":
    IKPullService()
    rospy.spin()