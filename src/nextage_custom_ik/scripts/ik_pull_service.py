#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import rospy
import numpy as np
import torch
import os
import math
import rospkg
from datetime import datetime
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
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

        # Force Sensor Tare Publisher
        self.tare_pub = rospy.Publisher("/ft/trigger_tare", Bool, queue_size=1)

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
        self.current_force_mag = {'left': 0.0, 'right': 0.0}
        self.force_log_l = []
        self.force_log_r = []
        
    def tare_ft_sensors(self):
        msg = Bool(True)
        rospy.loginfo("Sending FT tare trigger...")
        rospy.sleep(0.1)
        self.tare_pub.publish(msg)

    def get_arm_joints(self, arm):
        names = self.joint_names_map[arm]
        try:
            return [self.current_joints[n] for n in names]
        except KeyError:
            return None

    def handle_pull(self, req):
        self.tare_ft_sensors()
        
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
        rate = rospy.Rate(1.0 / dt)
        total_time = req.distance / req.speed
        steps = int(total_time / dt)
        
        mode_str = "DRY RUN PREVIEW" if req.dry_run else "REAL EXECUTION"
        rospy.loginfo(f"[IK-Service] {mode_str}: Pulling {arm} for {req.distance:.2f}m ({steps} steps)")

        # State Reset
        if not req.dry_run:
            self.monitoring_active = True
            self.clear_force_history()
        
        force_triggered = False
        ik_failed = False
        fail_reason = ""
        
        # --- Trajectory Generation Loop ---
        for i in range(steps):
            if rospy.is_shutdown(): break

            # 1. Force Check
            if not req.dry_run:
                current_force = self.current_force_mag[arm]
                if current_force > req.force_threshold:
                    force_triggered = True
                    rospy.logwarn(f"[IK-Service] FORCE LIMIT TRIGGERED ({current_force:.2f}N)")
                    break

            # 2. FK & Target Generation
            current_pos_hom, _ = solver.batch_FK(q_curr)
            next_pos_hom = current_pos_hom.clone()
            
            # Move along vector
            move_step = torch.tensor(dir_vec * req.speed * dt, dtype=torch.float32, device=self.device)
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
        output_gif = ""

        if req.dry_run or ik_failed:
            # Generate GIF for preview or failure analysis
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            timestamp_str = f"preview_{current_time}" if req.dry_run else f"fail_{current_time}"
            output_gif = os.path.join(self.vis_dir, f"ik_pull_{arm}_{timestamp_str}.gif")
            rospy.loginfo(f"[IK-Service] Generating GIF at {output_gif}...")
            self.visualizers[arm].generate_gif(joint_log, filename=output_gif, fps=30)

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

        return ExecuteForcePullResponse(
            success=success, 
            message=final_msg, 
            force_limit_met=force_triggered,
            gif_path=output_gif,
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