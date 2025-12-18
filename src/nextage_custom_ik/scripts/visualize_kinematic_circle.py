#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospkg
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from nextage_custom_ik.nextage_kinematic import NextageKinematic

rospack = rospkg.RosPack()
custom_ik_pkg_path = rospack.get_path('nextage_vision')

# ==========================================
# SETUP
# ==========================================
device = "cpu"
deg = torch.pi / 180.0

# 1. Initialize Dual Solvers
print("Initializing Solvers...")
larm = NextageKinematic(side='left', device=device)
rarm = NextageKinematic(side='right', device=device)

# 2. Setup Initial Guesses (Rest Poses)
th_left = larm.th_rest.clone().detach().unsqueeze(0)
th_right = rarm.th_rest.clone().detach().unsqueeze(0)

# ==========================================
# HELPER: PATH GENERATION (Circle)
# ==========================================
def phi2hom(phi, side='left', device="cpu"):
    """
    Generate circular path.
    Left Center:  x=0.3, y=0.2, z=0.15
    Right Center: x=0.3, y=-0.2, z=0.15 (Mirrored Y)
    """
    H = torch.eye(4, device=device).unsqueeze(0)
    
    # 1. Orientation Helper
    def rot_axis(u, angle):
        u = torch.tensor(u, device=device).float()
        u = u / torch.norm(u)
        if isinstance(angle, torch.Tensor):
            angle = angle.to(device=device, dtype=torch.float32)
        else:
            angle = torch.tensor(angle, device=device, dtype=torch.float32)
        c = torch.cos(angle)
        s = torch.sin(angle)
        K = torch.tensor([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]], device=device)
        return torch.eye(3, device=device) + s * K + (1 - c) * torch.matmul(K, K)

    # Orientation
    base_rot = rot_axis([1, 0, 0], 0*deg) 
    H[0, :3, :3] = base_rot

    # Position
    y_center = 0.2 if side == 'left' else -0.2
    center = torch.tensor([0.3, y_center, 0.15], device=device)
    radius = 0.10
    
    offset = torch.tensor([
        radius * torch.cos(phi), 
        radius * torch.sin(phi),
        0.0
    ], device=device)
    
    H[0, :3, 3] = center + offset
    return H

# ==========================================
# PLOTTING HELPER
# ==========================================
def plot_arm(ax, solver, theta, color_bone='black'):
    # Get Link Positions
    _, link_poses = solver.batch_FK(theta)
    
    # Prepend Base (Identity) -> World Frame
    all_frames = [torch.eye(4).unsqueeze(0)] + link_poses
    
    coords = []
    frames_data = []
    
    for H_batch in all_frames:
        H = H_batch[0].detach().cpu().numpy()
        coords.append(H[:3, 3])
        frames_data.append(H)
    coords = np.array(coords)

    # Plot Bones
    line = ax.plot(coords[:,0], coords[:,1], coords[:,2], lw=4, c=color_bone, marker='o')
    
    # Plot Frames
    lines = []
    for H in frames_data:
        pos = H[:3, 3]
        l = 0.05 
        ux = H[:3, 0]*l + pos; uy = H[:3, 1]*l + pos; uz = H[:3, 2]*l + pos
        
        l1 = ax.plot([pos[0], ux[0]], [pos[1], ux[1]], [pos[2], ux[2]], c='r', lw=2)[0]
        l2 = ax.plot([pos[0], uy[0]], [pos[1], uy[1]], [pos[2], uy[2]], c='g', lw=2)[0]
        l3 = ax.plot([pos[0], uz[0]], [pos[1], uz[1]], [pos[2], uz[2]], c='b', lw=2)[0]
        lines.extend([l1, l2, l3])
        
    return line + lines

# ==========================================
# MAIN LOOP
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (Forward)')
ax.set_ylabel('Y (Left)')
ax.set_zlabel('Z (Up)')
ax.set_xlim(-0.2, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.1, 0.8)

# Pre-calculate Paths for Visualization
print("Plotting Goal Paths...")
for s in ['left', 'right']:
    gp = []
    for phi_static in torch.arange(0, 361, 5) * deg:
        H_goal = phi2hom(phi_static, side=s, device=device)
        gp.append(H_goal[0, :3, 3].cpu().numpy())
    gp = np.array(gp)
    ax.plot(gp[:,0], gp[:,1], gp[:,2], 'k--', lw=1, alpha=0.5)

ims = []
print("Generating Animation...")

for phi in torch.arange(0, 360, 10) * deg:
    
    # --- LEFT ARM ---
    H_tgt_L = phi2hom(phi, side='left', device=device)
    th_left = larm.batch_ik_newton(H_tgt_L, th_left, limits=larm.joint_limits)
    
    # --- RIGHT ARM ---
    H_tgt_R = phi2hom(phi, side='right', device=device)
    th_right = rarm.batch_ik_newton(H_tgt_R, th_right, limits=rarm.joint_limits)
    
    # --- PLOT BOTH ---
    # Left Arm = Blueish/Black, Right Arm = Reddish/Grey (to distinguish)
    art_L = plot_arm(ax, larm, th_left, color_bone='black')
    art_R = plot_arm(ax, rarm, th_right, color_bone='gray')
    
    ims.append(art_L + art_R)
    
    if (phi/deg) % 40 == 0:
        print("Processed frame: {:.0f} deg".format(phi/deg))

print("Saving animation...")
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)

vis_dir = os.path.join(custom_ik_pkg_path, "vis")
os.makedirs(vis_dir, exist_ok=True)

ani.save(os.path.join(vis_dir, "nextage_dual_arm.gif"), writer='pillow')
print("Done.")
plt.show()