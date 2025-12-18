#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from nextage_custom_ik.nextage_kinematic import NextageKinematic
# ==========================================
# SETUP
# ==========================================
device = "cpu"
deg = torch.pi / 180.0

print("Initializing Solvers...")
larm = NextageKinematic(side='left', device=device)
rarm = NextageKinematic(side='right', device=device)

# Limits
lims_L = larm.joint_limits
lims_R = rarm.joint_limits

# Initial Values (from the class we just loaded)
th_wait_L_list = larm.th_rest.cpu().numpy()
th_wait_R_list = rarm.th_rest.cpu().numpy()

# ==========================================
# PLOTTING SETUP
# ==========================================
fig = plt.figure(figsize=(14, 10))
ax = fig.add_axes([0.05, 0.35, 0.9, 0.6], projection='3d')
ax.set_title("Nextage Kinematics (Split-Link Model)")
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_xlim(-0.2, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.1, 1.0)

# Torso
ax.plot([-0.05, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.6], c='blue', lw=8, alpha=0.3)

line_L, = ax.plot([], [], [], lw=4, c='black', marker='o')
line_R, = ax.plot([], [], [], lw=4, c='gray', marker='o')

# Frame Visualizers
def init_frame_lines(n_frames=16): # 16 is enough for the split links
    lines = []
    for _ in range(n_frames):
        lx, = ax.plot([], [], [], c='r', lw=2); ly, = ax.plot([], [], [], c='g', lw=2); lz, = ax.plot([], [], [], c='b', lw=2)
        lines.append((lx, ly, lz))
    return lines
frame_lines_L = init_frame_lines(); frame_lines_R = init_frame_lines()

def update_arm_viz(solver, th_list, line_bone, frames_lines):
    th_tensor = torch.tensor(th_list, device=device, dtype=torch.float32).unsqueeze(0)
    _, link_poses = solver.batch_FK(th_tensor)
    all_frames = [torch.eye(4).unsqueeze(0)] + link_poses
    
    # Bone Coords
    coords = np.array([H[0, :3, 3].detach().cpu().numpy() for H in all_frames])
    line_bone.set_data(coords[:, 0], coords[:, 1])
    line_bone.set_3d_properties(coords[:, 2])
    
    # Frames
    axis_len = 0.05
    for i, H in enumerate(all_frames):
        if i >= len(frames_lines): break
        mat = H[0].detach().cpu().numpy()
        pos = mat[:3, 3]; rot = mat[:3, :3]
        lx, ly, lz = frames_lines[i]
        lx.set_data([pos[0], pos[0]+rot[0,0]*axis_len], [pos[1], pos[1]+rot[1,0]*axis_len]); lx.set_3d_properties([pos[2], pos[2]+rot[2,0]*axis_len])
        ly.set_data([pos[0], pos[0]+rot[0,1]*axis_len], [pos[1], pos[1]+rot[1,1]*axis_len]); ly.set_3d_properties([pos[2], pos[2]+rot[2,1]*axis_len])
        lz.set_data([pos[0], pos[0]+rot[0,2]*axis_len], [pos[1], pos[1]+rot[1,2]*axis_len]); lz.set_3d_properties([pos[2], pos[2]+rot[2,2]*axis_len])

def update_plot(val=None):
    update_arm_viz(larm, [s.val for s in sliders_L], line_L, frame_lines_L)
    update_arm_viz(rarm, [s.val for s in sliders_R], line_R, frame_lines_R)
    fig.canvas.draw_idle()

sliders_L = []; sliders_R = []; names = ["Link 0", "Link 1", "Link 2", "Link 3", "Link 4", "Link 5"]
# Sliders: Right Arm on Left Side
for i in range(6):
    ax_s = fig.add_axes([0.1, 0.25 - i*0.035, 0.35, 0.025])
    s = Slider(ax_s, f'R_{names[i]}', lims_R['min'][i].item(), lims_R['max'][i].item(), valinit=th_wait_R_list[i]); s.on_changed(update_plot); sliders_R.append(s)
# Sliders: Left Arm on Right Side
for i in range(6):
    ax_s = fig.add_axes([0.6, 0.25 - i*0.035, 0.35, 0.025])
    s = Slider(ax_s, f'L_{names[i]}', lims_L['min'][i].item(), lims_L['max'][i].item(), valinit=th_wait_L_list[i]); s.on_changed(update_plot); sliders_L.append(s)

reset_ax = fig.add_axes([0.45, 0.02, 0.1, 0.04]); button = Button(reset_ax, 'Reset', hovercolor='0.975')
def reset(e): 
    for i, s in enumerate(sliders_L): s.set_val(th_wait_L_list[i])
    for i, s in enumerate(sliders_R): s.set_val(th_wait_R_list[i])
button.on_clicked(reset); update_plot(); plt.show()