import io
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import PIL.Image
from mpl_toolkits.mplot3d import Axes3D

class NextageVisualizer:
    def __init__(self, solver, device="cpu"):
        self.solver = solver
        self.device = device

    def plot_arm(self, ax, theta, color_bone='black'):
        # Compute FK for the specific frame
        # theta shape expected: (1, 6)
        if len(theta.shape) == 1:
            theta = theta.unsqueeze(0)
            
        _, link_poses = self.solver.batch_FK(theta)
        
        # Prepend Base (Identity) -> World Frame
        all_frames = [torch.eye(4, device=self.device).unsqueeze(0)] + link_poses
        
        coords = []
        frames_data = []
        
        for H_batch in all_frames:
            H = H_batch[0].detach().cpu().numpy()
            coords.append(H[:3, 3])
            frames_data.append(H)
        coords = np.array(coords)

        # Plot Bones
        line = ax.plot(coords[:,0], coords[:,1], coords[:,2], lw=4, c=color_bone, marker='o')
        
        # Plot Frames (RGB axis)
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