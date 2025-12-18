import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
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

    def generate_gif(self, trajectory_list, filename="IK_motion.gif", fps=10):
        if (not trajectory_list) or (len(trajectory_list) == 0):
            print("No trajectory data provided for GIF generation.")
            return
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-0.2, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.1, 0.8)

        ims = []
        
        # Downsample
        step = max(1, len(trajectory_list) // 50) 
        
        for i in range(0, len(trajectory_list), step):
            th = trajectory_list[i]
            art = self.plot_arm(ax, th)
            ims.append(art)
            
        ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)
        ani.save(filename, writer='pillow')
        plt.close(fig)