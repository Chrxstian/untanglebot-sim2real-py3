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

    def _create_animation(self, trajectory_list, fps):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xlim(-0.2, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.1, 0.8)

        ims = []
        step = max(1, len(trajectory_list) // 50) 
        for i in range(0, len(trajectory_list), step):
            art = self.plot_arm(ax, trajectory_list[i])
            ims.append(art)
            
        ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)
        return fig, ani

    def get_gif_bytes(self, trajectory_list, fps=30):
        if not trajectory_list: return None

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        frames = []
        step = max(1, len(trajectory_list) // 50)

        for i in range(0, len(trajectory_list), step):
            ax.cla()
            # Set limits and labels once per frame clear
            ax.set_xlim(-0.2, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.1, 0.8)
            self.plot_arm(ax, trajectory_list[i])
            
            # Draw and convert the canvas directly to a Pillow Image
            fig.canvas.draw()
            frames.append(PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), 
                                            fig.canvas.tostring_rgb()))

        plt.close(fig)
        if not frames: return None

        # Save the frame list to a single output buffer
        out = io.BytesIO()
        frames[0].save(out, format='GIF', save_all=True, append_images=frames[1:], 
                       duration=int(1000/fps), loop=0)
        return out.getvalue()

    def generate_gif(self, trajectory_list, filename, fps=30):
        """Generates the gif and saves it to a file."""
        if not trajectory_list: return
        
        fig, ani = self._create_animation(trajectory_list, fps)
        ani.save(filename, writer='pillow')
        plt.close(fig)