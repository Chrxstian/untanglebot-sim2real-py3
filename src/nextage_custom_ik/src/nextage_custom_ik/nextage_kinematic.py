import torch
import numpy as np
from nextage_custom_ik.manage_link_parameter import ManageLinkParameter
from nextage_custom_ik.robot_kinematic import RobotKinematic

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button

class NextageKinematic(RobotKinematic):
    def __init__(self, side='left', device="cpu"):
        self.side = side.lower()
        deg = torch.pi / 180.0
        
        # 6 Moving Joints (Chest is fixed)
        w_e_para = [1.0] * 6
        w_rest_para = [0.1] * 6
        
        # --- Joint Limits (extracted from urdf) ---
        if self.side == 'left':
            self.joint_limits = {
                'min': torch.tensor([-1.53, -2.44, -2.75, -1.83, -1.74, -2.84], device=device),
                'max': torch.tensor([ 1.53,  1.04,  0.00,  2.87,  1.74,  2.84], device=device)
            }
            # Wait Pose
            th_rest = [0.4769, -0.5823, -1.9836, -0.3516, 0.8405, -1.1423]
        else:
            self.joint_limits = {
                'min': torch.tensor([-1.53, -2.44, -2.75, -2.87, -1.74, -2.84], device=device),
                'max': torch.tensor([ 1.53,  1.04,  0.00,  1.83,  1.74,  2.84], device=device)
            }
            th_rest = [-0.4759, -0.5828, -1.9834, 0.3534, 0.8424, -1.1607]

        if isinstance(th_rest, torch.Tensor):
             th_rest = th_rest.clone().detach().to(device=device, dtype=torch.float32)
        else:
             th_rest = torch.tensor(th_rest, device=device, dtype=torch.float32)

        mlp = ManageLinkParameter(device)
        
        # KINEMATIC CHAIN

        # --- (-1) WORLD -> WAIST ---
        mlp.add_link(mlp.rot([1,0,0], 0), [-0.05, 0.0, 0.0], None)

        # --- 0. WAIST -> CHEST (FIXED) ---
        mlp.add_link(mlp.rot([1,0,0], 0), [0.0, 0.0, 0.0], None) 

        # --- 1. CHEST -> SHOULDER BASE (BONE) ---
        y_pos = 0.145 if self.side == 'left' else -0.145
        r_dir = -1.0 if self.side == 'left' else 1.0
        mlp.add_link(mlp.rot([1,0,0], r_dir * 0.261799), [0.0, y_pos, 0.370], 'angle')

        # --- 2. JOINT 1 (SHOULDER PITCH) ---
        # This joint rotates around the y axis but the framework only allows rotation around the z axis.
        # Therefore we rotate the link frame -90 deg around x, perform the rotation, then rotate back +90 deg around x using a dummy link.
        mlp.add_link(mlp.rot([1,0,0], -90*deg), [0.0, 0.0, 0.0], 'angle')
        mlp.add_link(mlp.rot([1,0,0], 90*deg), [0.0, 0.0, 0.0], None)

        # --- 3. JOINT 2 (ELBOW PITCH) ---
        # This joint rotates around the y axis but the framework only allows rotation around the z axis.
        # Therefore we rotate the link frame -90 deg around x, perform the rotation, then rotate back +90 deg around x using a dummy link.
        y_bone = 0.095 if self.side == 'left' else -0.095
        mlp.add_link(mlp.rot([1,0,0], -90*deg), [0.0, y_bone, -0.25], 'angle')
        mlp.add_link(mlp.rot([1,0,0], 90*deg), [0.0, 0.0, 0.0], None)

        # --- 4. JOINT 3 (ELBOW YAW) ---
        mlp.add_link(mlp.rot([1,0,0], 0), [-0.03, 0.0, 0.0], 'angle')

        # --- 5. JOINT 4 (WRIST PITCH) ---
        # This joint rotates around the y axis but the framework only allows rotation around the z axis.
        # Therefore we rotate the link frame -90 deg around x, perform the rotation, then rotate back +90 deg around x using a dummy link.
        mlp.add_link(mlp.rot([1,0,0], -90*deg), [0.0, 0.0, -0.235], 'angle')
        mlp.add_link(mlp.rot([1,0,0], 90*deg), [0.0, 0.0, 0.0], None)

        # --- 6. JOINT 5 (WRIST ROLL) ---
        # This joint rotates around the x axis but the framework only allows rotation around the z axis.
        # Therefore we rotate the link frame 90 deg around y, perform the rotation, then rotate back -90 deg around y using a dummy link.
        mlp.add_link(mlp.rot([0,1,0], 90*deg), [-0.047, 0.0, -0.09], 'angle')
        mlp.add_link(mlp.rot([0,1,0], -90*deg), [0.0, 0.0, 0.0], None)

        # --- 7. EEF ---
        # EEF has same orientation as joint 5, so we need to add the 90 deg rotation around y again.
        mlp.add_link(mlp.rot([0,1,0], 90*deg), [-0.20, 0.0, 0.0], None)
        
        mlp.make_homs()
        RobotKinematic.__init__(self, w_e_para, w_rest_para, th_rest, mlp, device)