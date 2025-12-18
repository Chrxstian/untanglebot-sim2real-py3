import torch

class ManageLinkParameter(object):
    def __init__(self, device="cpu"):
        self.link_list = []
        self.homs = []
        self.joint_label_list = []
        self.deg = torch.pi/180
        self._device = device

    def rot(self,u,th):
        """u∈R^3方向を軸にθ[rad]回転する行列 (Rodriguesの回転公式)"""
        u = torch.tensor(u, device=self._device).float()
        th = torch.tensor(th, device=self._device).float()
        u = u/torch.linalg.norm(u)
        a = torch.tensor([[0,-u[2],u[1]], [u[2],0,-u[0]], [-u[1],u[0],0]], device=self._device)
        u = u.reshape([3,1])
        return torch.cos(th)*torch.eye(3, device=self._device) + torch.sin(th)*a + (1-torch.cos(th))*u@u.T

    def _rt2hom(self,R,t):
        """姿勢R∈SO(3), 位置t∈R^3 から同次変換行列Hを算出"""
        H = torch.eye(4, device=self._device)
        H[:3,:3] = R
        H[:3, 3] = t
        return H

    def add_link(self,rot,torch_xyz,joint_label):
        self.link_list.append((rot, torch.tensor(torch_xyz, device=self._device)))
        self.joint_label_list.append(joint_label) # 'angle', 'liner', None
        # linerはx軸方向にスライドする

    def make_homs(self):
        self.homs = [self._rt2hom(R,t) for (R,t) in self.link_list]
        self.joint_num = len(self.homs)
        self.move_joint_num = self.joint_num - self.joint_label_list.count(None)