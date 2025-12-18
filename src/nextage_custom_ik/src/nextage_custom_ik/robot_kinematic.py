import torch
import functorch    

class RobotKinematic(object):
    # --- 修正箇所: 変数名を分かりやすく変更 ---
    def __init__(self, w_q_para, w_rest_para, th_rest, mlp, device="cpu"):
        self._device = device
        self.mlp = mlp
        self.set_joint_space_weight(w_q_para) # w_e -> w_q
        self.set_rest_parameters(w_rest_para, th_rest)

    def set_joint_space_weight(self, w_q_para):
        self.w_q = torch.diag(torch.tensor(w_q_para, device=self._device, dtype=torch.float32))

    def set_rest_parameters(self, w_rest, th_rest):
        self.w_rest = torch.diag(torch.tensor(w_rest, device=self._device, dtype=torch.float32))
        if isinstance(th_rest, torch.Tensor):
            self.th_rest = th_rest.clone().detach().to(device=self._device, dtype=torch.float32)
        else:
            self.th_rest = torch.tensor(th_rest, device=self._device, dtype=torch.float32)

    def _mat2vec_12d(self, m):
        t = m[:3, 3]; r_flat = m[:3, :3].T.flatten()
        return torch.cat([t, r_flat])

    # def _batch_mat2vec_12d(self, m):
    #     t = m[:, :3, 3]; r_flat = m[:, :3, :3].transpose(1,2).reshape(m.shape[0], 9)
    #     return torch.cat([t, r_flat], dim=1)
    def _batch_mat2vec_12d(self, m):  # 4x4行列 m=H or m=dH・H^-1 の6Dベクトル表示
        v = torch.ones([m.shape[0],12], device=self._device)#
        v[:,:3] = m[:,:3,3]
        v[:,3:6] = m[:,0,:3]
        v[:,6:9] = m[:,1,:3]
        v[:,9:] = m[:,2,:3]
        return v

    def _Rz(self, th):
        c, s = torch.cos(th), torch.sin(th)
        zero = torch.zeros_like(th)
        one = torch.ones_like(th)
        row1 = torch.stack([c, -s, zero, zero])
        row2 = torch.stack([s,  c, zero, zero])
        row3 = torch.stack([zero, zero, one, zero])
        row4 = torch.stack([zero, zero, zero, one])
        return torch.stack([row1, row2, row3, row4])

    def _linerRx(self, th):
        zero = torch.zeros_like(th)
        one = torch.ones_like(th)
        row1 = torch.stack([one, zero, zero, th])
        row2 = torch.stack([zero, one, zero, zero])
        row3 = torch.stack([zero, zero, one, zero])
        row4 = torch.stack([zero, zero, zero, one])
        return torch.stack([row1, row2, row3, row4])

    def _batch_Rz(self,th):
        c,s = torch.cos(th),torch.sin(th)
        m = torch.eye(4, dtype=th.dtype, device=self._device).repeat(th.shape[0], 1, 1)
        m[:,0,0] = c; m[:,0,1] = -s
        m[:,1,0] = s; m[:,1,1] = c
        return m

    def _batch_linerRx(self,th):
        m = torch.eye(4, dtype=th.dtype, device=self._device).repeat(th.shape[0], 1, 1)
        m[:,0,3] = th
        return m
    
    def view_FK(self, th, k):
        H = torch.eye(4, dtype=th.dtype, device=self._device); th_idx = 0
        for j in range(k):
            if j >= self.mlp.joint_num: break 
            
            H_link = self.mlp.homs[j].to(H.dtype)
            label = self.mlp.joint_label_list[j]
            
            if label == 'angle': 
                H_motion = self._Rz(th[th_idx]); th_idx += 1
                H = H @ H_link @ H_motion
            elif label == 'liner': 
                H_motion = self._linerRx(th[th_idx]); th_idx += 1
                H = H @ H_link @ H_motion
            else: # label is None
                H = H @ H_link
        return H

    def batch_FK(self, th, end_effector_idx=None):
        if end_effector_idx is None:
            end_effector_idx = self.mlp.joint_num

        H = torch.eye(4, dtype=th.dtype, device=self._device).repeat(th.shape[0], 1, 1)
        th_idx = 0
        link_poses = []
        for j in range(end_effector_idx):
            H = torch.matmul(H, self.mlp.homs[j].to(H.dtype).repeat(th.shape[0], 1, 1))
            label = self.mlp.joint_label_list[j]
            if label == 'angle':
                H = torch.matmul(H, self._batch_Rz(th[:,th_idx]))
                th_idx += 1
            elif label == 'liner':
                H = torch.matmul(H, self._batch_linerRx(th[:,th_idx]))
                th_idx += 1
            link_poses.append(H)
        return H, link_poses

    def batch_analytical_jacobian_12d(self, th):
        batch_size = th.shape[0]
        # FKを計算して、最終的なEE姿勢と、各関節が動き終わった"後"のリンク姿勢を取得
        ee_pose, link_tip_poses = self.batch_FK(th)
        p_e = ee_pose[:, :3, 3]; R_e = ee_pose[:, :3, :3]

        J_12d = torch.zeros(batch_size, 12, self.mlp.move_joint_num, dtype=th.dtype, device=self._device)
        th_idx = 0
        
        # 逐次的にFKを再計算しながら、各関節の「動き始める直前」の座標系を求める
        H_cumulative = torch.eye(4, dtype=th.dtype, device=self._device).repeat(batch_size, 1, 1)
        fk_th_idx = 0

        for j in range(self.mlp.joint_num):
            # 静的変換を適用
            H_cumulative = torch.matmul(H_cumulative, self.mlp.homs[j].to(th.dtype).repeat(batch_size, 1, 1))
            
            # この H_cumulative が、関節 j の動きの基準となる座標系の姿勢
            motion_base_frame = H_cumulative

            label = self.mlp.joint_label_list[j]
            if label is not None:
                p_j = motion_base_frame[:, :3, 3]
                x_j = motion_base_frame[:, :3, 0]
                z_j = motion_base_frame[:, :3, 2]

                if label == 'angle': # 回転関節 (ローカルZ軸まわり)
                    motion_axis = z_j
                    J_12d[:, :3, th_idx] = torch.cross(motion_axis, p_e - p_j, dim=1)
                    skew_z = torch.zeros(batch_size, 3, 3, dtype=th.dtype, device=self._device)
                    skew_z[:, 0, 1] = -motion_axis[:, 2]; skew_z[:, 0, 2] = motion_axis[:, 1]
                    skew_z[:, 1, 0] = motion_axis[:, 2]; skew_z[:, 1, 2] = -motion_axis[:, 0]
                    skew_z[:, 2, 0] = -motion_axis[:, 1]; skew_z[:, 2, 1] = motion_axis[:, 0]
                    dR_dth = torch.bmm(skew_z, R_e)
                    J_12d[:, 3:, th_idx] = dR_dth.reshape(batch_size, 9)

                elif label == 'liner': # 直動関節 (ローカルX軸まわり)
                    motion_axis = x_j
                    J_12d[:, :3, th_idx] = motion_axis

                th_idx += 1

            # 次の基準座標系を計算するために、この関節の動きを累積する
            if label == 'angle': 
                H_cumulative = torch.matmul(H_cumulative, self._batch_Rz(th[:, fk_th_idx]))
                fk_th_idx += 1
            elif label == 'liner': 
                H_cumulative = torch.matmul(H_cumulative, self._batch_linerRx(th[:, fk_th_idx]))
                fk_th_idx += 1

        return J_12d

    def batch_inverse_kinematics(self, target_pos_batch, initial_theta_batch, max_iter=20, lr=0.5):
        th_batch = initial_theta_batch.clone()
        p_target = target_pos_batch[:, :3, 3]
        R_target = target_pos_batch[:, :3, :3]
        for _ in range(max_iter):
            current_pos, _ = self.batch_FK(th_batch)
            p_current = current_pos[:, :3, 3]
            R_current = current_pos[:, :3, :3]
            error_p = p_target - p_current
            error_o = 0.5 * (torch.cross(R_current[:,:,0], R_target[:,:,0], dim=1) +
                               torch.cross(R_current[:,:,1], R_target[:,:,1], dim=1) +
                               torch.cross(R_current[:,:,2], R_target[:,:,2], dim=1))
            error_vec = torch.cat([error_p, error_o], dim=1)
            J_batch = self.batch_geometric_jacobian(th_batch)
            lambda_val = 0.1
            JJT = torch.bmm(J_batch, J_batch.transpose(1, 2))
            I = torch.eye(6, device=self._device, dtype=th_batch.dtype) * (lambda_val**2)
            inv_JJT = torch.linalg.inv(JJT + I)
            J_pinv = torch.bmm(J_batch.transpose(1, 2), inv_JJT)
            delta_th = torch.bmm(J_pinv, error_vec.unsqueeze(-1)).squeeze(-1)
            th_batch += lr * delta_th
        return th_batch
    
    def ik_sup_objective(self, th):
        q_q_rest = th - self.th_rest
        return (q_q_rest @ self.w_rest @ q_q_rest.T) * 0.5

    def batch_ik_vel_12d(self, target_ee_vel, th, eps=0.1):
        """
        12D解析的ヤコビアンを用いた、バッチ対応の速度ベースIK。
        [変更点] linalg.inv() を linalg.solve() に置き換え、計算を最適化。
        
        :param target_ee_vel: 目標エンドエフェクタ速度(4x4の同次変換行列形式), shape: [batch, 4, 4]
        :param th: 現在の関節角度, shape: [batch, num_joints]
        :param eps: 補助タスクの重み
        :return: 計算された関節角速度, shape: [batch, num_joints]
        """
        batch_size = th.shape[0]

        # 1. 12D解析的ヤコビアンを計算
        J = self.batch_analytical_jacobian_12d(th) # shape: [B, 12, N_joints]

        # 2. 主タスクとNull空間射影のための行列を準備
        w_q_b = self.w_q.expand(batch_size, -1, -1)
        J_T = J.transpose(1, 2)
        
        # --- 主タスク用の中間行列 A = (J * Wq * J^T + lambda*I) ---
        Wq_JT = torch.bmm(w_q_b, J_T)
        J_Wq_JT = torch.bmm(J, Wq_JT)
        J_Wq_JT.add_(torch.eye(12, device=self._device, dtype=th.dtype) * 1e-6) # 特異点対策(Damping)

        # --- Null空間射影用の中間行列 B = (J * J^T + lambda*I) ---
        JJT = torch.bmm(J, J_T)
        JJT.add_(torch.eye(12, device=self._device, dtype=th.dtype) * 1e-4) # Damping
        
        # 3. 補助タスク（Rest姿勢）の勾配を計算
        # 解析的な勾配計算: g_del = W_rest * (th - th_rest)
        # (th - self.th_rest) の形状は [B, N] なので、行列積のために次元を追加 -> [B, N, 1]
        diff = (th - self.th_rest).unsqueeze(-1)
        # self.w_rest (形状 [N, N]) は自動的にブロードキャストされ、[B, N, N]として扱われる
        g_del = torch.matmul(self.w_rest, diff)

        # 4. 主タスク速度とNull空間射影を linalg.solve を使って計算
        target_vel_12d = self._batch_mat2vec_12d(target_ee_vel).unsqueeze(-1)
        # ステップ1: solve
        x = torch.linalg.solve(J_Wq_JT, target_vel_12d)
        # ステップ2: 行列積
        primary_task_vel = torch.bmm(Wq_JT, x)

        # ★★★ Null空間射影行列 P の計算 ★★★
        #   1. JJT @ X = J を X について解く (X = inv(JJT) @ J)
        #   2. J_hash_J = J_T @ X
        #   3. P = I - J_hash_J
        I = torch.eye(self.mlp.move_joint_num, device=self._device, dtype=th.dtype).expand(batch_size, -1, -1)
        # ステップ1: solve (第二引数が行列でも可)
        X = torch.linalg.solve(JJT, J)
        # ステップ2: 行列積
        J_hash_J = torch.bmm(J_T, X)
        # ステップ3: Null空間射影行列の計算
        P = I - J_hash_J
        
        # 5. 最終的な関節角速度を計算
        secondary_task_vel = torch.bmm(P, -eps * g_del)
        delta_th = primary_task_vel + secondary_task_vel
        
        return delta_th.squeeze(-1)
    
    def batch_ik_newton(self, target_pos_batch, initial_theta_batch, limits=None, max_iter=25, eps_stop=1e-4, eps_secondary=0.1):
        """
        12D解析的ヤコビアンを用いた、ニュートン法ベースのバッチ対応IK。
        [変更点] 全ての自動微分を解析的計算に置換。
                冗長な計算を削除。
                inv()をsolve()に統一し、autocastにも対応。
        """
        th_batch = initial_theta_batch.clone()
        batch_size = th_batch.shape[0]
        
        target_vec_12d = self._batch_mat2vec_12d(target_pos_batch)
        th_rest_b = self.th_rest.expand(batch_size, -1)
        
        ep_batch = torch.full((batch_size,), eps_secondary, device=self._device, dtype=th_batch.dtype)

        for _ in range(max_iter):
            current_pos, _ = self.batch_FK(th_batch)
            current_vec_12d = self._batch_mat2vec_12d(current_pos)
            error_vec = target_vec_12d - current_vec_12d
            
            if torch.max(torch.linalg.norm(error_vec[:, :3], dim=1)) < eps_stop:
                break
            
            J = self.batch_analytical_jacobian_12d(th_batch)
            J_T = J.transpose(1, 2)
            
            # --- 補助タスク（Rest姿勢）の勾配を解析的に計算 ---
            diff = (th_batch - th_rest_b).unsqueeze(-1)
            g_del = torch.matmul(self.w_rest, diff)

            # --- 行列HとPを linalg.solve を使って計算 ---
            # 1. 主タスク用の加重疑似逆行列Hのための準備
            J_Wq = torch.matmul(J, self.w_q) # self.w_qは対角行列なのでbmmでなくて良い
            J_Wq_JT = torch.bmm(J_Wq, J_T)
            J_Wq_JT.add_(torch.eye(12, device=self._device, dtype=th_batch.dtype) * 1e-6)

            # 2. Null空間射影Pのための準備
            JJT = torch.bmm(J, J_T)
            JJT.add_(torch.eye(12, device=self._device, dtype=th_batch.dtype) * 1e-4)

            # H @ error_vec を計算
            x_float_H = torch.linalg.solve(J_Wq_JT.float(), error_vec.unsqueeze(-1).float())
            delta_main = torch.bmm(J_Wq.transpose(1, 2), x_float_H.to(J_Wq.dtype))

            # P = I - J_T @ inv(JJT) @ J を計算
            I = torch.eye(self.mlp.move_joint_num, device=self._device, dtype=th_batch.dtype).expand(batch_size, -1, -1)
            X_float_P = torch.linalg.solve(JJT.float(), J.float())
            J_hash_J = torch.bmm(J_T, X_float_P.to(J_T.dtype))
            P = I - J_hash_J
            
            # --- 補助タスクの計算と更新 ---
            e_rest = torch.bmm(P, g_del) # ここでg_delを再利用
            ep_batch = ep_batch / (1.0 + torch.linalg.norm(e_rest, dim=1).squeeze(-1)**2)
            
            # --- 最終的な更新量の計算 ---
            delta = delta_main + torch.bmm(P, -ep_batch.view(-1, 1, 1) * g_del)
            th_batch = th_batch + delta.squeeze(-1)

            if limits is not None:
                # Clamp values to be within [min, max]
                th_batch = torch.max(th_batch, limits['min'])
                th_batch = torch.min(th_batch, limits['max'])
            
        return th_batch
    

    def _batch_rodrigues(self, axis_angle):
        """
        ロドリゲスの回転公式をバッチ処理で適用し、軸角度ベクトルから回転行列を生成します。
        
        :param axis_angle: 回転を表す軸角度ベクトル, shape: [batch, 3]
                           ベクトルの方向が回転軸、ベクトルの大きさが回転角[rad]に対応します。
        :return: 回転行列, shape: [batch, 3, 3]
        """
        batch_size = axis_angle.shape[0]
        device = axis_angle.device
        dtype = axis_angle.dtype

        # 回転角(ノルム)と回転軸(単位ベクトル)を計算
        angle = torch.linalg.norm(axis_angle, dim=1, keepdim=True)
        # ゼロ除算を避けるための小さな値
        small_angle_mask = angle.squeeze(-1) < 1e-6
        # 回転が非常に小さい場合は、回転軸をz軸などに設定し、angleを0として扱う
        axis = torch.where(small_angle_mask.unsqueeze(-1).expand_as(axis_angle), 
                           torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
                           axis_angle / angle)
        
        angle = angle.squeeze(-1)

        # 歪対称行列 K を作成
        K = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        I = torch.eye(3, device=device, dtype=dtype).expand(batch_size, -1, -1)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # cos(a) や (1-cos(a)) の計算時に次元を合わせる
        cos_a_exp = cos_a.view(-1, 1, 1)
        sin_a_exp = sin_a.view(-1, 1, 1)
        
        # ロドリゲスの公式
        R = I + sin_a_exp * K + (1 - cos_a_exp) * torch.bmm(K, K)
        
        return R

    def batch_impedance_control(self, target_pose_batch, force_torque_batch, stiffness_batch):
        """
        力覚センサーの値に基づき、目標姿勢を動的に補正するバッチ対応インピーダンス制御。
        アドミッタンス制御の形式で、力に応じて目標姿勢を「逃がす」ことで柔軟な動作を実現します。

        :param target_pose_batch: 元の目標姿勢 (同次変換行列), shape: [batch, 4, 4]
        :param force_torque_batch: センサーで計測された力・トルク [fx,fy,fz,tx,ty,tz], shape: [batch, 6]
        :param stiffness_batch: 目標の剛性 [K_px, K_py, K_pz, K_ox, K_oy, K_oz], shape: [batch, 6] or [6]
        :return: 力に応じて補正された新しい目標姿勢 (同次変換行列), shape: [batch, 4, 4]
        """
        batch_size = target_pose_batch.shape[0]
        device = target_pose_batch.device
        dtype = target_pose_batch.dtype

        if stiffness_batch.dim() == 1:
            stiffness_batch = stiffness_batch.expand(batch_size, -1)

        # 1. 剛性(Stiffness)からコンプライアンス(Compliance)を計算
        # ゼロ除算を防止するために微小値を加算
        compliance = 1.0 / (stiffness_batch + 1e-6)

        # 2. 力・トルクとコンプライアンスから、並進・回転の変位量を計算
        # F_ext = K * delta_x  =>  delta_x = K^-1 * F_ext = Compliance * F_ext
        delta_p = force_torque_batch[:, :3] * compliance[:, :3]
        delta_o_axis_angle = force_torque_batch[:, 3:] * compliance[:, 3:]
        
        # 3. 回転変位量(軸角度ベクトル)を回転行列に変換
        delta_R = self._batch_rodrigues(delta_o_axis_angle)

        # 4. 変位量を同次変換行列にまとめる
        delta_H = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
        delta_H[:, :3, :3] = delta_R
        delta_H[:, :3, 3] = delta_p
        
        # 5. 元の目標姿勢に変位を適用し、補正後の目標姿勢を計算
        # H_modified = H_target * delta_H
        modified_target_pose_batch = torch.bmm(target_pose_batch, delta_H)

        return modified_target_pose_batch
    

    def batch_impedance_control_force_only(self, target_pose_batch, force_batch, stiffness_batch):
        """
        【力センサーのみ版】
        力覚センサーの値に基づき、目標姿勢の「位置」を動的に補正するバッチ対応インピーダンス制御。
        トルクセンサーがない場合を想定し、並進力のみを使用して目標位置を「逃がす」ことで柔軟な動作を実現します。
        目標の「向き」は変更されません。

        :param target_pose_batch: 元の目標姿勢 (同次変換行列), shape: [batch, 4, 4]
        :param force_batch: センサーで計測された力 [fx,fy,fz], shape: [batch, 3]
        :param stiffness_batch: 目標の剛性 [K_px, K_py, K_pz], shape: [batch, 3] or [3]
        :return: 力に応じて位置が補正された新しい目標姿勢 (同次変換行列), shape: [batch, 4, 4]
        """
        batch_size = target_pose_batch.shape[0]
        device = target_pose_batch.device
        dtype = target_pose_batch.dtype

        # stiffness_batchがバッチ対応でない場合、次元を拡張
        if stiffness_batch.dim() == 1:
            stiffness_batch = stiffness_batch.expand(batch_size, -1)
            
        # 1. 剛性(Stiffness)からコンプライアンス(Compliance)を計算
        # ゼロ除算を防止するために微小値を加算
        compliance = 1.0 / (stiffness_batch + 1e-6)

        # 2. 力とコンプライアンスから、並進の変位量のみを計算
        delta_p = force_batch * compliance
        
        # 3. 並進変位のみを含む同次変換行列を作成 (回転はなし)
        # 単位行列で初期化
        delta_H = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
        # 回転部分(3x3)は単位行列のままにし、位置部分(3x1)に変位を設定
        delta_H[:, :3, 3] = delta_p
        
        # 4. 元の目標姿勢に変位を適用し、補正後の目標姿勢を計算
        # H_modified = H_target * delta_H
        modified_target_pose_batch = torch.bmm(target_pose_batch, delta_H)

        return modified_target_pose_batch