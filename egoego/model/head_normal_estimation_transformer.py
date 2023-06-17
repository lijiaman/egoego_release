import time 

import numpy as np 

from collections import defaultdict

import torch 
from torch import nn
import pytorch3d.transforms as transforms 

from egoego.model.mlp import MLP

from egoego.model.transformer_module import Decoder 

from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync

def get_heading_q_batch(_q):
    q = _q.clone()
    q[:, 1] = 0.0
    q[:, 2] = 0.0
    
    q_norm = torch.norm(q, dim = 1,  p=2).view(-1, 1)

    return q/q_norm

def de_heading_batch(q):
    q_deheaded = get_heading_q_batch(q)
    q_deheaded_inv = transforms.quaternion_invert(q_deheaded)

    return q_deheaded_inv, transforms.quaternion_multiply(q_deheaded_inv, q)

def position_loss(gt_pos, pred_pos):
    # gt_pos/pred_pos: (BS*T) X 3 
    return (gt_pos - pred_pos).abs().sum(dim=1)

def orientation_loss(gt_quat, pred_quat):
    # gt_quat/pred_quat: (BS*T) X 4
    pred_quat_invert = transforms.quaternion_invert(pred_quat) # (BS*T) X 4 
    dist = transforms.quaternion_multiply(gt_quat, pred_quat_invert) 
    """make the diff quat to be identity"""
    quat_iden = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gt_quat.dtype, device=gt_quat.device).repeat(gt_quat.size()[0], 1)
    loss = torch.abs(dist) - quat_iden

    return loss.pow(2).sum(dim=1)

def rotation_matrix_from_two_vectors(vec1, vec2):
    # Find the rotation matrix that aligns vec1 to vec2 
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1-c) / (s**2))

    return rotation_matrix 

def cal_rotation_from_floor_normal(pred_floor_normal):
    ori_floor_normal = np.asarray([0, 0, 1]) 
    rel_rot = rotation_matrix_from_two_vectors(pred_floor_normal, ori_floor_normal)
    return rel_rot 

class HeadNormalFormer(nn.Module):

    def __init__(self, opt, device, eval_whole_pipeline=False):
        super(HeadNormalFormer, self).__init__()
       
        self.opt = opt 

        self.device = device 

        self.htype = htype = "relu"
        self.mlp_hsize = [512, 256]
        self.cnn_fdim = 512

        if eval_whole_pipeline:
            self.action_n_dec_layers = opt.normal_n_dec_layers
            self.action_n_head = opt.normal_n_head
            self.action_d_k = opt.normal_d_k
            self.action_d_v = opt.normal_d_v 
            self.action_d_feats = self.cnn_fdim # Optical flow feature vector
            self.action_d_model = opt.normal_d_model
            self.action_max_timesteps = opt.normal_window
        else:
            # New version 
            self.action_n_dec_layers = opt.n_dec_layers
            self.action_n_head = opt.n_head
            self.action_d_k = opt.d_k
            self.action_d_v = opt.d_v 
            self.action_d_feats = self.cnn_fdim # Optical flow feature vector
            self.action_d_model = opt.d_model
            self.action_max_timesteps = opt.window

        self.transformer_window_size = self.action_max_timesteps # Used only in testing mode. 

        # Introduce transformer for capturing sequential information
        # Input: BS X D X T 
        # Output: BS X T X D'
        self.action_transformer = Decoder(d_feats=6+3+6+3, d_model=self.action_d_model, \
            n_layers=self.action_n_dec_layers, n_head=self.action_n_head, \
            d_k=self.action_d_k, d_v=self.action_d_v, \
            max_timesteps=self.action_max_timesteps, \
            use_full_attention=True) 
    
        self.action_normal_mlp = MLP(self.action_d_model, self.mlp_hsize, htype)
        self.action_normal_fc = nn.Linear(self.mlp_hsize[-1], 3)
    
    def prep_padding_mask(self, seq_len):
        # Generate padding mask 
        actual_seq_len = seq_len # BS
        tmp_mask = torch.arange(self.transformer_window_size).expand(seq_len.shape[0], \
        self.transformer_window_size).to(seq_len.device) < actual_seq_len[:, None].repeat(1, self.transformer_window_size)
        # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :]

        return padding_mask

    def forward(self, data):
        slam_rot_mat = data['head_rot_mat'].to(self.device).float() # BS X (T+1) X 3 X 3  
        slam_trans = data['head_trans'].to(self.device).float() # BS X (T+1) X 3 
        seq_len = data['seq_len'].to(self.device).float()

        if slam_trans.shape[1] > self.transformer_window_size:
            slam_rot_mat = slam_rot_mat[:, :self.transformer_window_size+1]
            slam_trans = slam_trans[:, :self.transformer_window_size+1]
            seq_len = torch.tensor(self.transformer_window_size + 1).float()[None].to(self.device)

        # Convert SLAM representation 
        slam_rot_6d = transforms.matrix_to_rotation_6d(slam_rot_mat) # BS X (T+1) X 6 

        slam_rot_mat_diff = torch.matmul(slam_rot_mat[:, 1:, :, :], slam_rot_mat[:, :-1, :, :].transpose(2, 3)) # BS X T X 3 X 3 
        slam_trans_diff = slam_trans[:, 1:, :] - slam_trans[:, :-1, :] # BS X T X 3 

        slam_rot_diff_6d = transforms.matrix_to_rotation_6d(slam_rot_mat_diff) # BS X T X 6 

        decoder_input = torch.cat((slam_rot_6d[:, :-1, :], slam_trans[:, :-1, :], \
                    slam_rot_diff_6d, slam_trans_diff), dim=-1) # BS X T X (6+3+6+3)
        decoder_input = decoder_input.transpose(1, 2) # BS X (6+3+6+3) X T 

        if seq_len.shape[0] == 1 and seq_len[0] - 1 < self.transformer_window_size:
            padding_input = torch.zeros(decoder_input.shape[0], \
            decoder_input.shape[1], self.transformer_window_size-int(seq_len[0].cpu().item())+1).to(decoder_input.device)

            decoder_input = torch.cat((decoder_input, padding_input), dim=-1) 

        assert decoder_input.shape[-1] == self.transformer_window_size

        padding_mask = self.prep_padding_mask(seq_len-1) # BS 

        # Get position vec for position-wise embedding
        bs = decoder_input.shape[0]
        num_steps = decoder_input.shape[2]
        pos_vec = torch.arange(num_steps)+1 # timesteps
        pos_vec = pos_vec[None, None, :].to(self.device).repeat(bs, 1, 1) # BS X 1 X timesteps

        seq_rnn_out, _ = self.action_transformer(decoder_input, padding_mask, pos_vec) # B X T X D

        # Predict angular velocity to integrate to rotation. 
        seq_normal_out = self.action_normal_mlp(seq_rnn_out[:, 0, :])
       
        seq_normal = self.action_normal_fc(seq_normal_out) # BS X 3

        feature_pred = defaultdict(list)  

        feature_pred['pred_normal'] = seq_normal # BS X 3 

        return feature_pred

    def align_xy_plane_traj(self, traj_est, traj_ref):
        # traj_est: T X 7 
        # traj_ref: T X 7 
        traj_est = traj_est.copy()
        traj_ref  = traj_ref.copy()

        traj_est[:, 2] = 1
        traj_ref[:, 2] = 1

        num_timesteps = traj_est.shape[0] 
                    
        tstamps = []

        sec_interval = 1./30 
        curr_timestamp = time.time() 
        for idx in range(num_timesteps):
            curr_timestamp += sec_interval 
            curr_line = str(int(curr_timestamp*1e9))
            tstamps.append(float(curr_line)) 
        
        traj_est = PoseTrajectory3D(
            positions_xyz=traj_est[:,:3],
            orientations_quat_wxyz=traj_est[:,3:],
            timestamps=np.array(tstamps))

        traj_ref = PoseTrajectory3D(
            positions_xyz=traj_ref[:,:3],
            orientations_quat_wxyz=traj_ref[:,3:],
            timestamps=np.array(tstamps))

        # Calculate APE 
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

        correct_scale = True
        align = True 
                   
        only_scale = correct_scale and not align
      
        # try:
        tmp_r, tmp_t, tmp_s = traj_est.align(traj_ref, correct_scale, only_scale, n=-1)

        # print("Align xy plane scale:{0}".format(tmp_s))
        
        return tmp_r, traj_est._positions_xyz, traj_ref._positions_xyz # 3 X 3, T X 3, T X 3  

    def forward_for_eval(self, data, pred_scale=None, use_gt_aligned_rot=False):
        self.action_transformer.eval()

        feature_pred = self.forward(data)

        pred_normal = feature_pred['pred_normal'] # BS X 3

        assert pred_normal.shape[0] == 1 # bs = 1 
        pred_aligned_rot_mat = cal_rotation_from_floor_normal(pred_normal[0].data.cpu().numpy()) # 3 X 3 

        pred_aligned_rot_mat = torch.from_numpy(pred_aligned_rot_mat).to(self.device).float() 

        if use_gt_aligned_rot:
            pred_aligned_rot_mat = data['aligned_rot_mat'].to(self.device).float()[0] # 3 X 3 

        ori_slam_rot_mat = data['head_rot_mat'].to(self.device).float() # BS X (T+1) X 3 X 3  
        ori_slam_trans = data['head_trans'].to(self.device).float() # BS X (T+1) X 3 

        if pred_scale is not None:
            gt_scale = pred_scale[None] # BS
        else:
            gt_scale = data['aligned_scale'].to(self.device).float()# BS 
        
        # Apply rotation and scale to translation 
        ori_slam_trans_diff = ori_slam_trans[:, 1:, :] - ori_slam_trans[:, :-1, :] # BS X T X 3 

        trans_diff_after_rot_scale = torch.matmul(pred_aligned_rot_mat[None, None, \
                                :, :].repeat(ori_slam_trans_diff.shape[0], ori_slam_trans_diff.shape[1], 1, 1), \
                                ori_slam_trans_diff.detach()[:, :, :, None]).squeeze(-1) * gt_scale[:, None, None] # BS X (T-1) X 3 
        
        trans_after_rot_scale = [ori_slam_trans[:, 0]] # BS X 3 
        for t_idx in range(ori_slam_trans.shape[1]-1):
            trans_after_rot_scale.append(trans_after_rot_scale[-1]+trans_diff_after_rot_scale[:, t_idx])

        trans_after_rot_scale = torch.stack(trans_after_rot_scale, dim=1) # BS X T X 3 

        # Apply rotation to slam rotation 
        aligned_slam_rot_mat = torch.matmul(pred_aligned_rot_mat[None, None, \
                                :, :].repeat(ori_slam_rot_mat.shape[0], ori_slam_rot_mat.shape[1], 1, 1), \
                                ori_slam_rot_mat) # BX X T X 3 X 3
        aligned_slam_quat = transforms.matrix_to_quaternion(aligned_slam_rot_mat) # BS X T X 4 

        feature_pred_for_eval = defaultdict(list)  

        # Calculate a rotation to align the trajectory projected in xy plane. (remove the effects of orientation)
        traj_est = torch.cat((trans_after_rot_scale, aligned_slam_quat), dim=-1) # BS X T X 7 
        traj_ref = data['ori_head_pose'] # BS X T X 7
        traj_est = traj_est[0].data.cpu().numpy() # T X 7
        traj_ref = traj_ref[0].data.cpu().numpy() # T X 7
 
        if traj_est.shape[0] > traj_ref.shape[0]:
            traj_est = traj_est[:traj_ref.shape[0]]
        rot_mat_align_xy_plane, tmp_debug_traj_est, tmp_debug_traj_ref = self.align_xy_plane_traj(traj_est, traj_ref) # 3 X 3 
        rot_mat_align_xy_plane = torch.from_numpy(rot_mat_align_xy_plane).to(self.device).float() # 3 X 3 

        ori_gt_quat = data['ori_head_pose'][:, :, 3:].to(self.device).float() # BS X T X 4 
        ori_gt_trans = data['ori_head_pose'][:, :, :3].to(self.device).float() # BS X T X 3 
        ori_gt_rot_mat = transforms.quaternion_to_matrix(ori_gt_quat) # BS X T X 3 X 3 
        de_headed_gt_rot_mat = ori_gt_rot_mat 
        de_headed_gt_trans = ori_gt_trans 

        de_headed_slam_rot_mat = torch.matmul(rot_mat_align_xy_plane[None, None, :, :].repeat(aligned_slam_rot_mat.shape[0], \
                            aligned_slam_rot_mat.shape[1], 1, 1), aligned_slam_rot_mat.float())
        de_headed_slam_trans = trans_after_rot_scale - trans_after_rot_scale[:, 0:1, :]
        de_headed_slam_trans = torch.matmul(rot_mat_align_xy_plane[None, None, :, :].repeat(aligned_slam_rot_mat.shape[0], \
                            aligned_slam_rot_mat.shape[1], 1, 1), de_headed_slam_trans[:, :, :, None].float()).squeeze(-1)
        de_headed_slam_trans = de_headed_slam_trans + ori_gt_trans[:, 0:1, :]

        feature_pred_for_eval['head_trans'] = de_headed_slam_trans 
        feature_pred_for_eval['head_rot_mat'] = de_headed_slam_rot_mat 
        de_headed_slam_quat = transforms.matrix_to_quaternion(de_headed_slam_rot_mat)
        feature_pred_for_eval['head_pose'] = torch.cat((de_headed_slam_trans, de_headed_slam_quat), dim=-1) # BS X T X 7

        feature_pred_for_eval['gt_head_trans'] = de_headed_gt_trans 
        feature_pred_for_eval['gt_head_rot_mat'] = de_headed_gt_rot_mat
        de_headed_gt_quat = transforms.matrix_to_quaternion(de_headed_gt_rot_mat)
        feature_pred_for_eval['gt_head_pose'] = torch.cat((de_headed_gt_trans, de_headed_gt_quat), dim=-1)

        self.action_transformer.train() 

        return feature_pred_for_eval

    def forward_for_eval_upper_bound(self, data):

        gt_head_trans = data['ori_head_pose'][:, :, :3] # BS X T X 3
        
        pred_aligned_rot_mat = data['aligned_rot_mat'].to(self.device).float()[0] # 3 X 3 

        ori_slam_rot_mat = data['head_rot_mat'].to(self.device).float() # BS X (T+1) X 3 X 3  
        ori_slam_trans = data['head_trans'].to(self.device).float() # BS X (T+1) X 3 
   
        gt_scale = data['aligned_scale'].to(self.device).float()# BS  
        
        # Apply rotation and scale to translation 
        ori_slam_trans_diff = ori_slam_trans[:, 1:, :] - ori_slam_trans[:, :-1, :] # BS X T X 3 

        trans_diff_after_rot_scale = torch.matmul(pred_aligned_rot_mat[None, None, \
                                :, :].repeat(ori_slam_trans_diff.shape[0], ori_slam_trans_diff.shape[1], 1, 1), \
                                ori_slam_trans_diff.detach()[:, :, :, None]).squeeze(-1) * gt_scale[:, None, None] # BS X (T-1) X 3 
        
        trans_after_rot_scale = [gt_head_trans[:, 0]] # BS X 3 
        for t_idx in range(ori_slam_trans.shape[1]-1):
            trans_after_rot_scale.append(trans_after_rot_scale[-1]+trans_diff_after_rot_scale[:, t_idx])

        trans_after_rot_scale = torch.stack(trans_after_rot_scale, dim=1) # BS X T X 3 

        # Apply rotation to slam rotation 
        aligned_slam_rot_mat = torch.matmul(pred_aligned_rot_mat[None, None, \
                                :, :].repeat(ori_slam_rot_mat.shape[0], ori_slam_rot_mat.shape[1], 1, 1), \
                                ori_slam_rot_mat) # BX X T X 3 X 3
        aligned_slam_quat = transforms.matrix_to_quaternion(aligned_slam_rot_mat) # BS X T X 4 

        feature_pred_for_eval = defaultdict(list)  

        feature_pred_for_eval['head_trans'] = trans_after_rot_scale
        feature_pred_for_eval['head_rot_mat'] =  aligned_slam_rot_mat
        feature_pred_for_eval['head_pose'] = torch.cat((trans_after_rot_scale, aligned_slam_quat), dim=-1) # BS X T X 7

        return feature_pred_for_eval

    def compute_loss(self, feature_pred, data):
        pred_floor_normal = feature_pred['pred_normal'] # B X 3
        gt_floor_normal = data['floor_normal'].to(self.device).squeeze(-1)  

        normal_loss = position_loss(gt_floor_normal, pred_floor_normal).mean() 

        loss = normal_loss

        return loss, normal_loss
