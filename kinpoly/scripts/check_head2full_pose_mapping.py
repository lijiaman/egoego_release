import sys 
sys.path.append("../")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import sys
import pickle
import time
import math
import torch
import numpy as np
sys.path.append(os.getcwd())

import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from relive.utils import *
from relive.models.mlp import MLP
from relive.models.head_mapping_transformer import HeadMappingTransformer 
from relive.data_loaders.statear_smpl_dataset import StateARDataset
from relive.utils.torch_ext import get_scheduler
from relive.utils.statear_smpl_config import Config
from relive.utils.torch_utils import rotation_from_quaternion
from relive.utils.metrics import *

from utils_common import vis_single_head_pose_traj, vis_multiple_head_pose_traj
from utils_common import show3Dpose_animation

def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50
    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat

def remove_init_heading_for_seq(seq_pose, seq_full_pose):
    # seq_pose: T X 7 
    # seq_full_pose: T X 24 X 3 
    init_heading_q = get_heading_q(seq_pose[0, 3:]) # 4
    rm_heading_q = quaternion_inverse(init_heading_q) # 4

    # Apply heading remove quaternion to all the remaining poses 
    seq_rot_q_list = []
    timesteps = seq_pose.shape[0]
    for t_idx in range(timesteps):
        curr_q = quaternion_multiply(rm_heading_q, seq_pose[t_idx, 3:]) # 4 
        seq_rot_q_list.append(curr_q) 

    seq_rot_q = np.asarray(seq_rot_q_list) # T X 4 

    # Apply heading remove quaternion to head translation in all the timesteps 
    seq_trans_list = []
    for t_idx in range(timesteps):
        curr_trans = quat_mul_vec(rm_heading_q, seq_pose[t_idx, :3])
        seq_trans_list.append(curr_trans)

    seq_trans = np.asarray(seq_trans_list) # T X 3 

    # Apply heading remove quaternion to body joint positions in all the timesteps 
    seq_body_pose_list = []
    num_joints = seq_full_pose.shape[1] 
    for t_idx in range(timesteps):
        curr_t_pos_list = []
        for j_idx in range(num_joints):
            curr_joint_pos = quat_mul_vec(rm_heading_q, seq_full_pose[t_idx, j_idx, :])
            curr_t_pos_list.append(curr_joint_pos)
        curr_t_pos = np.asarray(curr_t_pos_list) # 24 X 3 
        
        seq_body_pose_list.append(curr_t_pos) 
    seq_body_pose = np.asarray(seq_body_pose_list) # T X 24 X 3 

    # Put the first head pose translation to (0, 0, 0)
    delta_trans = seq_trans[0:1, :].copy()
    seq_trans -= delta_trans 
    seq_body_pose -= delta_trans[np.newaxis, :, :] 

    processed_seq_pose = np.concatenate((seq_trans, seq_rot_q), axis=1) # T X 7 

    return processed_seq_pose, seq_body_pose  

def gen_head_loss(val_head_pose, train_head_pose, val_full_pose, train_full_pose):
    # val_head_pose: T X 7 
    # train_head_pose: T X 7 
    # val_full_pose: T X 24 X 3 
    # train_full_pose: T X 24 X 3 

    # Need to remove heading for the initial pose, and align the translation to (0, 0, 0) 
    val_head_pose, val_full_pose = remove_init_heading_for_seq(val_head_pose, val_full_pose)
    train_head_pose, train_full_pose = remove_init_heading_for_seq(train_head_pose, train_full_pose)

    val_head_mat = get_root_matrix(val_head_pose)
    train_head_mat = get_root_matrix(train_head_pose)
    head_dist = get_frobenious_norm(val_head_mat, train_head_mat)
 
    val_head_trans = val_head_pose[:, :3] # T X 3
    train_head_trans = train_head_pose[:, :3] # T X 3 
    head_trans_dist = np.linalg.norm(val_head_trans - train_head_trans, axis=1).mean() * 1000

    return head_dist, head_trans_dist, val_head_pose, train_head_pose, val_full_pose, train_full_pose  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--data', default=None)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--action', type=str, default='all')
    parser.add_argument('--perspective', type=str, default='first')
    parser.add_argument('--wild', action='store_true', default=False)
    args = parser.parse_args()

    if args.data is None:
        args.data = args.mode if args.mode in {'train', 'test'} else 'train'

    cfg = Config(args.action, args.cfg, wild = args.wild, \
        create_dirs=(args.iter == 0), mujoco_path = "assets/mujoco_models/%s.xml")
    
    """setup"""
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda', index=args.gpu_index) if (torch.cuda.is_available()) else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    tb_logger = Logger(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """Datasets"""
    train_dataset = StateARDataset(cfg, "train")
   
    # For validation
    val_dataset = StateARDataset(cfg, "test")

    fr_num_start = 80
    fr_num_end = 150

    for i_epoch in range(args.iter, cfg.num_epoch):
        fr_num = fr_num_start + i_epoch/cfg.num_epoch * (fr_num_end - fr_num_start) // 5 * 5
        print(f"fr_num: {fr_num}, cfg: {args.cfg}")
        
        t0 = time.time()
        train_generator = train_dataset.sampling_generator_for_val(num_samples=cfg.num_sample, \
            batch_size=1, num_workers=0, fr_num=fr_num)
        val_generator = val_dataset.sampling_generator_for_val(num_samples=cfg.num_sample, \
            batch_size=1, num_workers=0, fr_num=fr_num)
        
        for it, val_data_dict in enumerate(val_generator):
        
            val_data_dict = {k:v.clone().to(device).type(dtype) for k, v in val_data_dict.items()}
            val_head_pose = val_data_dict['head_pose']
            val_full_pose = val_data_dict['wbpos'] 

            dist_list = []
            dist_head_pose_list = [] # Store the head pose from training data 
            dist_full_pose_list = [] # Store the full body pose from training data 
            for train_it, train_data_dict in enumerate(train_generator):
                train_data_dict = {k:v.clone().to(device).type(dtype) for k, v in train_data_dict.items()}
                train_head_pose = train_data_dict['head_pose'] 
                train_full_pose = train_data_dict['wbpos'] # B(1) X T X 72 

                # Calculate head pose distance 
                curr_head_dist, curr_head_trans_dist, aligned_val_head_pose, aligned_train_head_pose, aligned_val_full_pose, aligned_train_full_pose = \
                    gen_head_loss(val_head_pose[0].data.cpu().numpy(), train_head_pose[0].data.cpu().numpy(), \
                        val_full_pose[0].data.cpu().numpy().reshape(-1, 24, 3), train_full_pose[0].data.cpu().numpy().reshape(-1, 24, 3))

                head_trans_threshold = 100
                if curr_head_trans_dist < head_trans_threshold:
                    dist_head_pose_list.append(aligned_train_head_pose)
                    dist_full_pose_list.append(aligned_train_full_pose)
                    dist_list.append(curr_head_dist) 

            sorted_idx = np.asarray(dist_list).argsort() # From small to large value 
            topk = 5
            selected_idx = sorted_idx[:topk]
            for tmp_idx in range(len(selected_idx)):
                s_idx = selected_idx[tmp_idx]

                curr_head_pose = dist_head_pose_list[s_idx]
                curr_full_pose = dist_full_pose_list[s_idx]
                
                # Visualize validation head pose and full pose, also visualize the full/head pose in training data which has the smallest error with validation pose. 
                dest_vis_folder = os.path.join(cfg.log_dir, "head_traj_check_vis", str(i_epoch)+"_iter_"+str(it)+"_b_"+str(tmp_idx))
                if not os.path.exists(dest_vis_folder):
                    os.makedirs(dest_vis_folder)
                anim_gt_head_seq_path = os.path.join(dest_vis_folder, "validation_head_traj.gif")
                anim_pred_head_seq_path = os.path.join(dest_vis_folder, "train_head_traj.gif")
                anim_cmp_seq_path = os.path.join(dest_vis_folder, "cmp_head_traj.gif")

                head_val_trans = torch.from_numpy(aligned_val_head_pose[:, :3]).float().to(device) # T X 3
                head_val_quat = torch.from_numpy(aligned_val_head_pose[:, 3:]).float().to(device) # T X 4
                head_train_trans = torch.from_numpy(curr_head_pose[:, :3]).float().to(device) # T X 3
                head_train_quat = torch.from_numpy(curr_head_pose[:, 3:]).float().to(device) # T X 4 

                head_val_rot_mat = quat2mat(head_val_quat)
                head_train_rot_mat = quat2mat(head_train_quat)

                if not os.path.exists(anim_gt_head_seq_path):
                    vis_single_head_pose_traj(head_val_trans, head_val_rot_mat, anim_gt_head_seq_path)
                    vis_single_head_pose_traj(head_train_trans, head_train_rot_mat, anim_pred_head_seq_path)
                    vis_multiple_head_pose_traj([head_val_trans, head_train_trans], \
                        [head_val_rot_mat, head_train_rot_mat], anim_cmp_seq_path)

                # Visualize the full body pose 
                jpos_val = aligned_val_full_pose[np.newaxis] # 1 X T X 24 X 3 
                jpos_train = curr_full_pose[np.newaxis] # 1 X T X 24 X 3
              
                vis_joints_data = np.concatenate((jpos_val, jpos_train), axis=0) # 2 X T X 24 X 3 
               
                dest_full_pose_path = os.path.join(dest_vis_folder, "full_pose_cmp.gif")
                if not os.path.exists(dest_full_pose_path):
                    show3Dpose_animation(vis_joints_data, dest_full_pose_path, use_mujoco=True) 
