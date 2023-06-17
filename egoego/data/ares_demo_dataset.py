import numpy as np
import os
import random
import joblib 
import pickle 
import cv2 
import time 

from scipy.ndimage.interpolation import rotate

import torch
from torch.utils.data import Dataset
import pytorch3d.transforms as transforms 

import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.core import lie_algebra

class ARESDemoDataset(Dataset):
    def __init__(
        self,
        data_root_folder,
    ):

        self.data_root_folder = data_root_folder 
        
        self.head_data_path = os.path.join(data_root_folder, "demo_ares_data.p")
        self.window_data_dict = joblib.load(self.head_data_path)
          
        # Add slam result to data loader 
        slam_res_folder = os.path.join(data_root_folder, "droid_slam_res")  
        for tmp_k in self.window_data_dict:
            scene_name = self.window_data_dict[tmp_k]['seq_name'].split("-")[0]
            npy_name = "-".join(self.window_data_dict[tmp_k]['seq_name'].split("-")[1:])
            slam_seq_npy = os.path.join(slam_res_folder, scene_name, npy_name+".npy")

            if os.path.exists(slam_seq_npy):
                aligned_slam_trans, aligned_slam_rot_mat, aligned_slam_quat_wxyz = \
                self.load_slam_res_and_align_first(slam_seq_npy, self.window_data_dict[tmp_k]['head_qpos'])

                ori_slam_trans, ori_slam_rot_mat, ori_slam_quat_wxyz = \
                self.load_data_from_droidslam(slam_seq_npy)

                self.window_data_dict[tmp_k]['aligned_slam_trans'] = aligned_slam_trans
                self.window_data_dict[tmp_k]['aligned_slam_rot_quat'] = aligned_slam_quat_wxyz 
                self.window_data_dict[tmp_k]['aligned_slam_rot_mat'] = aligned_slam_rot_mat

                self.window_data_dict[tmp_k]['ori_slam_trans'] = ori_slam_trans
                self.window_data_dict[tmp_k]['ori_slam_rot_quat'] = ori_slam_quat_wxyz 
                self.window_data_dict[tmp_k]['ori_slam_rot_mat'] = ori_slam_rot_mat

        print("Total number of sequences:{0}".format(len(self.window_data_dict)))
    
    def load_data_from_droidslam(self, data_path):
        rot_data = np.load(data_path) # T X 7
        trans = rot_data[:, :3] # T X 3 
        rot_quat_wxyz = rot_data[:, 3:] # T X 4 (w, x, y, z)
        
        rot_quat_wxyz = torch.from_numpy(rot_quat_wxyz).float() 

        rot_mat = transforms.quaternion_to_matrix(rot_quat_wxyz)
    
        return trans, rot_mat.data.cpu().numpy(), rot_quat_wxyz.data.cpu().numpy()

    def load_slam_res_and_align_first(self, data_path, gt_head_pose):
        slam_trans, slam_rot_mat, slam_quat_wxyz = self.load_data_from_droidslam(data_path) # T' X 3, T' X 3 X 3  
      
        gt_head_trans = gt_head_pose[:, :3] # T X 3 
        gt_head_quat_wxyz = gt_head_pose[:, 3:] # T X 4
        gt_head_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(gt_head_quat_wxyz).float()).data.cpu().numpy()

        pred2gt_rot = np.matmul(gt_head_rot_mat[0], slam_rot_mat[0].T) # 3 X 3 
        # print("pred2gt_rot:{0}".format(pred2gt_rot))
        seq_pred_rot_mat = torch.from_numpy(slam_rot_mat).float() # T X 3 X 3 
        pred2gt_rot_seq = torch.from_numpy(pred2gt_rot).float()[None, :, :] # 1 X 3 X 3 
        aligned_seq_pred_root_mat = torch.matmul(pred2gt_rot_seq, seq_pred_rot_mat) # T X 3 X 3 
        aligned_seq_pred_root_quat_wxyz = transforms.matrix_to_quaternion(aligned_seq_pred_root_mat)

        aligned_seq_pred_root_mat = aligned_seq_pred_root_mat.data.cpu().numpy()
        aligned_seq_pred_root_quat_wxyz = aligned_seq_pred_root_quat_wxyz.data.cpu().numpy()

        seq_pred_trans = torch.from_numpy(slam_trans).float()[:, :, None] # T X 3 X 1 
        aligned_seq_pred_trans = torch.matmul(pred2gt_rot_seq, seq_pred_trans)[:, :, 0] # T X 3 
        aligned_seq_pred_trans = aligned_seq_pred_trans.data.cpu().numpy() 

        # Make initial x,y,z aligned
        move_to_gt_trans = gt_head_trans[0:1, :] - aligned_seq_pred_trans[0:1, :]
        aligned_seq_pred_trans = aligned_seq_pred_trans + move_to_gt_trans 

        return aligned_seq_pred_trans, aligned_seq_pred_root_mat, aligned_seq_pred_root_quat_wxyz

    def load_of_feats(self, of_files):
        ofs = []
        for of_file in of_files:
            curr_of_file = of_file.replace("/viscam/u/jiamanli/datasets/egomotion_syn_dataset/habitat_rendering_replica_all", \
            self.data_root_folder)
            of_i = np.load(curr_of_file.replace("raft_flows", "raft_of_feats"))
            ofs.append(of_i)
        ofs = np.stack(ofs) # T X D 
        
        return ofs

    def __len__(self):
        return len(self.window_data_dict)

    def __getitem__(self, index):
        seq_name = self.window_data_dict[index]['seq_name']
        seq_head_pose = self.window_data_dict[index]['head_qpos'] # T X 7 
        seq_head_vels = self.window_data_dict[index]['head_vels'][:-1] # (T-1) X 6, the last frame's head velocity cannot calculated, remove the cloned last head vel.   
        seq_of_files = self.window_data_dict[index]['of_files'] # (T-1)

        seq_len = seq_head_vels.shape[0]

        random_t_idx = 0
        end_t_idx = seq_len        

        window_head_pose = seq_head_pose[random_t_idx:end_t_idx+1]
        window_head_vels = seq_head_vels[random_t_idx:end_t_idx]
        window_of_files = seq_of_files[random_t_idx:end_t_idx]

        # Read optical flow data 
        window_of_data = self.load_of_feats(window_of_files) 

        actual_seq_len = window_head_vels.shape[0]

        query = {}
        query["head_pose"] = window_head_pose # (T'+1) X 7
        query['head_vels'] = window_head_vels # T' X 6 
        query['of'] = window_of_data # T' X 224 X 224 X 2 /T' X D
       
        query['seq_name'] = seq_name 

        query['seq_len'] = actual_seq_len
        
        if "aligned_slam_trans" in self.window_data_dict[index]:
            aligned_slam_trans = self.window_data_dict[index]['aligned_slam_trans'][random_t_idx:end_t_idx+1]
            aligned_slam_rot_quat = self.window_data_dict[index]['aligned_slam_rot_quat'][random_t_idx:end_t_idx+1]
            aligned_slam_rot_mat = self.window_data_dict[index]['aligned_slam_rot_mat'][random_t_idx:end_t_idx+1]

            query['aligned_slam_trans'] = aligned_slam_trans 
            query['aligned_slam_rot_quat'] = aligned_slam_rot_quat
            query['aligned_slam_rot_mat'] = aligned_slam_rot_mat

            query['ori_slam_trans'] = self.window_data_dict[index]['ori_slam_trans'][random_t_idx:end_t_idx+1]
            query['ori_slam_rot_quat'] = self.window_data_dict[index]['ori_slam_rot_quat'][random_t_idx:end_t_idx+1]
            query['ori_slam_rot_mat'] = self.window_data_dict[index]['ori_slam_rot_mat'][random_t_idx:end_t_idx+1]

        return query
