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

class ARESHeadPoseDataset(Dataset):
    def __init__(
        self,
        data_root_folder,
        train: bool,
        window: int = 120,
        for_eval = False,
    ):
        self.train = train
        
        self.window = window

        if self.train:
            self.augment = True 
        else:
            self.augment = False 

        self.data_root_folder = data_root_folder 
        
        if self.train:
            self.head_data_path = os.path.join(data_root_folder, "ares_egoego_processed/train_ares_smplh_motion.p")
        else:
            self.head_data_path = os.path.join(data_root_folder, "ares_egoego_processed/test_ares_smplh_motion.p")
        
        self.window_head_data_path = self.head_data_path.replace(".p", "_window_"+str(self.window)+".p")
        if os.path.exists(self.window_head_data_path):
            self.window_data_dict = joblib.load(self.window_head_data_path)
        else:
            ori_data_dict = joblib.load(self.head_data_path)
            self.window_data_dict = self.filter_data(ori_data_dict)

        # Add slam result to data loader 
        slam_res_folder = os.path.join(data_root_folder, "ares/droid_slam_res") # 2 is better than 1 and each calib 
        missing_slam_cnt = 0
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
            else:
                missing_slam_cnt += 1

        print("Total number of sequences:{0}".format(len(self.window_data_dict)))
        print("Total numer of sequences missing SLAM:{0}".format(missing_slam_cnt))

        # Filter out sequences that do not contain aligned_slam_trans 
        self.data_dict = {} 
        real_cnt = 0
        for tmp_k in self.window_data_dict:
            if "aligned_slam_trans" in self.window_data_dict[tmp_k]:
                self.data_dict[real_cnt] = self.window_data_dict[tmp_k]
                real_cnt += 1
        print("Used number of sequences:{0}".format(len(self.data_dict)))

        self.for_eval = for_eval

    def filter_data(self, ori_data_dict):
        new_cnt = 0
        new_data_dict = {}
        for k in ori_data_dict:
            curr_data = ori_data_dict[k]
            seq_len = curr_data['head_qpos'].shape[0]
            num_of_files = len(curr_data['of_files'])

            if seq_len > self.window and seq_len - 1 == num_of_files:
                new_data_dict[new_cnt] = curr_data 
                new_cnt += 1 

        print("The numer of sequences in original data:{0}".format(len(ori_data_dict)))
        print("After filtering, remaining sequences:{0}".format(len(new_data_dict)))
        joblib.dump(new_data_dict, open(self.window_head_data_path, "wb"))

        return new_data_dict 
    
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

    def load_of(self, of_files):
        ofs = []
        for of_file in of_files:
            of_i = np.load(of_file)
            if self.augment and self.train:
                of_i = self.augment_flow(of_i)
            ofs.append(of_i)
        ofs = np.stack(ofs)
        
        return ofs

    def load_of_feats(self, of_files):
        ofs = []
        for of_file in of_files:
            curr_of_file = of_file.replace("/viscam/u/jiamanli/datasets/egomotion_syn_dataset", \
            "data/ares")
            of_i = np.load(curr_of_file.replace("raft_flows", "raft_of_feats"))
            ofs.append(of_i)
        ofs = np.stack(ofs) # T X D 
        
        return ofs

    def random_crop(self, image, crop_size=(224, 224)):
        h, w, _ = image.shape
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])
        bottom = top + crop_size[0]
        right = left + crop_size[1]
        image = image[top:bottom, left:right, :]

        return image

    def augment_flow(self, flow):
        """Random scaling/cropping"""
        scale_size = np.random.randint(*(230, 384))
        flow = cv2.resize(flow, (scale_size, scale_size))
        flow = self.random_crop(flow)

        """Random gaussian noise"""
        flow += np.random.normal(loc=0.0, scale=1.0, size=flow.shape).reshape(flow.shape)

        return flow

    def __len__(self):
        return len(self.data_dict)

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
        alignment_transformation = None
      
        # try:
        tmp_r, tmp_t, tmp_s = traj_est.align(traj_ref, correct_scale, only_scale, n=-1)

        print("Align xy plane scale:{0}".format(tmp_s))
        
        return tmp_r, traj_est._positions_xyz, traj_ref._positions_xyz # 3 X 3, T X 3, T X 3 

    def apply_align_on_xy_plane(self, tmp_r, aligned_slam_trans, aligned_slam_rot_mat, ori_gt_trans):
        aligned_slam_trans = torch.from_numpy(aligned_slam_trans).float()
        aligned_slam_rot_mat = torch.from_numpy(aligned_slam_rot_mat).float()
        ori_gt_trans = torch.from_numpy(ori_gt_trans).float()

        rot_mat_align_xy_plane = torch.from_numpy(tmp_r).float()

        de_headed_slam_rot_mat = torch.matmul(rot_mat_align_xy_plane[None, :, :].repeat(aligned_slam_rot_mat.shape[0], \
                            1, 1), aligned_slam_rot_mat.float())
        de_headed_slam_trans = aligned_slam_trans - aligned_slam_trans[0:1, :]
        de_headed_slam_trans = torch.matmul(rot_mat_align_xy_plane[None, :, :].repeat(aligned_slam_rot_mat.shape[0], \
                            1, 1), de_headed_slam_trans[:, :, None].float()).squeeze(-1)
        de_headed_slam_trans = de_headed_slam_trans + ori_gt_trans[0:1, :]

        return de_headed_slam_trans.data.cpu().numpy(), de_headed_slam_rot_mat.data.cpu().numpy()  

    def get_aligned_slam_traj(self, aligned_slam_trans, aligned_slam_quat, aligned_slam_rot_mat, \
        gt_trans, gt_quat, gt_rot_mat):
        traj_est = np.concatenate((aligned_slam_trans, aligned_slam_quat), axis=1)
        traj_ref = np.concatenate((gt_trans, gt_quat), axis=1)

        tmp_r, _, _ = self.align_xy_plane_traj(traj_est, traj_ref)
        de_headed_slam_trans, de_headed_slam_rot_mat = self.apply_align_on_xy_plane(tmp_r, \
        aligned_slam_trans, aligned_slam_rot_mat, gt_trans)

        de_headed_slam_quat = transforms.matrix_to_quaternion(torch.from_numpy(de_headed_slam_rot_mat).float()).data.cpu().numpy()

        return de_headed_slam_trans, de_headed_slam_quat, de_headed_slam_rot_mat

    def __getitem__(self, index):
        seq_name = self.data_dict[index]['seq_name']
        seq_head_pose = self.data_dict[index]['head_qpos'] # T X 7 
        seq_head_vels = self.data_dict[index]['head_vels'][:-1] # (T-1) X 6, the last frame's head velocity cannot calculated, remove the cloned last head vel.   
        seq_of_files = self.data_dict[index]['of_files'] # (T-1)

        seq_len = seq_head_vels.shape[0]

        if self.for_eval:
            random_t_idx = 0
            end_t_idx = seq_len
        else:
            random_t_idx = random.sample(list(range(seq_len-self.window+1)), 1)[0]
            end_t_idx = random_t_idx + self.window

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
        
        if "aligned_slam_trans" in self.data_dict[index]:
            aligned_slam_trans = self.data_dict[index]['aligned_slam_trans'][random_t_idx:end_t_idx+1]
            aligned_slam_rot_quat = self.data_dict[index]['aligned_slam_rot_quat'][random_t_idx:end_t_idx+1]
            aligned_slam_rot_mat = self.data_dict[index]['aligned_slam_rot_mat'][random_t_idx:end_t_idx+1]

            # gt_trans = window_head_pose[:, :3] # T X 3 
            # gt_rot_quat = window_head_pose[:, 3:] # T X 4 
            # gt_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(gt_rot_quat).float()).data.cpu().numpy() # T X 3 X 3 

            # # # De-head 
            # de_headed_slam_trans, de_headed_slam_quat, de_headed_slam_rot_mat = \
            # self.get_aligned_slam_traj(aligned_slam_trans, aligned_slam_rot_quat, aligned_slam_rot_mat, \
            # gt_trans, gt_rot_quat, gt_rot_mat) 

            # query['aligned_slam_trans'] = de_headed_slam_trans 
            # query['aligned_slam_rot_quat'] = de_headed_slam_quat
            # query['aligned_slam_rot_mat'] = de_headed_slam_rot_mat

            query['aligned_slam_trans'] = aligned_slam_trans 
            query['aligned_slam_rot_quat'] = aligned_slam_rot_quat
            query['aligned_slam_rot_mat'] = aligned_slam_rot_mat

            query['ori_slam_trans'] = self.data_dict[index]['ori_slam_trans'][random_t_idx:end_t_idx+1]
            query['ori_slam_rot_quat'] = self.data_dict[index]['ori_slam_rot_quat'][random_t_idx:end_t_idx+1]
            query['ori_slam_rot_mat'] = self.data_dict[index]['ori_slam_rot_mat'][random_t_idx:end_t_idx+1]

            # query['aligned_rot_mat'] = self.data_dict[index]['aligned_rot_mat']
            # query['aligned_scale'] = self.data_dict[index]['aligned_scale']
            # query['slam_floor_normal'] = self.data_dict[index]['slam_floor_normal']

        return query
