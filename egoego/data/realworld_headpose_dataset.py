import numpy as np
import os
import random
import joblib 
import pickle 
import json 
import cv2 
import time 

from scipy.ndimage.interpolation import rotate

import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.core import lie_algebra

import torch
from torch.utils.data import Dataset
import pytorch3d.transforms as transforms 

class RealWorldHeadPoseDataset(Dataset):
    def __init__(
        self,
        data_root_folder,
        train: bool,
        window: int = 120,
        for_eval = False,
        eval_on_kinpoly_mocap = False,
    ):
        self.train = train
        
        self.window = window

        if self.train and not for_eval:
            self.augment = True 
        else:
            self.augment = False 

        self.data_root_folder = data_root_folder

        if eval_on_kinpoly_mocap:
            slam_res_folder = os.path.join(data_root_folder, "kinpoly-mocap/droid_slam_res")

            mocap_data_path = os.path.join(data_root_folder, "kinpoly-mocap/mocap_annotations.p")
            mocap_data = joblib.load(mocap_data_path)
            test_cnt = 0
            self.data_dict = {}
            for tmp_seq_name in mocap_data:
                curr_seq_data = mocap_data[tmp_seq_name]

                slam_seq_npy = os.path.join(slam_res_folder, "-".join(tmp_seq_name.split("-")[1:])+".npy")

                if os.path.exists(slam_seq_npy):
                    # Load slam result to data folder 
                    aligned_slam_trans, aligned_slam_rot_mat, aligned_slam_quat_wxyz = self.load_slam_res_and_align_first(slam_seq_npy, curr_seq_data['head_pose'])
                    ori_slam_trans, ori_slam_rot_mat, ori_slam_quat_wxyz = self.load_data_from_droidslam(slam_seq_npy)

                    aligned_slam_trans = aligned_slam_trans[::2]
                    aligned_slam_quat_wxyz = aligned_slam_quat_wxyz[::2]
                    aligned_slam_rot_mat = aligned_slam_rot_mat[::2]

                    ori_slam_trans = ori_slam_trans[::2]
                    ori_slam_quat_wxyz = ori_slam_quat_wxyz[::2]
                    ori_slam_rot_mat = ori_slam_rot_mat[::2]

                    if aligned_slam_trans.shape[0] == curr_seq_data['head_pose'].shape[0] or aligned_slam_trans.shape[0] == curr_seq_data['head_pose'].shape[0] - 1:

                        if for_eval:
                            self.data_dict[test_cnt] = {}
                            self.data_dict[test_cnt]['seq_name'] = tmp_seq_name
                            self.data_dict[test_cnt]['head_pose'] = curr_seq_data['head_pose'][:aligned_slam_trans.shape[0]] # T X 7 (trans + quaternion) 
                            self.data_dict[test_cnt]['head_vels'] = curr_seq_data['head_vels'][:aligned_slam_trans.shape[0]] # T X 6 (linear velocity + angular velocity)

                            self.data_dict[test_cnt]['aligned_slam_trans'] = aligned_slam_trans # T X 3 
                            self.data_dict[test_cnt]['aligned_slam_rot_quat'] = aligned_slam_quat_wxyz # T X 4 
                            self.data_dict[test_cnt]['aligned_slam_rot_mat'] = aligned_slam_rot_mat # T X 3 X 3 

                            self.data_dict[test_cnt]['ori_slam_trans'] = ori_slam_trans # T X 3 
                            self.data_dict[test_cnt]['ori_slam_rot_quat'] = ori_slam_quat_wxyz # T X 4 
                            self.data_dict[test_cnt]['ori_slam_rot_mat'] = ori_slam_rot_mat # T X 3 X 3 

                            of_folder = os.path.join(self.data_root_folder, "kinpoly/fpv_of_feats")

                            ori_of_files = curr_seq_data['of_files']
                            curr_of_files = []
                            for of_path in ori_of_files:
                                new_of_path = os.path.join(of_folder, of_path.split("/")[-2], of_path.split("/")[-1])
                                curr_of_files.append(new_of_path)
                            
                            self.data_dict[test_cnt]['of_files'] = curr_of_files[:aligned_slam_trans.shape[0]]
                            
                            test_cnt += 1

            print("The number of sequences for testing Kinpoly-MoCap:{0}".format(len(self.data_dict)))
        else:
            self.head_data_path = os.path.join(data_root_folder, "kinpoly-realworld/real_annotations.p")
            self.all_data_dict = joblib.load(self.head_data_path)
            
            # Divide data to train and test split 
            self.divide_train_val()

            if self.train:
                self.data_dict = self.train_data_dict
            else:
                self.data_dict = self.val_data_dict 

            # Add slam result to data loader 
            slam_res_folder = os.path.join(data_root_folder, "kinpoly-realworld/droid_slam_res") # 2 is better than 1 and each calib 
            for tmp_k in self.data_dict:
                npy_name = "-".join(self.data_dict[tmp_k]['seq_name'].split("-")[1:])
                slam_seq_npy = os.path.join(slam_res_folder, npy_name+".npy")

                if os.path.exists(slam_seq_npy):
                    aligned_slam_trans, aligned_slam_quat_wxyz, aligned_slam_rot_mat = self.load_slam_res_and_align_first(slam_seq_npy, self.data_dict[tmp_k]['head_pose'])
                    ori_slam_trans, ori_slam_quat_wxyz, ori_slam_rot_mat = self.load_data_from_droidslam(slam_seq_npy)
                    
                    self.data_dict[tmp_k]['aligned_slam_trans'] = aligned_slam_trans
                    self.data_dict[tmp_k]['aligned_slam_rot_quat'] = aligned_slam_quat_wxyz 
                    self.data_dict[tmp_k]['aligned_slam_rot_mat'] = aligned_slam_rot_mat

                    self.data_dict[tmp_k]['ori_slam_trans'] = ori_slam_trans
                    self.data_dict[tmp_k]['ori_slam_rot_quat'] = ori_slam_quat_wxyz 
                    self.data_dict[tmp_k]['ori_slam_rot_mat'] = ori_slam_rot_mat

        self.for_eval = for_eval # For computing head pose estimation metrics 

        self.eval_on_kinpoly_mocap = eval_on_kinpoly_mocap 

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

    def divide_train_val(self):
        train_data_path = self.head_data_path.replace("real_", "window_"+str(self.window)+"_train_real_")
        val_data_path = self.head_data_path.replace("real_", "window_"+str(self.window)+"_val_real_")

        of_folder = os.path.join(self.data_root_folder, "kinpoly/fpv_of_feats")

        train_val_split_json = self.head_data_path.replace("real_annotations.p", "train_val_seq_split.json")
        if not os.path.exists(train_val_split_json):
            seq_names = list(self.all_data_dict.keys())
            total_num_seq = len(seq_names)
            train_seq_names = random.sample(seq_names, int(total_num_seq*0.85))
            val_seq_names = []
            for s_name in seq_names:
                if s_name not in train_seq_names:
                    val_seq_names.append(s_name)

            split_dict = {}
            split_dict['train_seq'] = train_seq_names
            split_dict['val_seq'] = val_seq_names
            json.dump(split_dict, open(train_val_split_json, 'w'))
        else:
            split_dict = json.load(open(train_val_split_json, 'r'))
            train_seq_names = split_dict['train_seq']
            val_seq_names = split_dict['val_seq']

        if not os.path.exists(train_data_path):
            self.train_data_dict = {}
            self.val_data_dict = {} 
            train_cnt = 0
            val_cnt = 0
       
            for s_name in train_seq_names:
                curr_seq_data = self.all_data_dict[s_name]
                curr_seq_len = curr_seq_data['head_pose'].shape[0]

                if curr_seq_len > self.window + 1:
                    self.train_data_dict[train_cnt] = {}
                    self.train_data_dict[train_cnt]['seq_name'] = s_name 
                    self.train_data_dict[train_cnt]['head_pose'] = curr_seq_data['head_pose'] # T X 7 (trans + quaternion) 
                    self.train_data_dict[train_cnt]['head_vels'] = curr_seq_data['head_vels'] # T X 6 (linear velocity + angular velocity)
                    
                    ori_of_files = curr_seq_data['of_files']
                    curr_of_files = []
                    for of_path in ori_of_files:
                        new_of_path = os.path.join(of_folder, of_path.split("/")[-2], of_path.split("/")[-1])
                        curr_of_files.append(new_of_path)

                    self.train_data_dict[train_cnt]['of_files'] = curr_of_files 
                    
                    train_cnt += 1

            for s_name in val_seq_names:
                curr_seq_data = self.all_data_dict[s_name]
                curr_seq_len = curr_seq_data['head_pose'].shape[0]

                if curr_seq_len > self.window + 1:
                    self.val_data_dict[val_cnt] = {}
                    self.val_data_dict[val_cnt]['seq_name'] = s_name 
                    self.val_data_dict[val_cnt]['head_pose'] = curr_seq_data['head_pose'] # T X 7 (trans + quaternion) 
                    self.val_data_dict[val_cnt]['head_vels'] = curr_seq_data['head_vels']
                    
                    ori_of_files = curr_seq_data['of_files']
                    curr_of_files = []
                    for of_path in ori_of_files:
                        new_of_path = os.path.join(of_folder, of_path.split("/")[-2], of_path.split("/")[-1])
                        curr_of_files.append(new_of_path)

                    self.val_data_dict[val_cnt]['of_files'] = curr_of_files 

                    val_cnt += 1 

            pickle.dump(self.train_data_dict, open(train_data_path, 'wb'))
            pickle.dump(self.val_data_dict, open(val_data_path, 'wb'))
        else:
            self.train_data_dict = joblib.load(train_data_path)
            self.val_data_dict = joblib.load(val_data_path)

        print("Total sequence in the whole dataset: {0}".format(len(self.all_data_dict)))
        print("Total training sequence: {0}".format(len(self.train_data_dict)))
        print("Total validation sequence: {0}".format(len(self.val_data_dict)))

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
            of_i = np.load(of_file)
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
        seq_head_pose = self.data_dict[index]['head_pose'] # T X 7 
        seq_head_vels = self.data_dict[index]['head_vels'][:-1] # (T-1) X 6, the last frame's head velocity cannot calculated, remove the cloned last head vel.   
        seq_of_files = self.data_dict[index]['of_files']

        seq_len = seq_head_vels.shape[0]

        if self.for_eval:
            random_t_idx = 0
            end_t_idx = seq_len
        else:
            if seq_len - self.window + 1 <= 0:
                import pdb 
                pdb.set_trace() 
            random_t_idx = random.sample(list(range(seq_len-self.window+1)), 1)[0]
            end_t_idx = random_t_idx + self.window

        window_head_pose = seq_head_pose[random_t_idx:end_t_idx+1]
        window_head_vels = seq_head_vels[random_t_idx:end_t_idx]
        window_of_files = seq_of_files[random_t_idx:end_t_idx]

        if not self.for_eval:
            assert len(window_of_files) == self.window 

        # Read optical flow data 
        window_of_data = self.load_of_feats(window_of_files) 

        actual_seq_len = window_of_data.shape[0]

        query = {}
        query["head_pose"] = window_head_pose # (T'+1) X 7
        query['head_vels'] = window_head_vels # T' X 6 
        query['of'] = window_of_data # T' X 224 X 224 X 2 / T' X D 
       
        query['seq_name'] = seq_name 

        query['seq_len'] = actual_seq_len 

        if "aligned_slam_trans" in self.data_dict[index]:
            aligned_slam_trans = self.data_dict[index]['aligned_slam_trans'][random_t_idx:end_t_idx+1]
            aligned_slam_rot_quat = self.data_dict[index]['aligned_slam_rot_quat'][random_t_idx:end_t_idx+1]
            aligned_slam_rot_mat = self.data_dict[index]['aligned_slam_rot_mat'][random_t_idx:end_t_idx+1]

            query['aligned_slam_trans'] = aligned_slam_trans 
            query['aligned_slam_rot_quat'] = aligned_slam_rot_quat
            query['aligned_slam_rot_mat'] = aligned_slam_rot_mat

            query['ori_slam_trans'] = self.data_dict[index]['ori_slam_trans'][random_t_idx:end_t_idx+1]
            query['ori_slam_rot_quat'] = self.data_dict[index]['ori_slam_rot_quat'][random_t_idx:end_t_idx+1]
            query['ori_slam_rot_mat'] = self.data_dict[index]['ori_slam_rot_mat'][random_t_idx:end_t_idx+1]

        return query
