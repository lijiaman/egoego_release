import numpy as np
import os
import random
import joblib 
import pickle 

import torch
from torch.utils.data import Dataset
import pytorch3d.transforms as transforms 

class AMASSHeadPoseDataset(Dataset):
    def __init__(
        self,
        all_data_dict, 
        data_root_folder, 
        train: bool,
        window: int = 120,
        for_eval = False,
    ):
        self.train = train
        
        self.window = window

        self.for_eval = for_eval # For computing head pose estimation metrics 

        self.all_data_dict = all_data_dict 

        self.data_root_folder = data_root_folder 

        # Divide data to train and test split 
        self.divide_train_val()

        if self.train:
            self.data_dict = self.train_data_dict
        else:
            self.data_dict = self.val_data_dict 

    def augment_w_rotation(self, ori_head_pose):
        ori_head_trans = ori_head_pose[:, :3] # T X 3 
        ori_head_quat = ori_head_pose[:, 3:] # T X 4 
        ori_head_rot_mat = transforms.quaternion_to_matrix(ori_head_quat) # T X 3 X 3 

        random_rot_mat = transforms.random_rotation()[None] # 1 X 3 X 3 

        aug_head_rot_mat = torch.matmul(random_rot_mat, ori_head_rot_mat) # T X 3 X 3 

        curr_head_trans = ori_head_trans - ori_head_trans[0:1, :] # T X 3 
        aug_head_trans = torch.matmul(random_rot_mat, curr_head_trans[:, :, None])[:, :, 0] # T X 3 
        # aug_head_trans = aug_head_trans + ori_head_trans[0:1, :] # T X 3 

        ori_floor_normal = torch.from_numpy(np.asarray([0, 0, 1])).float()[:, None] # 3 X 1 
        aug_floor_normal = torch.matmul(random_rot_mat[0], ori_floor_normal) # 3 X 1 

        return random_rot_mat, aug_head_rot_mat, aug_head_trans, aug_floor_normal 

    def augment_w_scale(self, head_trans):
        # scale range: 0.1~10 
        random_scale = np.random.uniform(low=0.1, high=10, size=(1))[0]
        head_trans_diff = head_trans[1:, :] - head_trans[:-1, :] # (T-1) X 3 

        trans_diff_after_scale = head_trans_diff * random_scale # (T-1) X 3 
        
        trans_after_scale = [head_trans[0:1]] # 1 X 3 
        for t_idx in range(head_trans.shape[0]-1):
            trans_after_scale.append(trans_after_scale[-1]+trans_diff_after_scale[t_idx:t_idx+1])
        
        aug_head_trans = torch.cat(trans_after_scale, dim=0) # T X 3 

        return random_scale, aug_head_trans 

    def augment_traj(self, ori_head_pose):
        # ori_head_pose: T X 7 
        aug_rot_mat, aug_head_rot_mat, aug_head_trans, aug_floor_normal = self.augment_w_rotation(ori_head_pose)
        aug_scale, aug_head_trans = self.augment_w_scale(aug_head_trans)

        return aug_head_rot_mat, aug_head_trans, aug_rot_mat, aug_scale, aug_floor_normal

    def divide_train_val(self):
        TRAIN_DATASETS = ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje', 'BMLmovi', 
                    'EKUT', 'ACCAD'] # training dataset
        TEST_DATASETS = ['Transitions_mocap', 'HumanEva'] # test datasets
        VAL_DATASETS = ['MPI_HDM05', 'SFU', 'MPI_mosh'] # validation datasets
        
        train_data_path = os.path.join(self.data_root_folder, "train_seq_names.pkl")
        val_data_path = os.path.join(self.data_root_folder, "val_seq_names.pkl") 

        if not os.path.exists(train_data_path):
            self.train_data_dict = {}
            self.val_data_dict = {} 
            train_cnt = 0
            val_cnt = 0
        
            for seq_name in self.all_data_dict:
                curr_seq_data = self.all_data_dict[seq_name]
                curr_seq_len = curr_seq_data['head_pose'].shape[0]

                if curr_seq_len > 30:
                    if seq_name.split("-")[0] in TRAIN_DATASETS:
                        self.train_data_dict[train_cnt] = seq_name
                        train_cnt += 1 
                    else:
                        self.val_data_dict[val_cnt] = seq_name 
                        val_cnt += 1 
        else:
            self.train_data_dict = pickle.load(open(train_data_path, 'r'))
            self.val_data_dict = pickle.load(open(val_data_path, 'r'))

        print("Total sequence in the whole dataset: {0}".format(len(self.all_data_dict)))
        print("Total training sequence: {0}".format(len(self.train_data_dict)))
        print("Total validation sequence: {0}".format(len(self.val_data_dict)))

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        seq_name = self.data_dict[index]
        seq_head_pose = torch.from_numpy(self.all_data_dict[seq_name]['head_pose']).float() # T X 7 
       
        seq_len = seq_head_pose.shape[0]

        if self.for_eval:
            random_t_idx = 0
            if seq_len - self.window - 1 <= 0:
                end_t_idx = seq_len # For clear visualization 
            else:
                end_t_idx = random_t_idx + self.window + 1 
        else:
            if seq_len - self.window - 1 <= 0:
                random_t_idx = 0
                end_t_idx = seq_len 
            else:
                random_t_idx = random.sample(list(range(seq_len-self.window-1)), 1)[0]
                end_t_idx = random_t_idx + self.window + 1 

        window_head_pose = seq_head_pose[random_t_idx:end_t_idx]

        # Apply random rotation and scale to head trajectory, and then calculate the corresponding floor normal. 
        aug_head_rot_mat, aug_head_trans, aug_rot_mat, aug_scale, aug_floor_normal = self.augment_traj(window_head_pose)

        actual_seq_len = window_head_pose.shape[0]
        if actual_seq_len < self.window + 1:
            # Add padding 
            padding_head_pose = torch.zeros(self.window+1-actual_seq_len, seq_head_pose.shape[1])
            window_head_pose = torch.cat((window_head_pose, padding_head_pose), dim=0) 

            padding_aug_rot_mat = torch.zeros(self.window+1-actual_seq_len, 3, 3)
            aug_head_rot_mat = torch.cat((aug_head_rot_mat, padding_aug_rot_mat), dim=0)

            padding_trans = torch.zeros(self.window+1-actual_seq_len, 3)
            aug_head_trans = torch.cat((aug_head_trans, padding_trans), dim=0)

        query = {}
        query["ori_head_pose"] = window_head_pose # T X 7
        query['head_rot_mat'] = aug_head_rot_mat # T X 3 X 3
        query['head_trans'] = aug_head_trans # T X 3

        query['seq_len'] = actual_seq_len 
    
        query['seq_name'] = seq_name 

        query['aligned_rot_mat'] = aug_rot_mat[0].T # 3 X 3, to convert random rotated traj to original traj
        query['aligned_scale'] = 1.0/aug_scale # 1, to convert random scaled traj to original scale 
        query['floor_normal'] = aug_floor_normal # 3  

        return query
