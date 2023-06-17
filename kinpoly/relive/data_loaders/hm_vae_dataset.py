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
import glob
import pdb
import os.path as osp
import yaml
sys.path.append(os.getcwd())

import cv2
import torch.utils.data as data
import torch
import joblib
sys.path.append(os.getcwd())
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

from relive.utils import *
from relive.utils.flags import flags

class HMVAEDataset(data.Dataset):
    def __init__(self, cfg, data_mode,  seed = 0, sim = False):
        np.random.seed(seed)
        data_file = osp.join(cfg.data_dir, "features", f"{cfg.data_file}.p")

        print(f"Loading data: {data_file}")
        self.data_traj = joblib.load(data_file)

        self.cfg = cfg
        self.rotrep = cfg.rotrep
        self.meta_id = meta_id = cfg.meta_id
        self.data_mode = data_mode
        self.fr_num = cfg.fr_num
        self.overlap = cfg.fr_margin* 2
        self.base_folder = cfg.data_dir
        self.of_folder = os.path.join(self.base_folder, 'fpv_of')
        self.traj_folder = os.path.join(self.base_folder, 'traj_norm') # ZL:use traj_norm
        meta_file = osp.join(self.base_folder, "meta", f"{meta_id}.yml") 
        self.meta = yaml.load(open(meta_file, 'r'), Loader=yaml.FullLoader)
        self.msync = self.meta['video_mocap_sync']
        self.dt = 1 / self.meta['capture']['fps']
        self.pose_only = cfg.pose_only
        self.base_rot =  [0.7071, 0.7071, 0.0, 0.0]
        self.counter = 0
        # get take names
        if data_mode == 'all' or self.cfg.wild:
            self.takes = self.cfg.takes['train'] + self.cfg.takes['test']
        else:
            self.takes = self.cfg.takes[data_mode]

        self.sim = sim
        self.preprocess_data(data_mode = data_mode)
        self.len = len(self.takes)

        print("==================================================")
        print(f"Data file: {self.cfg.data_file}")
        print(f"Feature Version: {self.cfg.use_of}")
        print(f"Number of takes {self.len}")
        print(f"Loading data_mode: {data_mode}")
        # print(self.takes)
        print("==================================================")

    def preprocess_data(self, data_mode = "train"):

        data_all = defaultdict(list)
        of_files_acc = []
        
        take_cnt = -1
        for take in tqdm(self.takes):
            curr_expert = self.data_traj[take]
            gt_qpos = curr_expert['qpos']
            seq_len = gt_qpos.shape[0]

            take_cnt += 1 
            if self.data_mode == "train" and seq_len <= self.fr_num and not self.cfg.wild:
                continue

            # data that needs pre-processing
            traj_pos = self.get_traj_de_heading(gt_qpos) # t_idx is [1, ..., t-2, t-1, t-2(shouldn't bed used! just for padding?)]
            traj_root_vel = self.get_root_vel(gt_qpos) # t_idx is [1, ..., t-2, t-1, t-1(to pad to the original length)] 
            traj = np.hstack((traj_pos, traj_root_vel)) # Trajecotry and trajecotry root velocity
            # target_qvel = np.concatenate((curr_expert['qvel'][1:, :], curr_expert['qvel'][-2:-1, :]))
            target_qvel = curr_expert['qvel']
            
            # if data_mode == "train" and not self.cfg.wild:
            #     data_all['wbpos'].append(curr_expert['wbpos']) # t_idx is [0, ..., t-1]
            #     data_all['wbquat'].append(curr_expert['wbquat'])
            #     data_all['bquat'].append(curr_expert['bquat'])
            data_all['wbpos'].append(curr_expert['wbpos']) # t_idx is [0, ..., t-1]
            data_all['wbquat'].append(curr_expert['wbquat'])
            data_all['bquat'].append(curr_expert['bquat'])
            
            data_all['seq_idx'].append(np.asarray([take_cnt]*seq_len)) # Added by jiaman 
            data_all['qvel'].append(target_qvel) # t_idx is [1, 1, ..., t-1], if remove start and end, it's [1, t-2]
            data_all['target'].append(traj) # t_idx is [1, ..., t-2, t-1, xx], remove start and end, it's [2, t-1]
            data_all['qpos'].append(gt_qpos) # t_idx is [0, t-1], if remove start and end, [1, t-2]
            data_all['head_vels'].append(curr_expert['head_vels']) # t_idx is [1, ..., t-1, t-1(xx)], if remove start and end, [2, t-1] 
            data_all['head_pose'].append(curr_expert['head_pose']) # t_idx is [0, ..., t-1], if remove start and end, it's [1, t-2]  
            data_all['action_one_hot'].append(curr_expert['action_one_hot'])
            data_all['obj_head_relative_poses'].append(curr_expert['obj_head_relative_poses'][:, :7]) # Taking in only the first object's pose
            
            # if data_mode == "train"
            if self.sim:
                data_all['obj_pose'].append(curr_expert['obj_pose'])
            else:
                data_all['obj_pose'].append(curr_expert['obj_pose'][:, :7])

            # break # For fast debug. 
            
        if data == 'train' or data == 'all':
            all_traj = np.vstack(data_all['target'])

        self.traj_dim = data_all['target'][0].shape[1]
        self.freq_indices = []
        self.all_indices =[]
        for i, traj in enumerate(data_all['target']):
            # self.freq_indices += [i for _ in range(np.ceil(traj.shape[0]/self.fr_num).astype(int))] # Each position represents a sequence index. 
            self.freq_indices += [i] # not using freq
            self.all_indices.append(i) # Store the sequence index corresonding to data_all 
        
        self.freq_indices = np.array(self.freq_indices)
        self.of_files = of_files_acc

        self.data = data_all

    def get_traj_de_heading(self, orig_traj):
        # Remove trejectory-heading + remove horizontal movements 
        # results: 57 (-2 for horizontal movements)
        # Contains deheaded root orientation
        if self.cfg.has_z: # has z means that we are predicting the z directly from the model
            traj_pos = orig_traj[:, 2:].copy() # qpos without x, y
            traj_pos[:, 5:] = np.concatenate((traj_pos[1:, 5:], traj_pos[-2:-1, 5:])) # body pose 1 step forward for autoregressive target
            traj_pos[:, 0] = np.concatenate((traj_pos[1:, 0], traj_pos[-2:-1, 0])) # z 1 step forward for autoregressive target
            
            for i in range(traj_pos.shape[0]):
                # traj_pos[i, 1:5] = self.remove_base_rot(traj_pos[i, 1:5])
                traj_pos[i, 1:5] = de_heading(traj_pos[i, 1:5])
        else: # does not have z means that we are getting the z from the GT and finite-integrate
            traj_pos = orig_traj[:, 3:].copy() # qpos without x, y, z
            traj_pos[:, 4:] = np.concatenate((traj_pos[1:, 4:], traj_pos[-2:-1, 4:])) # body pose 1 step forward for autoregressive target
            for i in range(traj_pos.shape[0]):
                traj_pos[i, :4] = de_heading(traj_pos[i, :4])

        return traj_pos

    def remove_base_rot(self, quat):
        return quaternion_multiply(quat, quaternion_inverse(self.base_rot))

    def get_root_vel(self, orig_traj):
        # Get root velocity: 1x6
        traj_root_vel = []
        for i in range(orig_traj.shape[0] - 1):
            # vel = get_qvel_fd(orig_traj[i, :], orig_traj[i + 1, :], self.dt, 'heading')
            curr_qpos = orig_traj[i, :]
            next_qpos = orig_traj[i + 1, :] 
            v = (next_qpos[:3] - curr_qpos[:3]) / self.dt

            # curr_qpos[3:7] = self.remove_base_rot(curr_qpos[3:7])
            v = transform_vec(v, curr_qpos[3:7], 'heading')
            try:
                qrel = quaternion_multiply(next_qpos[3:7], quaternion_inverse(curr_qpos[3:7]))
                axis, angle = rotation_from_quaternion(qrel, True) 
            except Exception:
                import pdb
                pdb.set_trace()

            
            if angle > np.pi: # -180 < angle < 180
                angle -= 2 * np.pi # 
            elif angle < -np.pi:
                angle += 2 * np.pi
            
            rv = (axis * angle) / self.dt
            rv = transform_vec(rv, curr_qpos[3:7], 'root')

            traj_root_vel.append(np.concatenate((v, rv)))

        traj_root_vel.append(traj_root_vel[-1].copy()) # copy last one since there will be one less through finite difference
        traj_root_vel = np.vstack(traj_root_vel)
        return traj_root_vel

    def get_sample_from_take_ind(self, take_ind, full_sample = False):
        # self.curr_key = self.takes[take_ind]
        tmp_take_idx = self.data['seq_idx'][take_ind]
        self.curr_key = self.takes[tmp_take_idx[0]] 
        self.curr_take_ind = take_ind # index for self.data, not the index for self.takes, because some sequence may be short, not reacing the fr_num
        if full_sample:
            self.fr_start = fr_start = 0
            self.fr_end = fr_end = self.data["qpos"][take_ind].shape[0]
        else:
            if self.data["qpos"][take_ind].shape[0] - self.fr_num <= 0:
                import pdb 
                pdb.set_trace()
            else:
                self.fr_start = fr_start = np.random.randint(0, self.data["qpos"][take_ind].shape[0] - self.fr_num) 
                self.fr_end = fr_end = fr_start + self.fr_num
        
        data_return = {}
        for k in self.data.keys():
            data_return[k] = self.data[k][take_ind][fr_start: fr_end]
            # print("data return len:{0}".format(data_return[k].shape)) 
       
        return data_return

    def __getitem__(self, index):
        # sample random sequence from data
        take_ind = self.sample_indices[index] 
        return self.get_sample_from_take_ind(take_ind)

    def get_seq_len(self, index):
        return self.data["qpos"][index].shape[0]

    def sample_seq(self, num_samples =1 , batch_size = 1):
        self.ind = ind = np.random.choice(self.freq_indices)
        data_dict = self.get_sample_from_take_ind(ind)
        return {k: torch.from_numpy(v)[None, ] for k, v in data_dict.items()}

    def get_seq_by_ind(self, ind, full_sample = False):
        data_dict = self.get_sample_from_take_ind(ind, full_sample = full_sample)
        return {k: torch.from_numpy(v)[None, ] for k, v in data_dict.items()}

    def get_len(self):
        return len(self.takes)

    def set_seq_counter(self, idx):
        self.counter = idx
        
    def iter_seq(self):
        take_ind = self.counter % len(self.takes)
        self.curr_key = curr_key = self.takes[take_ind]
        seq_len = self.data["qpos"][take_ind].shape[0] # not using the fr_num at allllllllll
        data_return = {}
        for k in self.data.keys():
            data_return[k] = self.data[k][take_ind]

        self.counter += 1

        return {k: torch.from_numpy(v)[None, ] for k, v in data_return.items()}

    def sampling_generator(self, batch_size=8, num_samples=5000, num_workers=1, fr_num = 80):
        self.fr_num = int(fr_num)
        self.iter_method = "sample"
        self.data_curr = [i for i in self.freq_indices if self.data["qpos"][i].shape[0] > fr_num]
        self.sample_indices = self.data_curr
        # self.sample_indices = np.random.choice(self.data_curr, num_samples, replace=True)
        self.data_len = len(self.sample_indices) # Change sequence length  
        # num_workers = 0 
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size,  shuffle=True, num_workers=num_workers)
        return loader

    def sampling_generator_for_val(self, batch_size=8, num_samples=5000, num_workers=1, fr_num = 80):
        self.fr_num = int(fr_num)
        self.iter_method = "sample"
        self.data_curr = [i for i in self.freq_indices if self.data["qpos"][i].shape[0] > fr_num]
        self.sample_indices = self.data_curr
        self.data_len = len(self.sample_indices) # Change sequence length  
        # num_workers = 0 
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size,  shuffle=False, num_workers=num_workers)
        return loader
    
    def __len__(self):
        return self.data_len

    def iter_data(self):
        data = {}
        num_takes = len(self.data['qpos'])
        test_on_train = False 
        if test_on_train:
            selected_list = random.sample(list(range(num_takes)), 200) # For testing training data
        else:
            selected_list = list(range(num_takes))
        # for take_ind in range(num_takes):
        for take_ind in selected_list: # For testing training data, only select part of them 
            tmp_take_idx = self.data['seq_idx'][take_ind]
            curr_key = self.takes[tmp_take_idx[0]] 
            # curr_key = self.takes[take_ind]
            seq_len = self.data["qpos"][take_ind].shape[0] # not using the fr_num at allllllllll
            data_return = {}
            for k in self.data.keys():
                data_return[k] = self.data[k][take_ind][None, ]

            data_return['cl'] = np.array([0, seq_len])
                
            data[curr_key] = data_return
        return data

    def get_data(self):
        return self.data
    
