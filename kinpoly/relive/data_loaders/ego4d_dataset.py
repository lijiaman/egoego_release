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

class Ego4DDataset(data.Dataset):
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

        # For using kinpoly data with extracted of feats 
        # if self.cfg.use_of:
        #     data_file = osp.join(cfg.data_dir, "features", f"{self.cfg.of_file}.p")
        #     print(f"Loading of data: {data_file}")
        #     self.of_data = joblib.load(data_file)

            
        print("==================================================")
        print(f"Data file: {self.cfg.data_file}")
        print(f"Feature Version: {self.cfg.use_of}")
        print(f"Number of takes {self.len}")
        print(f"Loading data_mode: {data_mode}")
        print("Actual data seq number:{0}".format(len(self.data['relative_quat'])))
        # print(self.takes)
        print("==================================================")

    def preprocess_data(self, data_mode = "train"):

        data_all = defaultdict(list)
        of_files_acc = []
        
        take_cnt = -1
        for take in tqdm(self.takes):
            curr_expert = self.data_traj[take]
            relative_quat = curr_expert['relative_quat']

            take_cnt += 1 

            # Check for nan 
            relative_quat_tensor = torch.from_numpy(curr_expert['relative_quat'].astype('float'))
            accel_tensor = torch.from_numpy(curr_expert['accel'].astype('float')) 
            if torch.isnan(relative_quat_tensor).sum() > 0:
                continue 
            if torch.isnan(accel_tensor).sum() > 0:
                continue 
            
        
            seq_len = relative_quat.shape[0]

            
            if self.data_mode == "train" and seq_len <= self.fr_num and not self.cfg.wild:
                continue

            if self.cfg.use_of:
                of_files = curr_expert['of_files']
                if len(of_files) !=  seq_len:
                    continue
                assert(len(of_files) ==  seq_len)
                new_of_files = []
                for tmp_of in of_files:
                    new_of_files.append("/".join(tmp_of.split("/")[:-1]) + \
                        "/" + take + "/" + tmp_of.split("/")[-1])
                of_files_acc.append(new_of_files)
            
            data_all['seq_idx'].append(np.asarray([take_cnt]*seq_len)) # Added by jiaman 
            
            data_all['relative_quat'].append(curr_expert['relative_quat'].astype('float')) # t_idx is [1, ..., t-1, t-1(xx)], if remove start and end, [2, t-1] 
            data_all['accel'].append(curr_expert['accel'].astype('float')) # t_idx is [0, ..., t-1], if remove start and end, it's [1, t-2]  

           
            # break # For fast debug. 

        self.freq_indices = []
        self.all_indices =[]
        for i, traj in enumerate(data_all['relative_quat']):
            # self.freq_indices += [i for _ in range(np.ceil(traj.shape[0]/self.fr_num).astype(int))] # Each position represents a sequence index. 
            self.freq_indices += [i]
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

    def load_of(self, of_files):
        ofs = []
        for of_file in of_files:
            of_i = np.load(of_file)
            if self.cfg.augment and self.data_mode== 'train':
                of_i = self.augment_flow(of_i)
            ofs.append(of_i)
        ofs = np.stack(ofs)
        
        return ofs
    
    def load_of_feats(self, of_files):
        ofs = []
        for of_file in of_files:
            of_i = np.load(of_file.replace("raft_flows", "raft_of_feats"))
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
        from scipy.ndimage.interpolation import rotate
        """Random scaling/cropping"""
        scale_size = np.random.randint(*(230, 384))
        flow = cv2.resize(flow, (scale_size, scale_size))
        flow = self.random_crop(flow)

        """Random gaussian noise"""
        flow += np.random.normal(loc=0.0, scale=1.0, size=flow.shape).reshape(flow.shape)
        return flow

    def get_sample_from_take_ind(self, take_ind, full_sample = False):
        # self.curr_key = self.takes[take_ind]
        tmp_take_idx = self.data['seq_idx'][take_ind]
        self.curr_key = self.takes[tmp_take_idx[0]] 
        self.curr_take_ind = take_ind # index for self.data, not the index for self.takes, because some sequence may be short, not reacing the fr_num
        if full_sample:
            self.fr_start = fr_start = 0
            self.fr_end = fr_end = self.data["relative_quat"][take_ind].shape[0]
        else:
            if self.data["relative_quat"][take_ind].shape[0] - self.fr_num <= 0:
                import pdb 
                pdb.set_trace()
            else:
                self.fr_start = fr_start = np.random.randint(0, self.data["relative_quat"][take_ind].shape[0] - self.fr_num) 
                self.fr_end = fr_end = fr_start + self.fr_num
        
        data_return = {}
        for k in self.data.keys():
            data_return[k] = self.data[k][take_ind][fr_start: fr_end]
            # print("data return len:{0}".format(data_return[k].shape)) 
        if self.cfg.use_of:
            # data_return['of_files'] = self.of_files[take_ind][fr_start:fr_end]
            # data_return['of'] = self.load_of(self.of_files[take_ind][fr_start: fr_end])
            data_return['of'] = self.load_of_feats(self.of_files[take_ind][fr_start: fr_end])
            # data_return['of'] = self.of_data[self.curr_key][fr_start: fr_end] # Only for kinpoly with extracted feats
            data_return['relative_quat'] =  data_return['relative_quat'].astype('float') 
            data_return['accel'] =  data_return['accel'].astype('float') # T X 3 

            # Detect nan, set mask when nan happens 
            tmp_seq_len = data_return['accel'].shape[0]
            accel_data = data_return['accel']
            mask = np.ones((tmp_seq_len, 1))
            for tmp_idx in range(tmp_seq_len):
                if np.isnan(accel_data[tmp_idx, 0]) or \
                np.isnan(accel_data[tmp_idx, 1]) or np.isnan(accel_data[tmp_idx, 2]):
                    mask[tmp_idx] = 0 

            data_return['accel_mask'] = mask 
                
            return data_return
        else:
            return data_return

    def __getitem__(self, index):
        # sample random sequence from data
        take_ind = self.sample_indices[index] 
        return self.get_sample_from_take_ind(take_ind)

    def get_seq_len(self, index):
        return self.data["relative_quat"][index].shape[0]

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
        seq_len = self.data["relative_quat"][take_ind].shape[0] # not using the fr_num at allllllllll
        data_return = {}
        for k in self.data.keys():
            data_return[k] = self.data[k][take_ind]

        if self.cfg.use_of:
            data_return['of'] = self.load_of(self.of_files[take_ind])[None, ]
            # data_return['of'] = self.of_data[curr_key]
        self.counter += 1

        return {k: torch.from_numpy(v)[None, ] for k, v in data_return.items()}

    def sampling_generator(self, batch_size=8, num_samples=5000, num_workers=1, fr_num = 80):
        self.fr_num = int(fr_num)
        self.iter_method = "sample"
        self.data_curr = [i for i in self.freq_indices if self.data["relative_quat"][i].shape[0] > fr_num]
        # self.sample_indices = np.random.choice(self.data_curr, num_samples, replace=True)
        self.sample_indices = self.data_curr
        self.data_len = len(self.sample_indices) # Change sequence length  
        # num_workers = 0 
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size,  shuffle=True, num_workers=num_workers)
        return loader

    def sampling_generator_for_val(self, batch_size=8, num_samples=5000, num_workers=1, fr_num = 80):
        self.fr_num = int(fr_num)
        self.iter_method = "sample"
        self.data_curr = [i for i in self.freq_indices if self.data["relative_quat"][i].shape[0] > fr_num]
        self.sample_indices = self.data_curr
        self.data_len = len(self.sample_indices) # Change sequence length  
        # num_workers = 0 
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size,  shuffle=False, num_workers=num_workers)
        return loader
    
    def __len__(self):
        return self.data_len

    def iter_data(self):
        data = {}
        num_takes = len(self.data['relative_quat'])
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
            seq_len = self.data["relative_quat"][take_ind].shape[0] # not using the fr_num at allllllllll
            data_return = {}
            for k in self.data.keys():
                data_return[k] = self.data[k][take_ind][None, ]

            if self.cfg.use_of:
                data_return['of'] = self.load_of(self.of_files[take_ind])[None, ]
                data_return['of_files'] = self.of_files[take_ind] 
                # data_return['of'] = self.of_data[curr_key][None, ] # Only for kinpoly with extracted of feats 
            data_return['cl'] = np.array([0, seq_len])
                
            data[curr_key] = data_return
        return data

    def get_data(self):
        return self.data
    
