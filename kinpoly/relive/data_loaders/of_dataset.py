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
import os.path as osp
import yaml
from tqdm import tqdm
sys.path.append(os.getcwd())

import cv2
import torch.utils.data as data
import torch
import joblib
sys.path.append(os.getcwd())
import torch.nn.functional as F
from collections import defaultdict

class OFDatatset(data.Dataset):
    def __init__(self, cfg, data_mode,  seed = 0):
        np.random.seed(seed)
        data_file = osp.join(cfg.data_dir, "features", f"{cfg.data_file}.p")
        wild_data_file = osp.join(cfg.data_dir, "features", f"{cfg.get('data_wild_file', 'traj_wild_smpl')}.p")
        wild_data_traj = joblib.load(wild_data_file)

        self.data_traj = joblib.load(data_file)
        self.data_traj.update(wild_data_traj)
        
        self.cfg = cfg
        self.data_mode = data_mode 
        self.overlap = cfg.fr_margin* 2
        self.base_folder = cfg.data_dir
        self.of_folder = os.path.join(self.base_folder, 'fpv_of')
        self.traj_folder = os.path.join(self.base_folder, 'traj_norm') # ZL:use traj_norm
        self.meta_id = meta_id = cfg.meta_id
        meta_file = osp.join(self.base_folder, "meta", f"{meta_id}.yml") 
        self.meta = yaml.load(open(meta_file, 'r'), Loader=yaml.FullLoader)
        self.msync = self.meta['video_mocap_sync']
        self.dt = 1 / self.meta['capture']['fps']
        self.pose_only = cfg.pose_only
        self.base_rot =  [0.7071, 0.7071, 0.0, 0.0]
        # get take names
        # if data_mode == 'all':
        #     self.takes = self.cfg.takes['train'] + self.cfg.takes['test']
        # else:
        #     self.takes = self.cfg.takes[data_mode]

        self.takes = list(self.data_traj.keys())

        self.preprocess_data(data_mode = data_mode)
        self.len = self.data['head_vels'].shape[0]
        print("==================================================")
        print(f"Data file: {self.cfg.data_file}")
        print(f"Number of frames {self.len}")
        print(f"Loading data_mode: {data_mode}")
        print(f"Augment Flow: {self.cfg.augment}")
        print("==================================================")

    def preprocess_data(self, data_mode = "train"):
        data_all = defaultdict(list)
        of_files_acc = []
        for take in tqdm(self.takes):
            curr_expert = self.data_traj[take]
            of_files = np.array(curr_expert['of_files'])
            
            data_all['head_vels'].append(curr_expert['head_vels'])
            data_all['of_data'].append(of_files)
        self.data = {k:np.concatenate(v) for k, v in data_all.items()}

    def __getitem__(self, index):
        cur_idx = self.indexes[index]
        of_data = self.load_of(self.data['of_data'][cur_idx:(cur_idx+1)])[0]

        return {
            "of_data": of_data,
            "head_vels": self.data['head_vels'][cur_idx]
        }
       
    def load_of(self, of_files):
        ofs = []
        for of_file in of_files:
            of_i = np.load(of_file)
            if self.cfg.augment and self.data_mode== 'train':
                of_i = self.augment_flow(of_i)
            ofs.append(of_i)
        ofs = np.stack(ofs)
        
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

    def sample_seq(self, num_samples =1 , batch_size = 1):
        ind = np.random.choice(self.freq_indices)
        data_dict = self.get_sample_from_take_ind(ind)
        
        return {k: torch.from_numpy(v)[None, ] for k, v in data_dict.items()}
        

    def sampling_generator(self, batch_size=8,  num_workers=8):
        self.indexes = torch.randperm(self.len)
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size,  shuffle=True, num_workers=num_workers)
        return loader

    def iter_generator(self, batch_size=8,  num_workers=8):
        self.indexes = list(range(self.len))
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size,  shuffle=True, num_workers=num_workers)
        return loader
    
    def __len__(self):
        return self.len

    def iter_data(self):
        data = {}
        for take_ind in range(len(self.takes)):
            curr_key = self.takes[take_ind]
            seq_len = self.data["qpos"][take_ind].shape[0] # not using the fr_num at allllllllll
            data_return = {}
            for k in self.data.keys():
                data_return[k] = self.data[k][take_ind][None, ]

            if self.cfg.feat_v == 0 or self.cfg.feat_v == 2 or self.cfg.feat_v == 4:
                # data_return['of'] = self.load_of(self.of_files[take_ind])[None, ]
                data_return['of'] = self.of_data[curr_key][None, ]
            data_return['cl'] = np.array([0, seq_len])
                
            data[curr_key] = data_return
        return data


    def get_data(self):
        return self.data
    


