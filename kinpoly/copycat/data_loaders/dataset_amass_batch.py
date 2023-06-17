from PIL import Image
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
import time
import random
import copy
import scipy.misc
import scipy.io as scio
import glob
import pickle as pk
import joblib
from collections import defaultdict

## AMASS Datatset with Class
class DatasetAMASSBatch(data.Dataset):
    def __init__(self, data_specs, mode = "all"):
        print("******* Reading AMASS Class Data, Pytorch! ***********")
        np.random.seed(0) # train test split need to stablize

        self.data_root = data_specs['file_path']
        self.pickle_data = joblib.load(open(self.data_root, "rb"))
        # self.amass_data = amass_path
        self.has_smpl_root = data_specs['has_smpl_root']
        self.load_class = data_specs['load_class']
        self.flip_cnd  = data_specs['flip_cnd']
        self.t_total = data_specs['t_total']
        self.flip_time = data_specs['flip_time']
        self.nc = data_specs['nc']
        self.to_one_hot = data_specs.get("to_one_hot", True)
        
        
        self.prepare_data()
        self.sample_keys = {}
        
        print("Dataset Root: ", self.data_root)
        print("Dataset Flip setting: ", self.flip_cnd)
        print("Dataset has SMPL root?: ", self.has_smpl_root)
        print("Dataset Num Sequences: ", self.seq_len)
        print("Traj Dimsnion: ", self.traj_dim)
        print("Load Class: ", self.load_class)
        print("******* Finished AMASS Class Data ***********")
        
    def prepare_data(self):
        self.data = self.process_data_pickle(self.pickle_data)
        self.traj_dim = list(self.data['trajs'].values())[0].shape[1]
        self.seq_len = len(list(self.data['trajs'].values()))

    def process_data_pickle(self, pk_data):
        self.data_keys = []
        trajs = {}
        target_trajs = {}
        entry_names = {}
        for k, v in pk_data.items():
            smpl_squence = v["pose"]
            if not self.has_smpl_root:
                smpl_squence = smpl_squence[:,6:132]
            import pdb
            pdb.set_trace()
            ## orig_traj -> target traj
            if smpl_squence.shape[0] >= self.t_total:
                trajs[k] = smpl_squence
                target_trajs[k] = smpl_squence
                entry_names[k] = k
                [self.data_keys.append(k) for i in range(smpl_squence.shape[0]//self.t_total)]

        
        return {
            "trajs": trajs, 
            "target_trajs": target_trajs, 
            "entry_names": entry_names
        }

    def __getitem__(self, index):

        curr_key = self.sample_keys[index]
        curr_traj = self.data['trajs'][curr_key]
        seq_len = curr_traj.shape[0]
        fr_start = torch.randint(seq_len - self.t_total, (1, )) if seq_len - self.t_total != 0 else 0
        fr_end = fr_start + self.t_total
        curr_traj = curr_traj[fr_start:fr_end]
        curr_tgt_traj = self.data['target_trajs'][curr_key][fr_start:fr_end]

        if self.flip_time:
            if np.random.binomial(1, 0.5):
                curr_tgt_traj = torch.flip(curr_tgt_traj, dims = [0])
                curr_traj = torch.flip(curr_traj, dims = [0])
        
        sample = {
            'traj': curr_traj,
            'target_trajs': curr_tgt_traj,
            'entry_name': self.data['entry_names'][curr_key], 
        }
        return sample

    def __len__(self):
        return self.dataset_len
            
    def string_to_one_hot(self, class_name):
        return np.array(self.chosen_classes == class_name).reshape(1, -1).astype(np.uint8)

    def string_to_cls_index(self, class_name):
        max_label = np.argmax(self.string_to_one_hot(class_name), axis = 1)
        return max_label

    def idx_to_one_hot(self, num_class, idx):
        hot = np.zeros(num_class)
        hot[idx] = 1
        return hot[np.newaxis,:]
    
    def sampling_generator(self, batch_size=8, num_samples=5000, num_workers = 8):
        self.sample_keys = np.random.choice(self.data_keys, num_samples, replace=True)
        self.dataset_len = len(self.sample_keys) # Change sequence length to fit 100 
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size,  shuffle=True, num_workers=num_workers)
        return loader
    
    def iter_generator(self, batch_size=8, num_workers= 8):
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size,  shuffle=False, num_workers=num_workers)
        return loader

if __name__ == '__main__':
    np.random.seed(0)
    data_specs = {
        "dataset_name": "amass_rf",
        "file_path": "/insert_directory_here/amass_take7_test.pkl",
        "flip_cnd": 0,
        "has_smpl_root": True,
        "traj_dim": 144,
        "t_total": 90,
        "nc": 2,
        "load_class": -1,
        "root_dim": 6,
        "flip_time": True
    }
    dataset = DatasetAMASS(data_specs)
    for i in range(10):
        generator = dataset.sampling_generator(num_samples=5000, batch_size=1, num_workers=1)
        for data in generator:
            import pdb
            pdb.set_trace()
        print("-------")