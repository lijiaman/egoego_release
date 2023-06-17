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
import joblib 
sys.path.append(os.getcwd())

import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from relive.utils import *
from relive.models.mlp import MLP
from relive.models.resnet_for_of import FeatureExtractor 
from relive.data_loaders.statear_smpl_dataset import StateARDataset
from relive.utils.torch_ext import get_scheduler
from relive.utils.statear_smpl_config import Config
from relive.utils.torch_utils import rotation_from_quaternion

from utils_common import vis_single_head_pose_traj, vis_multiple_head_pose_traj

def load_of(of_files, seq_folder):
    ofs = []
    for of_file in of_files:
        of_i = np.load(os.path.join(seq_folder, of_file))
        scale_size = 224
        of_i = cv2.resize(of_i, (scale_size, scale_size))
        ofs.append(of_i)
    ofs = np.stack(ofs)
    
    return ofs

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

    cfg = Config(args.action, args.cfg, wild = args.wild, \
        create_dirs=(args.iter == 0), mujoco_path = "assets/mujoco_models/%s.xml")
    
    """setup"""
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda', index=args.gpu_index) if (torch.cuda.is_available()) else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    traj_ar_net = FeatureExtractor()
    traj_ar_net.to(device)
    traj_ar_net.eval()

    data_folder = "/viscam/u/jiamanli/datasets/gimo_processed/raft_of"
    scene_names = os.listdir(data_folder) 
    # Generate feature for trainin dataset 
    with torch.no_grad():
        for scene_name in scene_names:
            if ".log" not in scene_name and "script" not in scene_name:
                scene_folder = os.path.join(data_folder, scene_name) 
                seq_files = os.listdir(scene_folder)
                for take in seq_files:
                    seq_folder = os.path.join(scene_folder, take)
                    ori_of_files = os.listdir(seq_folder) 
                    of_files = []
                    for tmp_f_name in ori_of_files:
                        if ".npy" in tmp_f_name:
                            of_files.append(tmp_f_name)
                    of_files.sort() 

                    tmp_folder_path = seq_folder.replace("raft_of", "raft_of_feats")
                    # if not os.path.exists(tmp_folder_path):
                    
                    seq_len = len(of_files)

                    block_size = 512 
                    num_blocks = seq_len // block_size + 1 

                    for block_idx in range(num_blocks):
                        curr_of_files = of_files[block_idx*block_size:(block_idx+1)*block_size]
                        if len(curr_of_files) > 0:
                            # try:
                            of_data = load_of(curr_of_files, seq_folder)

                            data_dict = {}
                            data_dict['of'] = torch.from_numpy(of_data).float().to(device)[None] # 1 X T X xxx  
                            feature_pred = traj_ar_net.forward(data_dict)
                            feature_pred = feature_pred.data.cpu().numpy() # B X T X 512 

                            num_bs, num_steps, _ = feature_pred.shape
                            
                            for t_idx in range(num_steps):
                                of_path = os.path.join(seq_folder, curr_of_files[t_idx])
                                dest_of_feats_path = of_path.replace("raft_of", "raft_of_feats")
                                dest_of_feats_folder = "/".join(dest_of_feats_path.split("/")[:-1])
                                if not os.path.exists(dest_of_feats_folder):
                                    os.makedirs(dest_of_feats_folder)
                                curr_of_feats = feature_pred[0, t_idx]
                                print("Save to of feats path:{0}".format(dest_of_feats_path))
                                np.save(dest_of_feats_path, curr_of_feats)
                                # except:
                                #     print("Error occured!")
                                #     print("For the sequence:{0}".format(tmp_folder_path))
                