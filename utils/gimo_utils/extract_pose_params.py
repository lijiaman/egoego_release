import torch
import numpy as np
import time
import torch.nn.functional as F
from human_body_prior.tools.model_loader import load_vposer

import os
import json
import pickle as pkl 

def extract_joint_aa():
    vposer, _ = load_vposer('./vposer_v1_0', vp_model='snapshot')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vposer = vposer.to(device)

    root_folder = "data/gimo/segmented_ori_data"
    dest_root_folder = "data/gimo/smplx_npz"
    scene_files = os.listdir(root_folder)
    for scene_name in scene_files:
        if ".py" not in scene_name and ".csv" not in scene_name:
            scene_folder = os.path.join(root_folder, scene_name)
            seq_files = os.listdir(scene_folder)
            for seq_name in seq_files:
                seq_folder = os.path.join(scene_folder, seq_name)
                smplx_folder = os.path.join(seq_folder, "smplx_local")

                if os.path.exists(smplx_folder):
                    pkl_files = os.listdir(smplx_folder)
                    pkl_files.sort() 
                    # num_pkls = len(pkl_files)//2 
                    num_pkls = len(pkl_files)//2 
                    pose_label = []
                    root_trans = []
                    root_orient = []
                    beta = None 
                    # for idx in range(num_pkls):
                    for pkl_name in pkl_files:
                        # pkl_path = os.path.join(smplx_folder, str(idx)+".pkl")
                        pkl_path = os.path.join(smplx_folder, pkl_name)
                        pkl_data = pkl.load(open(pkl_path, 'rb'))

                        curr_latent = pkl_data['latent'] # 32 
                        curr_root_trans = pkl_data['trans'] # 3
                        curr_root_orient = pkl_data['orient'] # 3 
                        pose_label.append(curr_latent)
                        root_trans.append(curr_root_trans)
                        root_orient.append(curr_root_orient) 

                        beta = pkl_data['beta'] # 10 

                    pose_label = torch.stack(pose_label) # T X 32
                    pose_label = pose_label.float().to(device)[None] # 1 X T X 32   

                    body_pose_gt = vposer.decode(pose_label, output_type='aa').reshape((-1, 21, 3)) # T X 21 X 3 

                    root_trans = torch.stack(root_trans) # T X 3 
                    root_orient = torch.stack(root_orient) # T X 3 

                    dest_folder = os.path.join(dest_root_folder, scene_name)
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    dest_npz_path = os.path.join(dest_folder, seq_name+".npz")
                    np.savez(dest_npz_path, poses=body_pose_gt.data.cpu().numpy(), \
                            root_trans=root_trans.data.cpu().numpy(), \
                            root_orient=root_orient.data.cpu().numpy(), \
                            beta=beta.data.cpu().numpy()) 


if __name__ == "__main__":
    extract_joint_aa() 
