from mujoco_py import load_model_from_path
from copycat.khrylib.utils import *
import joblib
from scipy.spatial.transform import Rotation as sRot

import glob
import os
import sys
import pdb
import os.path as osp
import copy
sys.path.append(os.getcwd())

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

from copycat.utils.transform_utils import (
    convert_aa_to_orth6d, convert_orth_6d_to_aa, vertizalize_smpl_root,
    rotation_matrix_to_angle_axis, rot6d_to_rotmat, convert_orth_6d_to_mat, angle_axis_to_rotation_matrix,
    angle_axis_to_quaternion
)

from scipy.spatial.transform import Rotation as sRot
from copycat.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_NAMES
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
from mujoco_py import load_model_from_path, MjSim


if __name__ == "__main__":
    device = (
            torch.device("cuda", index=0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    print(device)
    smpl_p = SMPL_Parser("data/smpl", gender = "male")
    smpl_p.to(device)

    smpl_model_file = 'assets/mujoco_models/humanoid_smpl_neutral_mesh.xml'
    smpl_model = load_model_from_path(smpl_model_file)
    smpl_qpose_addr = get_body_qposaddr(smpl_model)
    smpl_bone_names = list(smpl_qpose_addr.keys())
    print(smpl_bone_names)
    khry_model_file = 'assets/mujoco_models/humanoid_1205_v1.xml'
    khry_model = load_model_from_path(khry_model_file)
    khry_qpose_addr = get_body_qposaddr(khry_model)
    khry_bone_names = list(khry_qpose_addr.keys())
    print(khry_bone_names)

    smpl_sim = MjSim(smpl_model)
    khry_sim = MjSim(khry_model)

    smpl_1 = ['Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 
                    'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']


    smpl_2 = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', "LeftLeg", "RightLeg", "Spine1", "LeftFoot", \
            "RightFoot", "Spine2", "LeftToe", "RightToe", "Neck", "LeftChest", "RightChest", "Mouth", "LeftShoulder", \
            "RightShoulder", "LeftArm", "RightArm", "LeftWrist", "RightWrist", "LeftHand", "RightHand"
            ]

    smpl_chosen_bones_1 = ["Hips", "RightFoot","LeftFoot", "RightLeg", "RightUpLeg", "LeftLeg", "LeftUpLeg", "Spine", "Spine1", "Spine2",
                    "Neck", 'Mouth', "LeftChest", "LeftShoulder", "LeftArm", "LeftWrist",  "RightChest","RightShoulder", "RightArm", "RightWrist",
                ]

    khry_chosen_bones = ["Hips", "RightFoot", "LeftFoot", "RightLeg", "RightUpLeg", "LeftLeg", "LeftUpLeg", "Spine1", "Spine2", "Spine3", 
                        "Neck", 'Head', "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
                ]

    smpl_update_dict = {smpl_2[i]:smpl_1[i] for i in range(len(smpl_1))}  
    smpl_chosen_bones = []
    for i in smpl_chosen_bones_1:
        smpl_chosen_bones.append(smpl_update_dict[i])

    khry_2_smpl = {khry_chosen_bones[i] : smpl_chosen_bones[i] for i in range(len(smpl_chosen_bones))}
    smpl_2_khry = {smpl_chosen_bones[i] : khry_chosen_bones[i] for i in range(len(smpl_chosen_bones))}



    smpl_chosen_bones_1_grad = ["Hips", "RightFoot","LeftFoot", "RightLeg", "RightUpLeg", "LeftLeg", "LeftUpLeg", "Spine", "Spine1", "Spine2",
                    "Neck", "Mouth",  "LeftShoulder", "LeftArm", "LeftWrist",  "RightShoulder", "RightArm", "RightWrist",
                ]

    khry_chosen_bones_grad = ["Hips", "RightFoot", "LeftFoot", "RightLeg", "RightUpLeg", "LeftLeg", "LeftUpLeg", "Spine1", "Spine2", "Spine3", 
                        "Neck", "Head", "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand",
                ]

    smpl_chosen_bones_1_grad = ["Hips", "RightFoot","LeftFoot", "RightLeg", "RightUpLeg", "LeftLeg", "LeftUpLeg", "Spine", "Spine1", "Spine2",
                    "Neck", "Mouth",   "LeftArm", "LeftWrist",   "RightArm", "RightWrist",
                ]

    khry_chosen_bones_grad = ["Hips", "RightFoot", "LeftFoot", "RightLeg", "RightUpLeg", "LeftLeg", "LeftUpLeg", "Spine1", "Spine2", "Spine3", 
                        "Neck", "Head",  "LeftForeArm", "LeftHand",  "RightForeArm", "RightHand",
                ]


    smpl_chosen_bones = []
    for i in smpl_chosen_bones_1_grad:
        smpl_chosen_bones.append(smpl_update_dict[i])


    smpl_chosen_grad = [smpl_1.index(i) for i in smpl_chosen_bones]
    khry_chosen_grad = [khry_bone_names.index(i) for i in khry_chosen_bones_grad]


    # relive_mocap = joblib.load("/insert_directory_here/egopose_mocap_smpl.pkl")
    relive_mocap = joblib.load("/insert_directory_here/relive_mocap_smpl.pkl")

    
    relive_mocap_grad = defaultdict(dict)

    pose_aa_acc = []
    khry_qposes_acc = []
    for k in tqdm(relive_mocap.keys()):
        pose_aa = relive_mocap[k]['pose']
        khry_qposes = relive_mocap[k]['khry_qpos']
        pose_aa_acc.append(pose_aa)
        khry_qposes_acc.append(khry_qposes[:, -59:])

    pose_aa_acc = np.concatenate(pose_aa_acc)
    khry_qposes_acc = np.concatenate(khry_qposes_acc)
    

    khry_j3ds = []
    
        
    for curr_khry_qpos in khry_qposes_acc:
        if khry_qposes.shape[1] != 59:
            khry_sim.data.qpos[:] = curr_khry_qpos[-59:]
        else:
            khry_sim.data.qpos[:] = curr_khry_qpos

        khry_sim.forward()
        curr_khry_j3d = khry_sim.data.body_xpos[1:].copy()
        curr_khry_j3d = curr_khry_j3d[khry_chosen_grad]
        curr_khry_j3d -= curr_khry_j3d[0]
        khry_j3ds.append(curr_khry_j3d)
        
    khry_j3d_torch = torch.tensor(khry_j3ds).to(device)
    
    smpl_chosen_idx_torch = torch.from_numpy(np.array(smpl_chosen_grad)).to(device)
    print('---')

    chunk_size = 4096
    step_size = 50
    
    pose_aa_torch = torch.tensor(pose_aa_acc).float().to(device)
    khry_j3d_chunk = torch.split(khry_j3d_torch, chunk_size, dim=0)

    pose_aa_chunk = torch.split(pose_aa_torch, chunk_size, dim=0)
    pose_aa_chunk_acc = []


    for i in tqdm(range(len(pose_aa_chunk))):
        pose_aa_torch_new = pose_aa_chunk[i]
        khry_j3d_new = khry_j3d_chunk[i]
        for j in range(1000):
            pose_aa_torch_new =Variable(pose_aa_torch_new, requires_grad=True)
            verts, smpl_j3d_torch = smpl_p.get_joints_verts(pose_aa_torch_new)
            smpl_j3d_torch = smpl_j3d_torch[:, smpl_chosen_idx_torch]
            smpl_j3d_torch -= smpl_j3d_torch[:, 0:1].clone()
            loss = torch.norm(smpl_j3d_torch - khry_j3d_new, dim = 2).mean()
            loss.backward()
            
            pose_aa_torch_new = (pose_aa_torch_new -  pose_aa_torch_new.grad * step_size)
            if j % 100 ==0:
                print(loss.item() * 1000)
            
        pose_aa_chunk_acc.append(pose_aa_torch_new)

    pose_aa_torch = torch.cat(pose_aa_chunk_acc).detach().cpu().numpy()
    # if i % 10 ==0:
    
    seq_len_acc = 0
    for k in tqdm(relive_mocap.keys()):
        pose_aa = relive_mocap[k]['pose']
        seq_len = pose_aa.shape[0]
        
        relive_mocap_grad[k]['pose'] = pose_aa_torch[seq_len_acc:(seq_len_acc + seq_len)]
        relive_mocap_grad[k]['khry_qpos'] = relive_mocap[k]['khry_qpos']
        relive_mocap_grad[k]['trans'] = relive_mocap[k]['trans']
        relive_mocap_grad[k]['obj_pose'] = relive_mocap[k]['obj_pose']
        seq_len_acc += seq_len
    
    # np.sum([np.sum(np.abs(relive_mocap_grad[k]['pose'] -  relive_mocap[k]['pose'])) for k in relive_mocap.keys()])
    # joblib.dump(relive_mocap_grad, "/insert_directory_here/egopose_mocap_smpl_grad_stepsize.pkl")
    joblib.dump(relive_mocap_grad, "/insert_directory_here/relive_mocap_smpl_grad_stepsize.pkl")

