import os
import sys
import pdb
sys.path.append(os.getcwd())
sys.path.append("../../")
import numpy as np
import glob
import pickle as pk 
import joblib
import torch 
import yaml 
import time 

from tqdm import tqdm
from copycat.utils.transform_utils import (
    convert_aa_to_orth6d, convert_orth_6d_to_aa, vertizalize_smpl_root,
    rotation_matrix_to_angle_axis, rot6d_to_rotmat
)
from scipy.spatial.transform import Rotation as sRot
from copycat.smpllib.smpl_mujoco import smpl_to_qpose, SMPL_M_Viewer
from mujoco_py import load_model_from_path, MjSim
from copycat.utils.config import Config
from copycat.envs.humanoid_im import HumanoidEnv
from copycat.utils.tools import get_expert
from copycat.data_loaders.dataset_amass_single import DatasetAMASSSingle
from copycat.data_loaders.dataset_smpl_obj import DatasetSMPLObj
from copycat.envs.humanoid_im import HumanoidEnv as CC_HumanoidEnv

from copycat.utils.config import Config as CC_Config

from collections import defaultdict

from relive.utils import *

import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.core import lie_algebra


# from relive.envs.humanoid_v2 import HumanoidEnv
# from relive.data_loaders.statereg_dataset import Dataset
# from relive.utils.statear_smpl_config import Config
# from relive.utils import get_qvel_fd, de_heading
# from relive.utils.torch_humanoid import Humanoid
from relive.utils.transformation import quaternion_multiply, quaternion_inverse,  rotation_from_quaternion
from relive.utils.transform_utils import (
    convert_6d_to_mat, compute_orth6d_from_rotation_matrix, convert_quat_to_6d
)

import csv 
import json 

def get_head_vel(head_pose, dt = 1/30):
    # get head velocity 
    head_vels = []
    head_qpos = head_pose[0]

    for i in range(head_pose.shape[0] - 1):
        curr_qpos = head_pose[i, :]
        next_qpos = head_pose[i + 1, :] 
        v = (next_qpos[:3] - curr_qpos[:3]) / dt
        v = transform_vec(v, curr_qpos[3:7], 'heading') # get velocity within the body frame
        
        qrel = quaternion_multiply(next_qpos[3:7], quaternion_inverse(curr_qpos[3:7]))
        axis, angle = rotation_from_quaternion(qrel, True) 
        
        if angle > np.pi: # -180 < angle < 180
            angle -= 2 * np.pi # 
        elif angle < -np.pi:
            angle += 2 * np.pi
        
        rv = (axis * angle) / dt
        rv = transform_vec(rv, curr_qpos[3:7], 'root')

        head_vels.append(np.concatenate((v, rv)))

    head_vels.append(head_vels[-1].copy()) # copy last one since there will be one less through finite difference
    head_vels = np.vstack(head_vels)
    return head_vels

def read_csv_to_dict():
    data_dict = {} 
    csv_path = "/orion/u/yangzheng/code/gaze_dataset/dataset.csv" 
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        row_cnt = 0
        for row in reader:
            if row_cnt > 0:
                seq_name = row[0]
                scene_name = row[3]
                start_frame = int(row[1])
                end_frame = int(row[2])

                data_dict[scene_name+"-"+seq_name] = {}
                data_dict[scene_name+"-"+seq_name]['start_frame'] = start_frame 
                data_dict[scene_name+"-"+seq_name]['end_frame'] = end_frame
            row_cnt += 1

    return data_dict 

def quat_wxyz_to_xyzw(ori_quat):
    # ori_quat: T X 4/4 
    quat_w, quat_x, quat_y, quat_z = ori_quat[:, 0:1], ori_quat[:, 1:2], ori_quat[:, 2:3], ori_quat[:, 3:4]
    pred_quat = np.concatenate((quat_x, quat_y, quat_z, quat_w), axis=1)

    return pred_quat
     
def quat_xyzw_to_wxyz(ori_quat):
    # ori_quat: T X 4/4 
    quat_x, quat_y, quat_z, quat_w = ori_quat[:, 0:1], ori_quat[:, 1:2], ori_quat[:, 2:3], ori_quat[:, 3:4]
    pred_quat = np.concatenate((quat_w, quat_x, quat_y, quat_z), axis=1)

    return pred_quat

def align_slam_traj(traj_est, traj_ref):
    # traj_est: T X 7, slam results, quat is wxyz  
    # traj_ref: T X 7, gt results 
    if traj_ref.shape[0] != traj_est.shape[0]:
        traj_ref = traj_ref[:traj_est.shape[0]] 

    assert traj_est.shape[0] == traj_ref.shape[0]

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

    # Align the trajectories, get the Transformations
    align = True 
    correct_scale=True
    only_scale = correct_scale and not align
    alignment_transformation = None
    
    tmp_r, tmp_t, tmp_s = traj_est.align(traj_ref, correct_scale, only_scale, n=-1)

    aligned_traj_xyz, aligned_traj_quat_wxyz = traj_est._positions_xyz, traj_est._orientations_quat_wxyz

    return aligned_traj_xyz, aligned_traj_quat_wxyz 

def gen_res_for_infer(data_p_path, slam_res_folder, dest_data_p_path):
    # csv_dict = read_csv_to_dict()
    info_json_path = "/viscam/u/jiamanli/datasets/gimo_processed/smplx_npz_reduced_info.json"
    csv_dict = json.load(open(info_json_path, 'r')) 

    dest_data_dict = {} 
    data = joblib.load(data_p_path)
    for seq_name in data:
        if "middle" in seq_name:
            name_tag = "middle_"+seq_name.split("_")[1]
        elif "seminar_room0" in seq_name:
            name_tag = "seminar_room0_"+seq_name.split("_")[2]
        elif "seminar_room1" in seq_name:
            name_tag = "seminar_room1_"+seq_name.split("_")[2]
        else:
            name_tag = seq_name.split("_")[0]

        scene_name = name_tag.split("-2022")[0]
        
        slam_res_path = os.path.join(slam_res_folder, scene_name, "2022"+name_tag.split("-2022")[1]+".npy")
        slam_data = np.load(slam_res_path) # T X 7 
        
        # curr_csv_data = csv_dict[name_tag]
        # bedroom0122-2022-01-21-195259_b_0_406_frames_30_fps
        # bedroom0122-2022-01-21-195259_b_0
        str_list = seq_name.split("_")[:-4]
        curr_str = "_".join(str_list)
        curr_csv_data = csv_dict[curr_str]
        s_idx = curr_csv_data['start_frame']
        e_idx = curr_csv_data['end_frame'] 

        # Align slam result to be consistent with gt before input to model 
        head_pose = data[seq_name]['head_pose'] # T X 7 
        head_trans = head_pose[:, :3] # T X 3 

        head_quat = head_pose[:, 3:] # T X 4 
        gt_r = sRot.from_quat(quat_wxyz_to_xyzw(head_quat))
        # head_gt_rot_mat = quat2mat(torch.from_numpy(head_quat).float()).data.cpu().numpy() # T X 3 X 3, from quat w, x, y, z to rotation matrix  
        head_gt_rot_mat = gt_r.as_matrix() # T X 3 X 3

        slam_head_pose = slam_data[s_idx:e_idx] # quat in slam res is w, x, y, z
        slam_trans = slam_head_pose[:, :3]
        slam_quat = slam_head_pose[:, 3:] # w, x, y, z 
        pred_r = sRot.from_quat(quat_wxyz_to_xyzw(slam_quat)) # from_quat: x, y, z, w, but!!!! qpos is w, x, y, z!!!! Bug!!!!
        slam_rot_mat = pred_r.as_matrix() # T X 3 X 3 

        use_align_lib = True 
        # Align SLAM trajectory with GT trajectory for quantitative evaluation 
        if use_align_lib:
            aligned_seq_pred_trans, aligned_seq_pred_quat = align_slam_traj(slam_head_pose, head_pose)
        else: # Not used now!!!!!
            # Calculate the relative rotation matrix 
            pred2gt_rot = np.matmul(head_gt_rot_mat[0], slam_rot_mat[0].T) # 3 X 3 
            seq_pred_rot_mat = torch.from_numpy(slam_rot_mat).float() # T X 3 X 3 
            pred2gt_rot_seq = torch.from_numpy(pred2gt_rot).float()[None, :, :] # 1 X 3 X 3 
            
            aligned_seq_pred_root_mat = torch.matmul(pred2gt_rot_seq, seq_pred_rot_mat) # T X 3 X 3 
            aligned_seq_pred_root_mat = aligned_seq_pred_root_mat.data.cpu().numpy()  

            seq_pred_trans = torch.from_numpy(slam_trans).float()[:, :, None] # T X 3 X 1 
            aligned_seq_pred_trans = torch.matmul(pred2gt_rot_seq, seq_pred_trans)[:, :, 0] # T X 3 
            aligned_seq_pred_trans = aligned_seq_pred_trans.data.cpu().numpy() 

            aligned_r = sRot.from_matrix(aligned_seq_pred_root_mat)
            aligned_seq_pred_quat = aligned_r.as_quat() # T X 4 x, y, z , w 
            aligned_seq_pred_quat = quat_xyzw_to_wxyz(aligned_seq_pred_quat) # w, x, y, z 

            delta_pred2gt_trans = head_trans[0:1, :] # 1 X 3 
            aligned_seq_pred_trans = delta_pred2gt_trans + aligned_seq_pred_trans 

        # slam_quat = quat_xyzw_to_wxyz(slam_quat) # w, x, y, z 
        # curr_slam_head_pose = np.concatenate((slam_trans, slam_quat), axis=1) # T X 7 
        curr_slam_head_pose = np.concatenate((aligned_seq_pred_trans, aligned_seq_pred_quat), axis=1)
        slam_head_vel = get_head_vel(curr_slam_head_pose) # quat in head pose should be w, x, y, z 
        dest_data_dict[seq_name] = {}
        
        ori_data = data[seq_name]
        for k in ori_data:
            if k == "head_pose":
                dest_data_dict[seq_name]['head_pose'] = curr_slam_head_pose 
            elif k == "head_vels":
                dest_data_dict[seq_name]['head_vels'] = slam_head_vel 
            else:
                dest_data_dict[seq_name][k] = data[seq_name][k] 

    joblib.dump(dest_data_dict, open(dest_data_p_path, "wb"))

if __name__ == "__main__":
    data_p_path = "/viscam/u/jiamanli/datasets/gimo_processed/gimo_motion_for_kinpoly/MoCapData/features/mocap_annotations.p"
    ori_meta_path = "/viscam/u/jiamanli/datasets/gimo_processed/gimo_motion_for_kinpoly/MoCapData/meta/mocap_meta.yml"
    
    slam_res_folder = "/viscam/u/jiamanli/datasets/gimo_processed/droid_slam_res"
    
    dest_data_folder = "/viscam/u/jiamanli/datasets/gimo_processed/gimo_motion_for_kinpoly_slam_test/MoCapData/features"
    if not os.path.exists(dest_data_folder):
        os.makedirs(dest_data_folder)
    
    # Copy meta file 
    dest_meta_folder = dest_data_folder.replace("features", "meta")
    if not os.path.exists(dest_meta_folder):
        os.makedirs(dest_meta_folder)
    dest_meta_path = os.path.join(dest_meta_folder, "mocap_meta.yml")
    shutil.copy(ori_meta_path, dest_meta_path)
    
    dest_data_p_path = os.path.join(dest_data_folder, "mocap_annotations.p")
    gen_res_for_infer(data_p_path, slam_res_folder, dest_data_p_path)
