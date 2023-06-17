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

import time 

from relive.utils import *
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

import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.core import lie_algebra


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
        traj_est = traj_est[1:] 

    if traj_ref.shape[0] != traj_est.shape[0]:
        import pdb 
        pdb.set_trace() 
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
    dest_data_dict = {} 
    data = joblib.load(data_p_path)

    cnt = 0
    for seq_name in data:  
        scene_name = seq_name.split("-")[0]
        npz_name = "-".join(seq_name.split("-")[1:]).replace(".npz", ".npy")
        
        slam_res_path = os.path.join(slam_res_folder, scene_name, npz_name)
        # if not os.path.exists(slam_res_path):
        #     import pdb 
        #     pdb.set_trace() 
        # assert os.path.exists(slam_res_path)
        if os.path.exists(slam_res_path):
            cnt += 1 

            slam_data = np.load(slam_res_path) # T X 7 

            # Align slam result to be consistent with gt before input to model 
            head_pose = data[seq_name]['head_pose'] # T X 7 
            head_trans = head_pose[:, :3] # T X 3 

            head_quat = head_pose[:, 3:] # T X 4 
            gt_r = sRot.from_quat(quat_wxyz_to_xyzw(head_quat))
            # head_gt_rot_mat = quat2mat(torch.from_numpy(head_quat).float()).data.cpu().numpy() # T X 3 X 3, from quat w, x, y, z to rotation matrix  
            head_gt_rot_mat = gt_r.as_matrix() # T X 3 X 3

            slam_head_pose = slam_data # quat in slam res is x, y, z, w 
            slam_trans = slam_head_pose[:, :3]
            slam_quat = slam_head_pose[:, 3:] # x, y, z, w 
            pred_r = sRot.from_quat(quat_wxyz_to_xyzw(slam_quat)) # from_quat: x, y, z, w, but!!!! qpos is w, x, y, z!!!! Bug!!!!
            slam_rot_mat = pred_r.as_matrix() # T X 3 X 3 

           
            aligned_seq_pred_quat = slam_quat # w, x, y, z 
            aligned_seq_pred_trans = slam_trans 

            delta_pred2gt_trans = head_trans[0:1, :] - aligned_seq_pred_trans[0:1, :] # 1 X 3 
            aligned_seq_pred_trans = delta_pred2gt_trans + aligned_seq_pred_trans 
            
            curr_slam_head_pose = np.concatenate((aligned_seq_pred_trans, aligned_seq_pred_quat), axis=1)
            slam_head_vel = get_head_vel(curr_slam_head_pose) # quat in head pose should be w, x, y, z 

            # slam_quat = quat_xyzw_to_wxyz(slam_quat) # w, x, y, z 
            # curr_slam_head_pose = np.concatenate((slam_trans, slam_quat), axis=1) # T X 7 
            # slam_head_vel = get_head_vel(curr_slam_head_pose) # quat in head pose should be w, x, y, z 
            dest_data_dict[seq_name] = {}
            
            ori_data = data[seq_name]
            for k in ori_data:
                if k == "head_pose":
                    assert curr_slam_head_pose.shape[0] == ori_data['head_pose'].shape[0]
                    dest_data_dict[seq_name]['head_pose'] = curr_slam_head_pose
                    # assert curr_slam_head_pose.shape[0] == ori_data['head_pose'].shape[0] + 1
                    # dest_data_dict[seq_name]['head_pose'] = curr_slam_head_pose[1:] 
                elif k == "head_vels":
                    dest_data_dict[seq_name]['head_vels'] = slam_head_vel
                    # dest_data_dict[seq_name]['head_vels'] = slam_head_vel[1:] 
                else:
                    dest_data_dict[seq_name][k] = data[seq_name][k] 
        # else: # SHouldn't be used in final evaluation, here, just for debugging when partial slam results are available
        #     dest_data_dict[seq_name] = {}
            
        #     ori_data = data[seq_name]
        #     for k in ori_data:
        #         dest_data_dict[seq_name][k] = data[seq_name][k] 

    joblib.dump(dest_data_dict, open(dest_data_p_path, "wb"))

    print("original seq:{0}".format(len(data)))
    print("seq that got replaced by SLAM res:{0}".format(cnt))

if __name__ == "__main__":
    data_p_path = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/ego_syn_amass_for_kinpoly/MoCapData/features/mocap_annotations.p"
    ori_meta_path = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/ego_syn_amass_for_kinpoly/MoCapData/meta/mocap_meta.yml"
    
    # slam_res_folder = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/droid_slam_res"
    # slam_res_folder = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/head_estimator_res"
    # slam_res_folder = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/headformer_scale_slam_res"
    slam_res_folder = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/combined_headformer_and_scale_slam_res_for_s2"
    
    # dest_data_folder = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/ego_syn_amass_for_kinpoly_headformer_test/MoCapData/features"
    # dest_data_folder = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/ego_syn_amass_for_kinpoly_headformer_scale_test/MoCapData/features"
    dest_data_folder = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/ego_syn_amass_combined_headformer_and_scale_test/MoCapData/features"
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
