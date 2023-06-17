import sys
sys.path.append("../")
sys.path.append("../..")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import glob
import os
import pdb
import os.path as osp
sys.path.append(os.getcwd())
import pickle
import math
import time
import glob
import numpy as np
from datetime import datetime

sys.path.append(os.getcwd())
from collections import defaultdict

# from utils_common_root import show3Dpose_animation

from relive.utils.metrics import *
from relive.utils.transformation import quaternion_matrix, quaternion_from_matrix 
from relive.envs.visual.humanoid_vis import HumanoidVisEnv
from mujoco_py import load_model_from_path, MjSim
from relive.utils.statear_smpl_config import Config
from tqdm import tqdm

import torch 

from copycat.envs.humanoid_im import HumanoidEnv as CC_HumanoidEnv
from copycat.utils.config import Config as CC_Config
from copycat.data_loaders.dataset_smpl_obj import DatasetSMPLObj
from copycat.khrylib.rl.utils.visualizer import Visualizer
import joblib

from scipy.spatial.transform import Rotation as sRot
# from utils_common_root import vis_single_head_pose_traj, vis_multiple_head_pose_traj, vis_single_head_pose_traj_2d

cc_cfg = CC_Config("copycat", "kinpoly//", create_dirs=False)
# cc_cfg.data_specs['test_file_path'] = "/Users/jiamanli/github/kin-polysample_data/h36m_train_no_sit_30_qpos.pkl"
cc_cfg.data_specs['test_file_path'] = "kinpoly/sample_data/h36m_test.pkl"   
cc_cfg.data_specs['neutral_path'] = "kinpoly/sample_data/standing_neutral.pkl"
cc_cfg.mujoco_model_file = "kinpoly/assets/mujoco_models/humanoid_smpl_neutral_mesh_all_step.xml"
    
data_loader = DatasetSMPLObj(cc_cfg.data_specs, data_mode="test")
init_expert = data_loader.sample_seq()
env = CC_HumanoidEnv(cc_cfg, init_expert = init_expert, data_specs = cc_cfg.data_specs, mode="test")

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

def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx25x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)

def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)

def compute_vel(joints):
    velocities = joints[1:] - joints[:-1]
    velocity_normed = np.linalg.norm(velocities, axis=2)
    return np.mean(velocity_normed, axis=1)

def compute_error_vel(joints_gt, joints_pred, vis = None):
    vel_gt = joints_gt[1:] - joints_gt[:-1] 
    vel_pred = joints_pred[1:] - joints_pred[:-1]
    normed = np.linalg.norm(vel_pred - vel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    return np.mean(normed[new_vis], axis=1)

def compute_metrics(results, algo, cfg_name=None, dt = 1/30, use_vis=False, return_head_pose=False):
    if results is None:
        return
    
    res_dict = defaultdict(list)
    actino_suss = defaultdict(list)

    for take in tqdm(results.keys()):
        # action = take.split("-")[0] # for original kinpoly which require action labels as input 
        action = "None" # not using action labels andy more. 
        # if args.action != "all" and action != args.action:
        #     continue
        
        res = results[take]
        traj_pred = res['qpos'].copy()
        traj_gt = res['qpos_gt'].copy()
        
        head_pose_gt = res['head_pose_gt']
        # action_one_hot = action_one_hot_dict[action]
        
        vels_gt = get_joint_vels(traj_gt, dt)
        accels_gt = get_joint_accels(vels_gt, dt)
        vels_pred = get_joint_vels(traj_pred, dt)
        accels_pred = get_joint_accels(vels_pred, dt)

        slide_pred, jpos_pred, head_pose = compute_physcis_metris(traj_pred)
        slide_gt, jpos_gt, _ = compute_physcis_metris(traj_gt)
        jpos_pred = jpos_pred.reshape(-1, 24, 3) # numpy array 
        jpos_gt = jpos_gt.reshape(-1, 24, 3)

        root_mat_pred = get_root_matrix(traj_pred)
        root_mat_gt = get_root_matrix(traj_gt)
        root_dist = get_frobenious_norm(root_mat_pred, root_mat_gt)
        root_rot_dist = get_frobenious_norm_rot_only(root_mat_pred, root_mat_gt)

        head_mat_pred = get_root_matrix(head_pose)
        head_mat_gt = get_root_matrix(head_pose_gt)
        head_dist = get_frobenious_norm(head_mat_pred, head_mat_gt)
        head_rot_dist = get_frobenious_norm_rot_only(head_mat_pred, head_mat_gt)

        vel_dist = get_mean_dist(vels_pred, vels_gt)

        accel_dist = np.mean(compute_error_accel(jpos_pred, jpos_gt)) * 1000

        smoothness = get_mean_abs(accels_pred)
        smoothness_gt = get_mean_abs(accels_gt)

        jpos_pred -= jpos_pred[:, 0:1] # zero out root
        jpos_gt -= jpos_gt[:, 0:1] 
        mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis = 2).mean() * 1000

        # Add jpe for each joint 
        single_jpe = np.linalg.norm(jpos_pred - jpos_gt, axis = 2).mean(axis=0) * 1000 # J 
        
        # Remove joints 18, 19, 20, 21 
        mpjpe_wo_hand = single_jpe[:18].mean()

        # Jiaman: add root translation error 
        pred_root_trans = traj_pred[:, :3] # T X 3
        gt_root_trans = traj_gt[:, :3] # T X 3 
        root_trans_err = np.linalg.norm(pred_root_trans - gt_root_trans, axis = 1).mean() * 1000
        res_dict["root_trans_dist"].append(root_trans_err)

        pred_head_trans = head_pose[:, :3]
        gt_head_trans = head_pose_gt[:, :3] 
        head_trans_err = np.linalg.norm(pred_head_trans - gt_head_trans, axis = 1).mean() * 1000
        res_dict["head_trans_dist"].append(head_trans_err)

        # print(succ, succ_gt, take, slide_pred)
        
        res_dict["root_dist"].append(root_dist)
        res_dict["root_rot_dist"].append(root_rot_dist)
        res_dict["mpjpe"].append(mpjpe)
        res_dict["mpjpe_wo_hand"].append(mpjpe_wo_hand)
        res_dict["head_dist"].append(head_dist)
        res_dict["head_rot_dist"].append(head_rot_dist)
        res_dict["accel_dist"].append(accel_dist)
        res_dict["slide_pred"].append(slide_pred)
        # res_dict["pen_pred"].append(pen_pred)
        # res_dict["succ"].append(succ)
        
        # res_dict["accels_pred"].append(smoothness)
        # res_dict["accels_gt"].append(smoothness_gt)
        res_dict["vel_dist"].append(vel_dist)
        # res_dict["pen_gt"].append(pen_gt)
        res_dict["slide_gt"].append(slide_gt)
        # actino_suss[action].append(succ)

        res_dict['single_jpe'].append(single_jpe)
        for tmp_idx in range(single_jpe.shape[0]):
            res_dict['jpe_'+str(tmp_idx)].append(single_jpe[tmp_idx])

    res_dict = {k: np.mean(v) for k, v in res_dict.items()}

    if return_head_pose:
        return res_dict, head_pose, head_pose_gt 
    else:
        return res_dict

def compute_foot_sliding_for_smpl(pred_global_jpos, floor_height):
    # pred_global_jpos: T X J X 3 
    seq_len = pred_global_jpos.shape[0]

    # Put human mesh to floor z = 0 and compute. 
    pred_global_jpos[:, :, 2] -= floor_height

    lankle_pos = pred_global_jpos[:, 7, :] # T X 3 
    ltoe_pos = pred_global_jpos[:, 10, :] # T X 3 

    rankle_pos = pred_global_jpos[:, 8, :] # T X 3 
    rtoe_pos = pred_global_jpos[:, 11, :] # T X 3 

    H_ankle = 0.08 # meter
    H_toe = 0.04 # meter 

    lankle_disp = np.linalg.norm(lankle_pos[1:, :2] - lankle_pos[:-1, :2], axis = 1) # T 
    ltoe_disp = np.linalg.norm(ltoe_pos[1:, :2] - ltoe_pos[:-1, :2], axis = 1) # T 
    rankle_disp = np.linalg.norm(rankle_pos[1:, :2] - rankle_pos[:-1, :2], axis = 1) # T 
    rtoe_disp = np.linalg.norm(rtoe_pos[1:, :2] - rtoe_pos[:-1, :2], axis = 1) # T 

    lankle_subset = lankle_pos[:-1, -1] < H_ankle
    ltoe_subset = ltoe_pos[:-1, -1] < H_toe
    rankle_subset = rankle_pos[:-1, -1] < H_ankle
    rtoe_subset = rtoe_pos[:-1, -1] < H_toe
   
    lankle_sliding_stats = np.abs(lankle_disp * (2 - 2 ** (lankle_pos[:-1, -1]/H_ankle)))[lankle_subset]
    lankle_sliding = np.sum(lankle_sliding_stats)/seq_len * 1000

    ltoe_sliding_stats = np.abs(ltoe_disp * (2 - 2 ** (ltoe_pos[:-1, -1]/H_toe)))[ltoe_subset]
    ltoe_sliding = np.sum(ltoe_sliding_stats)/seq_len * 1000

    rankle_sliding_stats = np.abs(rankle_disp * (2 - 2 ** (rankle_pos[:-1, -1]/H_ankle)))[rankle_subset]
    rankle_sliding = np.sum(rankle_sliding_stats)/seq_len * 1000

    rtoe_sliding_stats = np.abs(rtoe_disp * (2 - 2 ** (rtoe_pos[:-1, -1]/H_toe)))[rtoe_subset]
    rtoe_sliding = np.sum(rtoe_sliding_stats)/seq_len * 1000

    sliding = (lankle_sliding + ltoe_sliding + rankle_sliding + rtoe_sliding) / 4.

    return sliding 

def compute_metrics_for_smpl(gt_global_quat, gt_global_jpos, gt_floor_height, \
    pred_global_quat, pred_global_jpos, pred_floor_height):
    # T X J X 4, T X J X 3 

    res_dict = defaultdict(list)

    traj_pred = torch.cat((pred_global_jpos[:, 0, :], pred_global_quat[:, 0, :]), dim=-1).data.cpu().numpy() # T X 7 
    traj_gt = torch.cat((gt_global_jpos[:, 0, :], gt_global_quat[:, 0, :]), dim=-1).data.cpu().numpy() # T X 7 

    root_mat_pred = get_root_matrix(traj_pred)
    root_mat_gt = get_root_matrix(traj_gt)
    root_dist = get_frobenious_norm(root_mat_pred, root_mat_gt)
    root_rot_dist = get_frobenious_norm_rot_only(root_mat_pred, root_mat_gt)

    head_idx = 15 
    head_pose = torch.cat((pred_global_jpos[:, head_idx, :], pred_global_quat[:, head_idx, :]), dim=-1).data.cpu().numpy()
    head_pose_gt = torch.cat((gt_global_jpos[:, head_idx, :], gt_global_quat[:, head_idx, :]), dim=-1).data.cpu().numpy()

    head_mat_pred = get_root_matrix(head_pose)
    head_mat_gt = get_root_matrix(head_pose_gt)
    head_dist = get_frobenious_norm(head_mat_pred, head_mat_gt)
    head_rot_dist = get_frobenious_norm_rot_only(head_mat_pred, head_mat_gt)

    # Compute accl and accl err. 
    accels_pred = np.mean(compute_accel(pred_global_jpos.data.cpu().numpy())) * 1000
    accels_gt = np.mean(compute_accel(gt_global_jpos.data.cpu().numpy())) * 1000 

    accel_dist = np.mean(compute_error_accel(pred_global_jpos.data.cpu().numpy(), gt_global_jpos.data.cpu().numpy())) * 1000

    # Compute foot sliding error
    pred_fs_metric = compute_foot_sliding_for_smpl(pred_global_jpos.data.cpu().numpy().copy(), pred_floor_height)
    gt_fs_metric = compute_foot_sliding_for_smpl(gt_global_jpos.data.cpu().numpy().copy(), gt_floor_height)

    jpos_pred = pred_global_jpos - pred_global_jpos[:, 0:1] # zero out root
    jpos_gt =  gt_global_jpos - gt_global_jpos[:, 0:1] 
    jpos_pred = jpos_pred.data.cpu().numpy()
    jpos_gt = jpos_gt.data.cpu().numpy()
    mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis = 2).mean() * 1000

    # Add jpe for each joint 
    single_jpe = np.linalg.norm(jpos_pred - jpos_gt, axis = 2).mean(axis=0) * 1000 # J 
    
    # Remove joints 18, 19, 20, 21 
    mpjpe_wo_hand = single_jpe[:18].mean()

    # Jiaman: add root translation error 
    pred_root_trans = traj_pred[:, :3] # T X 3
    gt_root_trans = traj_gt[:, :3] # T X 3 
    root_trans_err = np.linalg.norm(pred_root_trans - gt_root_trans, axis = 1).mean() * 1000
    res_dict["root_trans_dist"].append(root_trans_err)

    # Add accl and accer 
    res_dict['accel_pred'] = accels_pred 
    res_dict['accel_gt'] = accels_gt 
    res_dict['accel_err'] = accel_dist 

    # Add foot sliding metric 
    res_dict['pred_fs'] = pred_fs_metric 
    res_dict['gt_fs'] = gt_fs_metric  

    pred_head_trans = head_pose[:, :3]
    gt_head_trans = head_pose_gt[:, :3] 
    head_trans_err = np.linalg.norm(pred_head_trans - gt_head_trans, axis = 1).mean() * 1000
    res_dict["head_trans_dist"].append(head_trans_err)

    res_dict["root_dist"].append(root_dist)
    res_dict["root_rot_dist"].append(root_rot_dist)
    res_dict["mpjpe"].append(mpjpe)
    res_dict["mpjpe_wo_hand"].append(mpjpe_wo_hand)
    res_dict["head_dist"].append(head_dist)
    res_dict["head_rot_dist"].append(head_rot_dist)
   
    res_dict['single_jpe'].append(single_jpe)
    for tmp_idx in range(single_jpe.shape[0]):
        res_dict['jpe_'+str(tmp_idx)].append(single_jpe[tmp_idx])

    res_dict = {k: np.mean(v) for k, v in res_dict.items()}
   
    return res_dict

def get_body_part(body_name):
    bone_id = env.model._body_name2id[body_name]
    head_pos = env.data.body_xpos[bone_id]
    head_quat = env.data.body_xquat[bone_id]
    return head_pos, head_quat

def compute_physcis_metris(traj):
    env.reset()
    lfoot = []
    rfoot = []
    joint_pos = []
    head_pose = []
    seq_pen = []

    pen_seq_info = []

    for fr in range(len(traj)):        
        env.data.qpos[:env.qpos_lim] = traj[fr, :]
        # env.data.qpos[env.qpos_lim:] = obj_pose[fr]
        env.sim.forward()
        
        l_feet_pos, _ = get_body_part("L_Toe")
        r_feet_pos, _ = get_body_part("R_Toe")
        lfoot.append(l_feet_pos.copy())
        rfoot.append(r_feet_pos.copy())

        head_pose.append(np.concatenate(get_body_part("Head")))
        
        joint_pos.append(env.get_wbody_pos())
   
    joint_pos = np.array(joint_pos)
    head_pose = np.array(head_pose)
    lf_slide, lf_sliding_stats = compute_foot_sliding(lfoot, traj)
    rf_slide, rf_sliding_stats = compute_foot_sliding(rfoot, traj)

    sliding  = (lf_slide + rf_slide)/2

    return sliding, joint_pos, head_pose

def compute_foot_sliding(foot_data, traj_qpos):
    seq_len = len(traj_qpos)
    H = 0.033
    z_threshold = 0.65
    z = traj_qpos[1:, 2]
    foot = np.array(foot_data).copy()
    foot[:, -1] -= np.mean(foot[:3, -1]) # Grounding it
    foot_disp = np.linalg.norm(foot[1:, :2] - foot[:-1, :2], axis = 1)

    foot_avg = (foot[:-1, -1] + foot[1:, -1])/2
    subset = np.logical_and(foot_avg < H, z > z_threshold)
    # import pdb; pdb.set_trace()

    sliding_stats = np.abs(foot_disp * (2 - 2 ** (foot_avg/H)))[subset]
    sliding = np.sum(sliding_stats)/seq_len * 1000
    return sliding, sliding_stats

def norm_qpos(qpos):
    qpos_norm = qpos.copy()
    qpos_norm[:, 3:7] /= np.linalg.norm(qpos_norm[:, 3:7], axis=1)[:, None]

    return qpos_norm

def trans2velocity(root_trans):
    # root_trans: T X 3 
    root_velocity = root_trans[1:] - root_trans[:-1]
    return root_velocity # (T-1) X 3  

def velocity2trans(init_root_trans, root_velocity):
    # init_root_trans: 3
    # root_velocity: (T-1) X 3

    timesteps = root_velocity.shape[0] + 1
    absolute_pose_data = np.zeros((timesteps, 3)) # T X 3
    absolute_pose_data[0, :] = init_root_trans.copy() 

    root_trans = init_root_trans[np.newaxis].copy() # 1 X 3
    for t_idx in range(1, timesteps):
        root_trans += root_velocity[t_idx-1:t_idx, :] # 1 X 3
        absolute_pose_data[t_idx, :] = root_trans # 1 X 3  

    return absolute_pose_data # T X 3
