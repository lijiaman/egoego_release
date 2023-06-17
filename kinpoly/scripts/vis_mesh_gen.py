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

from utils_common_root import show3Dpose_animation

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
from utils_common_root import vis_single_head_pose_traj, vis_multiple_head_pose_traj, vis_single_head_pose_traj_2d

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
    


def convert_obj_qpos(action_one_hot, obj_pose):
    if np.sum(action_one_hot) == 0:
        obj_qos = np.zeros(35)
        for i in range(5):
            obj_qos[(i*7):(i*7+3)] = [(i + 1) * 100, 100, 0]
        return obj_qos
    else:
        action_idx = np.nonzero(action_one_hot)[0][0]
        obj_qos = np.zeros(35)
        # setting defult location for objects
        for i in range(5):
            obj_qos[(i*7):(i*7+3)] = [(i + 1) * 100, 100, 0]
            
        obj_start = action_index_map[action_idx]
        obj_end = obj_start + action_len[action_idx]
        obj_qos[obj_start:obj_end] = obj_pose

        return obj_qos


def compute_metrics(results, algo, cfg_name=None, dt = 1/30, use_vis=False):
    if results is None:
        return
    
    res_dict = defaultdict(list)
    actino_suss = defaultdict(list)

    for take in tqdm(results.keys()):
        # action = take.split("-")[0] # for original kinpoly which require action labels as input 
        action = "None" # not using action labels andy more. 
        if args.action != "all" and action != args.action:
            continue
        
        res = results[take]
        traj_pred = res['qpos'].copy()
        traj_gt = res['qpos_gt'].copy()
        

        head_pose_gt = res['head_pose_gt']
        action_one_hot = action_one_hot_dict[action]
        
        obj_pose = res['obj_pose']
        if res['obj_pose'].shape[-1] != 35:
            obj_pose = np.array([convert_obj_qpos(action_one_hot, obj_p) for obj_p in res['obj_pose']])
        

        vels_gt = get_joint_vels(traj_gt, dt)
        accels_gt = get_joint_accels(vels_gt, dt)
        vels_pred = get_joint_vels(traj_pred, dt)
        
        accels_pred = get_joint_accels(vels_pred, dt)

        pen_pred, slide_pred, jpos_pred, head_pose, succ = compute_physcis_metris(traj_pred, obj_pose, head_pose_gt = head_pose_gt, take= take, res = res)
        pen_gt, slide_gt, jpos_gt, _, succ_gt = compute_physcis_metris(traj_gt, obj_pose, head_pose_gt = head_pose_gt, take = take, res = None)
        jpos_pred = jpos_pred.reshape(-1, 24, 3) # numpy array 
        jpos_gt = jpos_gt.reshape(-1, 24, 3)

        # Visualize the skeleton. 
        if use_vis:
            cur_vis_out_folder = os.path.join("/viscam/u/jiamanli/vis_res", cfg_name.split(".")[0], take) 
            dest_vis_path = os.path.join(cur_vis_out_folder, "pred_gt_smpl_jnts_cmp.gif")
            if not os.path.exists(cur_vis_out_folder):
                os.makedirs(cur_vis_out_folder)
            vis_joints_data = np.concatenate((jpos_gt[np.newaxis], jpos_pred[np.newaxis]), axis=0) # 2 X T X 22 X 3 
            vis_block_size = 150 
            num_vis_blocks = vis_joints_data.shape[1] // vis_block_size + 1
            for tmp_b_idx in range(num_vis_blocks):
                if tmp_b_idx*vis_block_size < vis_joints_data.shape[1]:
                    if not os.path.exists(dest_vis_path.replace(".gif", "_block_"+str(tmp_b_idx)+".gif")):
                        show3Dpose_animation(vis_joints_data[:, \
                            tmp_b_idx*vis_block_size:(tmp_b_idx+1)*vis_block_size], \
                            dest_vis_path.replace(".gif", "_block_"+str(tmp_b_idx)+".gif"), use_mujoco=True) 

                        show3Dpose_animation(vis_joints_data[1:2, \
                            tmp_b_idx*vis_block_size:(tmp_b_idx+1)*vis_block_size], \
                            dest_vis_path.replace("cmp.gif", "pred_only_block_"+str(tmp_b_idx)+".gif"), use_mujoco=True) 
            
                        # Visualize corresponding input head pose 
                        cur_dest_folder = cur_vis_out_folder
                        anim_gt_head_seq_path = os.path.join(cur_dest_folder, "anim_head_traj_seq.gif").replace(".gif", "_block_"+str(tmp_b_idx)+".gif")
                        
                        # Visualize animation 
                        head_trans = head_pose[tmp_b_idx*vis_block_size:(tmp_b_idx+1)*vis_block_size, :3] # T X 3 

                        head_quat = head_pose[tmp_b_idx*vis_block_size:(tmp_b_idx+1)*vis_block_size, 3:] # T X 4 
                        gt_r = sRot.from_quat(quat_wxyz_to_xyzw(head_quat))
                        # head_gt_rot_mat = quat2mat(torch.from_numpy(head_quat).float()).data.cpu().numpy() # T X 3 X 3, from quat w, x, y, z to rotation matrix  
                        head_gt_rot_mat = gt_r.as_matrix() # T X 3 X 3

                        vis_single_head_pose_traj(head_trans, head_gt_rot_mat, anim_gt_head_seq_path)
                        # vis_single_head_pose_traj_2d(head_trans, head_gt_rot_mat, anim_gt_head_seq_path)
         

        # import pdb; pdb.set_trace()
        # print(take, pen_pred)

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
        res_dict["head_dist"].append(head_dist)
        res_dict["head_rot_dist"].append(head_rot_dist)
        res_dict["accel_dist"].append(accel_dist)
        res_dict["slide_pred"].append(slide_pred)
        res_dict["pen_pred"].append(pen_pred)
        res_dict["succ"].append(succ)
        
        # res_dict["accels_pred"].append(smoothness)
        # res_dict["accels_gt"].append(smoothness_gt)
        res_dict["vel_dist"].append(vel_dist)
        res_dict["pen_gt"].append(pen_gt)
        res_dict["slide_gt"].append(slide_gt)
        actino_suss[action].append(succ)

    res_dict = {k: np.mean(v) for k, v in res_dict.items()}
    prt_string = "".join([f"{k}:{v:.3f} \t " for k, v in res_dict.items()]) + f"--{args.cfg} | {args.iter} | {args.algo} | wild? {args.wild}" 
    logger.info(prt_string)
    print({k: np.mean(v) for k, v in actino_suss.items()})

    return res_dict


def get_body_part(body_name):
    bone_id = env.model._body_name2id[body_name]
    head_pos = env.data.body_xpos[bone_id]
    head_quat = env.data.body_xquat[bone_id]
    return head_pos, head_quat

def compute_physcis_metris(traj, obj_pose, head_pose_gt = None, take = None, res = None):
    
    env.reset()
    lfoot = []
    rfoot = []
    joint_pos = []
    head_pose = []
    seq_pen = []

    curr_action = take.split("-")[0]
    pen_seq_info = []



    for fr in range(len(traj)):
        
        env.data.qpos[:env.qpos_lim] = traj[fr, :]
        env.data.qpos[env.qpos_lim:] = obj_pose[fr]
        env.sim.forward()
        # env.render()
        margin = 0.005
        pen_acc = []
        pen_acc_check = []
        seq_len = len(obj_pose)
        # print(len(env.data.contact), env.data.ncon)

        # env.sim.model.geom_name2id("Hips")
        #https://github.com/rlworkgroup/gym-sawyer/blob/master/sawyer/mujoco/sawyer_env.py

        body_geom_range = list(range(1, 25))

        for contact in env.data.contact[:env.data.ncon]:
            if not (contact.geom1 in body_geom_range or contact.geom2 in body_geom_range):
                continue

            if (contact.geom1 in body_geom_range and contact.geom2 in body_geom_range):
                print("self collision......")
                continue

            # if contact.geom1 != 0 and  contact.geom2!= 0:
            #     print(take, contact.geom1, contact.geom2)
            #     pass
            pen = max(0, -contact.dist - margin)

            pen_acc.append([contact.geom1, contact.geom2, pen])
            pen_acc_check.append([contact.geom1, contact.geom2, -contact.dist])
        
        # print(traj[fr, :3] - get_body_part("Pelvis")[0])
        # print(np.sum(pen))
        pen_acc = np.array(pen_acc)
        pen_acc_check = np.array(pen_acc_check)
        if len(pen_acc) > 0 and np.sum(pen_acc[:, -1]) > 0:
            seq_pen.append(np.sum(pen_acc[:, -1])) 
            # print(take, fr, contact.geom1, contact.geom2, np.sum(pen_acc[:, -1]))
            
        pen_seq_info.append(pen_acc_check)
        
        l_feet_pos, _ = get_body_part("L_Toe")
        r_feet_pos, _ = get_body_part("R_Toe")
        lfoot.append(l_feet_pos.copy())
        rfoot.append(r_feet_pos.copy())

        head_pose.append(np.concatenate(get_body_part("Head")))
        
        joint_pos.append(env.get_wbody_pos())
    # import pdb; pdb.set_trace()

    succ =  compute_obj_interact(take, traj, obj_pose, pen_seq_info, head_pose, head_pose_gt, res = res)
    joint_pos = np.array(joint_pos)
    head_pose = np.array(head_pose)
    lf_slide, lf_sliding_stats = compute_foot_sliding(lfoot, traj)
    rf_slide, rf_sliding_stats = compute_foot_sliding(rfoot, traj)

    # if take == "sit-1011_take_11":
        # import pdb; pdb.set_trace()
    # print(lf_slide, rf_slide)

    sliding  = (lf_slide + rf_slide)/2
    # import pdb; pdb.set_trace()

    seq_pen = np.sum(seq_pen)/seq_len  * 1000 if len(seq_pen) > 0 else 0

    # print(sliding)    
    if np.isnan(sliding):
        import pdb; pdb.set_trace()

    
    return seq_pen, sliding, joint_pos, head_pose, succ

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

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx


def compute_obj_interact(take, traj, obj_pose, pen_seq_info, head_pose, head_pose_gt, res = None):
    curr_action = take.split("-")[0]
    succ = False
    body_geom_range = list(range(1, 25))


    if curr_action == "sit":
        chair_geom = [25, 26]
        who_hits_step = set()
        
        hit_contact = []
        for pen_info in pen_seq_info:
            hit = False
            for pen_info_ind in pen_info:
                pen_info_ind = np.array(pen_info_ind).astype(int)
                if (pen_info_ind[0] in chair_geom  or pen_info_ind[1] in chair_geom ) and (pen_info_ind[0] == 1 or pen_info_ind[1] == 1):
                   hit = True
                if (pen_info_ind[0] in chair_geom  or pen_info_ind[1] in chair_geom ) and (pen_info_ind[0] == 2 or pen_info_ind[1] == 2):
                   hit = True
                if (pen_info_ind[0] in chair_geom  or pen_info_ind[1] in chair_geom ) and (pen_info_ind[0] == 6 or pen_info_ind[1] == 6):
                   hit = True
                if (pen_info_ind[0] in chair_geom  or pen_info_ind[1] in chair_geom ) and (pen_info_ind[0] == 10 or pen_info_ind[1] == 10):
                   hit = True
                if (pen_info_ind[0] in chair_geom  or pen_info_ind[1] in chair_geom ) and (pen_info_ind[0] == 11 or pen_info_ind[1] == 11):
                   hit = True

                if (pen_info_ind[0] in chair_geom  or pen_info_ind[1] in chair_geom ):
                    who_hits_step.add(pen_info_ind[0])
                    who_hits_step.add(pen_info_ind[1])
            hit_contact.append(hit)
        cont_region = contiguous_regions(np.array(hit_contact) == 1)
        if len(cont_region) > 0:
            succ  = True
        else:
            succ = False
        # if not succ:
            # import pdb; pdb.set_trace()

    elif curr_action == "avoid":
        step_geom = [33]
        sitting_time = 1
        who_hits_step = set()
        body_geom_range = list(range(1, 13))
        hit_contact = []

        for pen_info in pen_seq_info:
            hit = False

            for pen_info_ind in pen_info:
                pen_info_ind = np.array(pen_info_ind).astype(int)
                if (pen_info_ind[0] in step_geom  or pen_info_ind[1] in step_geom ) and (pen_info_ind[0] in body_geom_range or pen_info_ind[1] in body_geom_range):
                   hit = True

                if (pen_info_ind[0] in step_geom  or pen_info_ind[1] in step_geom ) :
                    who_hits_step.add(pen_info_ind[0])
                    who_hits_step.add(pen_info_ind[1])

            hit_contact.append(hit)
        cont_region = contiguous_regions(np.array(hit_contact) == 1)

        
        head_pos = np.array(head_pose)[:, :3]
        head_pos_gt = np.array(head_pose_gt)[:, :3]
        pos_diff = np.linalg.norm(head_pos[-1] - head_pos_gt[-1]) # Mearusing the ending head pose difference (can't drift too much)
        

        if len(cont_region) > 0 or pos_diff > 0.5:
            succ = False
        else:
            succ = True
            
        # if not succ:
            # import pdb; pdb.set_trace()

    elif curr_action == "push":
        disp_threshold = 0.1
        box_pos = obj_pose[:, 7:10]
        disp = np.max(np.linalg.norm(box_pos[0] - box_pos, axis = 1))
        succ = disp > disp_threshold
        # if not succ:
            # import pdb; pdb.set_trace()

    elif curr_action == "step":
        step_geom = [34]
        
        who_hits_step = set()
        body_geom_range = [4,5,8,9]
        hit_contact = []
        pelvis_z = traj[:,2]
        pelvis_z_disp = pelvis_z - pelvis_z[0]

        for pen_info in pen_seq_info:
            hit = False
            for pen_info_ind in pen_info:
                pen_info_ind = np.array(pen_info_ind).astype(int)
                if (pen_info_ind[0] in step_geom  or pen_info_ind[1] in step_geom ) and (pen_info_ind[0] in body_geom_range or pen_info_ind[1] in body_geom_range):
                   hit = True

                if (pen_info_ind[0] in step_geom  or pen_info_ind[1] in step_geom ) :
                    who_hits_step.add(pen_info_ind[0])
                    who_hits_step.add(pen_info_ind[1])

            hit_contact.append(hit)
        cont_region = contiguous_regions(np.array(hit_contact) == 1)
        step_cont_region = contiguous_regions(pelvis_z_disp > 0.1)
        

        # if len(cont_region) > 0 and len(step_cont_region) > 0 and pelvis_z_disp_max < z_disp_threshold:
        if len(cont_region) > 0 and len(step_cont_region) > 0:
            succ = True
        else:
            succ = False

    elif curr_action == "None":
        succ = True

    if succ != True and not res is None and "fail_safe" in res and not res['fail_safe']:
        print("success but failed interaction: ", take)

    # if take == "step-2021-05-04-20-19-04":
        # import pdb; pdb.set_trace()

    if not res is None and "fail_safe" in res:
        succ = succ and not res['fail_safe']
    # root_delta = np.linalg.norm(traj[1:,:3] - traj[:-1,:3], axis = 1) # capturing fail safe
    # if np.max(root_delta) > 0.1: # Moving more than 10 cm 
    #     print("Fail safe abuse!", take)
    #     succ = False
            

    return succ
class ReliveVisulizer(Visualizer):

    def __init__(self, vis_file):
        super().__init__(vis_file)
        ngeom = 24
        # self.env_vis.model.geom_rgba[ngeom + 1: ngeom * 2 - 1] = np.array([0.7, 0.0, 0.0, 1])
        self.env_vis.model.geom_rgba[ngeom + 1: ngeom * 2 - 1] = np.array([0, 1.0, 1, 1])

        self.env_vis.viewer.cam.lookat[2] = 1.0
        self.env_vis.viewer.cam.azimuth = 45
        self.env_vis.viewer.cam.elevation = -8.0
        self.env_vis.viewer.cam.distance = 5.0
        self.T = 12

    def data_generator(self):
        results = sr_res

        for take in tqdm(results.keys()):
            res = results[take]

            print(take)
            curr_action = take.split("-")[0]
            poses = {}

            poses['gt'] = res['qpos_gt']
            poses['pred'] = res['qpos']
            poses['obj_pose'] = res['obj_pose']

            if poses['obj_pose'].shape[-1] != 35:
                poses['obj_pose'] = data_loader.convert_obj_qpos(res['obj_pose'], curr_action)
            
            self.num_fr = poses['pred'].shape[0]
            yield poses

    def update_pose(self):
        self.env_vis.data.qpos[:76] = self.data['pred'][self.fr]
        self.env_vis.data.qpos[76:152] = self.data['gt'][self.fr]
        # self.env_vis.data.qpos[76] = 100
        self.env_vis.data.qpos[152:] = self.data['obj_pose'][self.fr]
        # self.env_vis.data.qpos[152:] = self.data['obj_pose'][self.fr]

        self.env_vis.sim_forward()
        

    def record_video(self):
        frame_dir = f'{args.video_dir}/frames'
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)
        for fr in range(self.num_fr):
            self.fr = fr
            self.update_pose()
            for _ in range(20):
                self.render()
            if not args.preview:
                t0 = time.time()
                save_screen_shots(self.env_vis.viewer.window, f'{frame_dir}/%04d.png' % fr)
                print('%d/%d, %.3f' % (fr, self.num_fr, time.time() - t0))

        if not args.preview:
            out_name = f'{args.video_dir}/{args.cfg}_{"expert" if args.record_expert else args.iter}.mp4'
            cmd = ['/usr/local/bin/ffmpeg', '-y', '-r', '30', '-f', 'image2', '-start_number', '0',
                '-i', f'{frame_dir}/%04d.png', '-vcodec', 'libx264', '-crf', '5', '-pix_fmt', 'yuv420p', out_name]
            subprocess.call(cmd)
        

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='stats')
    parser.add_argument('--data', default='test')
    parser.add_argument('--wild', action='store_true', default=False)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--samp', type=int, default=0)
    parser.add_argument('--action', type=str, default='all')
    parser.add_argument('--algo', type=str, default='statear')
    parser.add_argument('--amass', action='store_true', default=False)
    parser.add_argument('--syn-amass', action='store_true', default=False)
    parser.add_argument('--gimo', action='store_true', default=False)
    parser.add_argument('--locomotion', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--viz', action='store_true', default=False)
    args = parser.parse_args()

    logger = create_logger(os.path.join("results", 'log_eval.txt'))

    if os.path.exists("/viscam/u/jiamanli/github/egomotion/kin-poly"):
        cc_cfg = CC_Config("copycat", "/viscam/u/jiamanli/github/egomotion/kin-poly//", create_dirs=False)

        # cc_cfg.data_specs['test_file_path'] = "/Users/jiamanli/github/kin-polysample_data/h36m_train_no_sit_30_qpos.pkl"
        cc_cfg.data_specs['test_file_path'] = "/viscam/u/jiamanli/github/egomotion/kin-poly/sample_data/h36m_test.pkl"
    else: # Local path 
        cc_cfg = CC_Config("copycat", "/Users/jiamanli/github/egomotion/kin-poly//", create_dirs=False)
        cc_cfg.data_specs['test_file_path'] = "/Users/jiamanli/github/egomotion/kin-poly/sample_data/h36m_test.pkl"
    
    if args.wild:
        cc_cfg.mujoco_model_file = "humanoid_smpl_neutral_mesh_all.xml"
    else:
        cc_cfg.mujoco_model_file = "humanoid_smpl_neutral_mesh_all_step.xml"
    
    data_loader = DatasetSMPLObj(cc_cfg.data_specs, data_mode="test")
    init_expert = data_loader.sample_seq()
    env = CC_HumanoidEnv(cc_cfg, init_expert = init_expert, data_specs = cc_cfg.data_specs, mode="test")

    
    action_one_hot_dict = {
        "sit": np.array([1,0,0,0]),
        "push": np.array([0,1,0,0]),
        "avoid": np.array([0,0,1,0]),
        "step": np.array([0,0,0,1]),
        "None": np.array([0,0,0,0]),
    }

    action_index_map = [0, 7, 21, 28]
    action_len = [7, 14, 7, 7]
    action_names = ["sit", "push", "avoid", "step"]
    # sr_res_path = 'results/%s/%s/%s/results/iter_%04d_%s.p' % (args.action, args.algo, args.cfg, args.iter, args.data)
  
    if os.path.exists("/viscam/u/jiamanli/datasets"):
        if args.amass:
            all_data = joblib.load("/viscam/u/jiamanli/datasets/amass_processed_for_kinpoly/MoCapData/features/mocap_annotations.p")
        elif args.locomotion:
            all_data = joblib.load("/viscam/u/jiamanli/datasets/amass_processed_all_locomotion_for_kinpoly/MoCapData/features/mocap_annotations.p")
        elif args.gimo:
            all_data = joblib.load("/viscam/u/jiamanli/datasets/gimo_processed/gimo_motion_for_kinpoly/MoCapData/features/mocap_annotations.p")
        elif args.syn_amass:
            all_data = joblib.load("/viscam/u/jiamanli/datasets/egomotion_syn_dataset/ego_syn_amass_for_kinpoly/MoCapData/features/mocap_annotations.p")
        
            TEST_SCENES = ['office_0', 'hotel_0', 'room_2', 'frl_apartment_4', 'apartment_0']
        else:
            all_data = joblib.load("/viscam/u/jiamanli/datasets/kin-poly/MoCapData/features/mocap_annotations.p")
            # all_data = joblib.load("/viscam/u/jiamanli/datasets/kin-poly/MoCapData_blocks_max_150/features/mocap_annotations.p")
    else:
        # all_data = joblib.load("/Users/jiamanli/datasets/kin-poly/MoCapData/features/mocap_annotations.p")
        all_data = joblib.load("/Users/jiamanli/datasets/kin-poly/MoCapData_blocks_max_150/features/mocap_annotations.p")

    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/MoCapData/features/mocap_annotations.p"
    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/kinpoly_transpose_fps60_res.p"
    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/kinpoly_transpose_fps60_res_debug.p"
    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/kinpoly_transpose_res.p"
    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/kinpoly_tip_res_for_eval/kinpoly_run_tip_res.p"
    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/head2root_cmp_vis/slam_res_for_eval.p"
    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/head2root_cmp_vis_using_tip/slam_res_using_tip_for_eval.p"
    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/head2root_cmp_vis_using_transpose/slam_res_using_transpose_for_eval.p"
    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/head2root_cmp_vis_using_transpose/slam_res_using_transpose_for_eval_fps60.p"

    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/humor_res/ori_humor_res/humor_for_eval_res_2.p"
    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/humor_res/video_condition_humor_res/humor_for_eval_res_2.p"
    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/humor_res/video_condition_reg_res/humor_for_eval_res_0.p"
    # sr_res_path = "/Users/jiamanli/datasets/kin-poly/humor_res/video_condition_input_cam_pose_reg_res/humor_for_eval_res_0.p"
    # sr_res_path = "/viscam/u/jiamanli/datasets/kin-poly/humor_res/video_condition_input_cam_pose_v2_reg_res/humor_for_eval_res_0.p"
    
    # sr_res_path = "/viscam/u/jiamanli/results/kinpoly/all/statear/kin_poly_wo_action_w_of/results/iter_1000_test_mocap_annotations.p"
    # sr_res_path = "/viscam/u/jiamanli/results/kinpoly/all/statear/kin_poly_wo_action/results/iter_1000_test_mocap_annotations.p"
    # sr_res_path = "/viscam/u/jiamanli/results/kinpoly/all/statear/kin_poly_wo_action/results/iter_2000_test_mocap_annotations.p"
    # sr_res_path = "/viscam/u/jiamanli/results/kinpoly/all/statear/kin_poly_w_action_w_of/results/iter_1000_test_mocap_annotations.p"
    # sr_res_path = "/viscam/u/jiamanli/results/kinpoly/all/statear/kin_poly_w_action/results/iter_1000_test_mocap_annotations.p"
    # sr_res_path = "/viscam/u/jiamanli/results/kinpoly/all/statear/kin_poly_w_action_v2/results/iter_1000_test_mocap_annotations.p"
    # sr_res_path = "/viscam/u/jiamanli/results/kinpoly/all/statear/kin_poly_w_action/results/iter_2000_test_mocap_annotations.p"
    # sr_res_path = "/Users/jiamanli/github/egomotion/kin-poly/results/all/statear/kin_poly_wo_action/results/iter_1000_test_mocap_annotations.p"
    # sr_res_path = "/Users/jiamanli/github/egomotion/kin-poly/results/all/statear/kin_poly/results/1000_mocap_annotations_coverage_full.pkl"
    sr_res_path = f"/viscam/u/jiamanli/results/kinpoly/all/statear/{args.cfg}/results/iter_{args.iter:04d}_{args.data}_mocap_annotations.p"
    if not os.path.exists(sr_res_path):
        sr_res_path = f"/viscam/u/jiamanli/results/kinpoly/all/statear/{args.cfg}/results/iter_{args.iter:04d}_{args.data}_mocap_annotations_samp_{args.samp:02d}.p"
    # sr_res_path = "/viscam/u/jiamanli/datasets/kin-poly/humor_res/video_condition_context_transformer_set_1_reg_res/humor_for_eval_res_0.p"
    # sr_res_path = "/viscam/u/jiamanli/datasets/kin-poly/humor_res/video_condition_input_cam_pose_v3_reg_res/humor_for_eval_res_0.p"
    use_humor_res = False 
    debug = args.debug  # If debug is true, print each sequence's errors, and visualize skeleton seq. 
    print("args:{0}".format(args))

    if debug:
        # Save processed qpos to files
        # dest_debug_folder = "/Users/jiamanli/datasets/kin-poly/for_debug_kinpoly_eval"
        dest_debug_folder = "/viscam/u/jiamanli/datasets/kin-poly/for_debug_kinpoly_eval"
        if not os.path.exists(dest_debug_folder):
            os.makedirs(dest_debug_folder)
        dest_pred_path = os.path.join(dest_debug_folder, "kinpoly_video_cond_reg_pred_res.p")
        dest_gt_path = os.path.join(dest_debug_folder, "kinpoly_video_cond_reg_gt_res.p")

        debug_pred_dict = {}
        debug_gt_dict = {} 

    # ego_res, _ = pickle.load(open(sr_res_path, 'rb')) if args.cfg is not None else (None, None)
    # ego_res = all_data # For debug 
    ego_res = joblib.load(sr_res_path)
    sr_res = defaultdict(dict)
    for k in ego_res.keys():
        # import pdb 
        # pdb.set_trace()
    # for k in all_data.keys(): # For debug 
        if k in all_data:
        # if k in all_data and "walk" in k: 
        # if k in all_data and k.split("-")[0] in TEST_SCENES:
        # if k in all_data and k.split("-")[0] in TEST_SCENES and "walk" in k:
            if use_humor_res:
                # 1. Need to remove the first and last frame from sequence since HuMoR processed data miss the start and end frames.
                # 2. Rotate the first frame of result to the consistent rotation with gt.
                # 3. Translate the first frame to be ligned with gt translation. 
                pred_qpos = ego_res[k]['qpos']
                gt_qpos = all_data[k]['qpos'][1:-1] # For HuMoR model
                # gt_qpos = all_data[k]['qpos']

                pred_root_quat = pred_qpos[0, 3:7] # 4
                gt_root_quat = gt_qpos[0, 3:7] 

                pred_root_quat = quat_wxyz_to_xyzw(pred_root_quat[np.newaxis]) # 1 X 4 
                gt_root_quat = quat_wxyz_to_xyzw(gt_root_quat[np.newaxis]) # 1 X 4 

                # if np.isnan(pred_root_quat[0]):
                #     print("NaN!!!Skip!!!")
                #     continue

                # print("pred root quat:{0}".format(pred_root_quat))
                # pred_root_mat = quaternion_matrix(pred_root_quat)[:3, :3]
                # gt_root_mat = quaternion_matrix(gt_root_mat)[:3, :3] 

                
                pred_r = sRot.from_quat(pred_root_quat) # from_quat: x, y, z, w, but!!!! qpos is w, x, y, z!!!! Bug!!!!
                pred_root_mat = pred_r.as_matrix() # (1) X 3 X 3 
                gt_r = sRot.from_quat(gt_root_quat) 
                gt_root_mat = gt_r.as_matrix() # (1) X 3 X 3 
                pred2gt_rot = np.matmul(gt_root_mat[0], pred_root_mat[0].T) # 3 X 3 
                # pred2gt_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])[0].data.cpu().numpy() 
                # pred2gt_rot = pred2gt_rot.T 
                # pred2gt_rot = torch.from_numpy(gt_root_mat).float().transpose(1, \
                #     2).bmm(torch.from_numpy(pred_root_mat).float()).data.cpu().numpy()[0] 


                seq_pred_root_quat = pred_qpos[:, 3:7] # T X 4, w, x, y, z 
                seq_pred_root_quat = quat_wxyz_to_xyzw(seq_pred_root_quat) # T X 4, x, y, z, w 
                seq_pred_r = sRot.from_quat(seq_pred_root_quat) 
                seq_pred_root_mat = seq_pred_r.as_matrix() # T X 3 X 3  

                pred2gt_rot_seq = torch.from_numpy(pred2gt_rot).float()[None, :, :] # 1 X 3 X 3 
                seq_pred_root_mat = torch.from_numpy(seq_pred_root_mat).float() # T X 3 X 3 
                aligned_seq_pred_root_mat = torch.matmul(pred2gt_rot_seq, \
                    seq_pred_root_mat) # T X 3 X 3 
                aligned_r = sRot.from_matrix(aligned_seq_pred_root_mat.data.cpu().numpy()) 
                aligned_pred_quat = aligned_r.as_quat() # T X 4, x, y, z, w  

                # rotate translation velocity
                pred_ori_root_v = trans2velocity(pred_qpos[:, :3]) # (T-1) X 3
                pred_aligned_root_v = torch.matmul(pred2gt_rot_seq.repeat(pred_ori_root_v.shape[0], 1, 1), \
                    torch.from_numpy(pred_ori_root_v).float()[:, :, None]).squeeze(-1) # (T-1) X 3
                aligned_trans = velocity2trans(gt_qpos[0, :3], pred_aligned_root_v.data.cpu().numpy()) # T X 3 

                # rotate translation 
                # pred_trans = torch.from_numpy(pred_qpos[:, :3]).float() # T X 3 
                # aligned_trans = torch.matmul(aligned_seq_pred_root_mat, pred_trans[:, :, None]).squeeze(-1) # T X 3 

                pred_qpos[:, :3] = aligned_trans
                pred_qpos[:, 3:7] = quat_xyzw_to_wxyz(aligned_pred_quat)  

                # align translation to start from the same point as gt 
                # delta_trans = gt_qpos[0:1, :3] - pred_qpos[0:1, :3] # 1 X 3
                # pred_qpos[:, :3] += delta_trans  
                if debug:
                    debug_pred_dict[k] = {}
                    debug_pred_dict[k]['qpos'] = pred_qpos 

                    debug_gt_dict[k] = {}
                    debug_gt_dict[k]['qpos'] = gt_qpos 

                sr_res[k] = {
                    # "qpos": norm_qpos(ego_res['traj_pred'][k]), 
                    # "qpos_gt": norm_qpos(ego_res['traj_orig'][k]), 
                    # "qpos": norm_qpos(ego_res[k]['qpos']), 
                    # "qpos_gt": norm_qpos(all_data[k]['qpos']), 
                    # "qpos": norm_qpos(pred_qpos[::2, :]), 
                    "qpos": norm_qpos(pred_qpos), 
                    "qpos_gt": norm_qpos(gt_qpos), 
                    "obj_pose": all_data[k]['obj_pose'], 
                    "head_pose_gt": all_data[k]['head_pose']
                }

            else:
                pred_qpos = ego_res[k]['qpos']
                # pred_qpos = np.asarray(ego_res[k]['pred'])
                # pred_qpos = ego_res[k]
                # gt_qpos = all_data[k]['qpos']
                gt_qpos = ego_res[k]['qpos_gt']
                # import pdb 
                # pdb.set_trace() 
                gt_head_pose = all_data[k]['head_pose']
                
                if pred_qpos.shape[0] != gt_qpos.shape[0]: 
                    gt_qpos = gt_qpos[:pred_qpos.shape[0]]
                    # if pred_qpos.shape[0] != gt_qpos.shape[0]: 
                    #     print(k)
                    #     continue

                if pred_qpos.shape[0] != gt_head_pose.shape[0]: 
                    # gt_head_pose = gt_head_pose[:-1]
                    gt_head_pose = gt_head_pose[:pred_qpos.shape[0]]
                # delta_trans = gt_qpos[0:1, :3] - pred_qpos[0:1, :3] # 1 X 3
                # pred_qpos[:, :3] += delta_trans  


                if debug:
                    debug_pred_dict[k] = {}
                    debug_pred_dict[k]['qpos'] = pred_qpos 

                    debug_gt_dict[k] = {}
                    debug_gt_dict[k]['qpos'] = gt_qpos 
                sr_res[k] = {
                    # "qpos": norm_qpos(ego_res['traj_pred'][k]), 
                    # "qpos_gt": norm_qpos(ego_res['traj_orig'][k]), 
                    # "qpos": norm_qpos(ego_res[k]['qpos']), 
                    # "qpos_gt": norm_qpos(all_data[k]['qpos']), 
                    # "qpos": norm_qpos(pred_qpos[::2, :]), 
                    "qpos": norm_qpos(pred_qpos), 
                    "qpos_gt": norm_qpos(gt_qpos), 
                    "obj_pose": all_data[k]['obj_pose'], 
                    "head_pose_gt": gt_head_pose
                }
                print("pred qpos:{0}".format(pred_qpos.shape))
                print("head pose gt:{0}".format(gt_head_pose.shape))
                if debug:
                    new_sr_res = {}
                    new_sr_res[k] = sr_res[k]
                    print("Seq name:{0}".format(k))
                    compute_metrics(new_sr_res, args.algo, args.cfg, use_vis=args.viz)
        else:
            print(k, "not in all data")

    if args.mode == "stats":
        print("The number of sequences in evaluation:{0}".format(len(sr_res)))
        compute_metrics(sr_res, args.algo)
    elif args.mode == "vis":
        vis = ReliveVisulizer("humanoid_smpl_neutral_mesh_all_vis.xml")
        vis.show_animation()

    if debug:
        pickle.dump(debug_pred_dict, open(dest_pred_path, 'wb'))
        pickle.dump(debug_gt_dict, open(dest_gt_path, 'wb'))

