import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('./kinpoly')

import argparse
import os
from pathlib import Path
import yaml
import numpy as np
import joblib 
import pickle 
import json 

import imageio 

import torch
import torch.nn as nn
from torch.optim import AdamW

from mujoco_py import load_model_from_path

from collections import defaultdict

from egoego.eval.head_pose_metrics import compute_head_pose_metrics 

from egoego.vis.head_motion import vis_single_head_pose_traj, vis_multiple_head_pose_traj
from egoego.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file, run_blender_rendering_and_save2video_head_pose

from utils.data_utils.process_kinpoly_qpos2smpl import qpos2smpl_vis, qpos_to_smpl_data 

import torch.nn.functional as F
from tqdm import tqdm

from scipy.spatial.transform import Rotation as sRot

import pytorch3d.transforms as transforms 

from kinpoly.relive.utils import *
from kinpoly.relive.models.mlp import MLP
from kinpoly.relive.models.traj_ar_smpl_net import TrajARNet
from kinpoly.relive.data_loaders.statear_smpl_dataset import StateARDataset
from kinpoly.relive.utils.torch_humanoid import Humanoid
from kinpoly.relive.data_process.process_trajs import get_expert
from kinpoly.relive.utils.torch_ext import get_scheduler
from kinpoly.relive.utils.statear_smpl_config import Config

from kinpoly.scripts.eval_metrics_imu_rec import compute_metrics, compute_metrics_for_smpl 

from kinpoly.relive.data_process.convert_amass_ego_syn_to_qpos import get_head_vel, get_obj_relative_pose 

from kinpoly.copycat.smpllib.smpl_mujoco import smpl_to_qpose

from trainer_amass_cond_motion_diffusion import get_trainer  

from utils.data_utils.process_amass_dataset import determine_floor_height_and_contacts 

def test(opt, device):
   
    # Prepare full body ground truth data. 
    full_body_gt_data_path = "data/amass_same_shape_egoego_processed/test_amass_smplh_motion.p"
    full_body_gt_data = joblib.load(full_body_gt_data_path)

    # 'root_orient', 'body_pose', 'trans', 'beta', 'seq_name', 'gender', 
    # 'head_qpos', 'head_vels', 'global_head_trans', 'global_head_rot_6d', 
    # 'global_head_rot_6d_diff', 'global_head_trans_diff

    # Deine diffusion model 
    diffusion_trainer = get_trainer(opt)

    # milestone = "4"
    # diffusion_trainer.load(milestone)

    weight_root_folder = "./pretrained_models"
    diffusion_weight_path = os.path.join(weight_root_folder, "stage2_diffusion_4.pt")
    diffusion_trainer.load_weight_path(diffusion_weight_path)

    e_root_list = []
    o_root_list = []
    t_root_list = []
    e_head_list = []
    o_head_list = []
    t_head_list = []
    mpjpe_list = []
    mpjpe_wo_hand_list = []
    single_jpe_list = []

    pred_accl_list = []
    gt_accl_list = [] 
    accer_list = [] 
    pred_fs_list = [] 
    gt_fs_list = [] 
    
    TEST_DATASETS = ['Transitions_mocap', 'HumanEva'] # HuMoR test datasets
    # VAL_DATASETS = ['MPI_HDM05', 'SFU', 'MPI_mosh'] # HuMoR validation datasets

    selected_subset_name = None 

    max_len = 0 
    with torch.no_grad():
        for k in full_body_gt_data:
            test_input_data_dict = full_body_gt_data[k]

            seq_name = test_input_data_dict['seq_name'].replace(".npz", "")

            if not (TEST_DATASETS[0] in seq_name or TEST_DATASETS[1] in seq_name):
                continue 

            curr_seq_full_body_data = full_body_gt_data[k]

            if curr_seq_full_body_data['trans'].shape[0] > max_len:
                max_len = curr_seq_full_body_data['trans'].shape[0]
           
            max_steps = 120 
            gt_trans = curr_seq_full_body_data['trans'][:max_steps] # T X 3 
            gt_root_orient = curr_seq_full_body_data['root_orient'][:max_steps] # T X 3 
            gt_pose_aa = curr_seq_full_body_data['body_pose'][:max_steps] # T X 63 

            gt_trans = torch.from_numpy(gt_trans)
            gt_root_orient = torch.from_numpy(gt_root_orient)
            gt_pose_aa = torch.from_numpy(gt_pose_aa)

            curr_gt_smpl_seq_root_trans = gt_trans.float().cuda() 
            curr_gt_smpl_seq_joint_rot_aa = torch.cat((gt_root_orient, gt_pose_aa), dim=-1).reshape(-1, 22, 3).float().cuda()

            # fk to get global head pose 
            global_jrot, global_jpos = diffusion_trainer.ds.fk_smpl(curr_gt_smpl_seq_root_trans, \
            curr_gt_smpl_seq_joint_rot_aa)
            # T X 22 X 4, T X 22 X 3 

            floor_height, _, _ = determine_floor_height_and_contacts(global_jpos.data.cpu().numpy(), fps=30)
            # print("floor height:{0}".format(floor_height)) 
            global_jpos[:, :, 2] -= floor_height # Move the human to touch the floor z = 0  

            head_idx = 15 
            global_head_jpos = global_jpos[:, head_idx, :] # T X 3 
            global_head_jrot = global_jrot[:, head_idx, :] # T X 4 

            s1_output = defaultdict(list) 
            s1_output['head_pose'] = torch.cat((global_head_jpos, global_head_jrot), dim=-1)[None] #  1 X T X 7 

            num_try = 1
            e_root = None
            o_root = None
            t_root = None
            e_head = None
            o_head = None
            t_head = None
            mpjpe = None
            single_jpe = None

            best_root_jpos = None 
            best_local_aa_rep = None 
            best_head_jpos = None 
            for try_idx in range(num_try):
                sample_bs = 1
                rep_head_pose = s1_output['head_pose'].repeat(sample_bs, 1, 1) # BS X T X 7 

                ori_local_aa_rep, ori_global_root_jpos = \
                diffusion_trainer.full_body_gen_cond_head_pose_sliding_window(\
                rep_head_pose, seq_name) 

                # Get global joint positions using fk 
                pred_fk_jrot, pred_fk_jpos = diffusion_trainer.ds.fk_smpl(ori_global_root_jpos.reshape(-1, 3), \
                ori_local_aa_rep.reshape(-1, 22, 3))
                # (BS*T) X 22 X 4, (BS*T) X 22 X 3 
                pred_fk_jrot = pred_fk_jrot.reshape(sample_bs, -1, 22, 4) # BS X T X 22 X 4 
                pred_fk_jpos = pred_fk_jpos.reshape(sample_bs, -1, 22, 3) # BS X T X 22 X 3 

                gt_move_trans =  global_jpos[0:1, 15:16, :].clone()[None].repeat(sample_bs, 1, 1, 1) # BS X 1 X 1 X 3 
                pred_move_trans = pred_fk_jpos[:, 0:1, 15:16, :].clone()  # BS X 1 X 1 X 3 

                gt_move_trans[:, :, :, 2] *= 0
                pred_move_trans[:, :, :, 2] *= 0 

                rep_global_jpos = global_jpos[None].repeat(sample_bs, 1, 1, 1) - gt_move_trans # BS X T X 22 X 3 
                pred_fk_jpos = pred_fk_jpos - pred_move_trans # BS X T X 22 X 3 
               
                ori_global_root_jpos = pred_fk_jpos[:, :, 0, :].clone() # BS X T X 3 
                curr_gt_smpl_seq_root_trans = rep_global_jpos[:, :, 0, :].clone() # BS X T X 3 

                curr_metric_dict = None 
                curr_best_mpjpe = None 
                curr_best_global_root_pos = None 
                curr_best_local_aa_rep = None 
                curr_best_global_head_jpos = None 
                for s_idx in range(sample_bs):
                    # Process Predicted data to touch floor z = 0, and move thead init head translation xy to 0. 
                    pred_floor_height, _, _ = determine_floor_height_and_contacts(pred_fk_jpos[s_idx].data.cpu().numpy(), fps=30)
                    # print("pred floor height:{0}".format(pred_floor_height))   

                    metric_dict = compute_metrics_for_smpl(global_jrot[:pred_fk_jrot.shape[1]], \
                    rep_global_jpos[s_idx, :pred_fk_jpos.shape[1]], 0., \
                    pred_fk_jrot[s_idx], pred_fk_jpos[s_idx], pred_floor_height)
                
                    # e_root, o_root, t_root, e_head, o_head, t_head, mpjpe, single_jpe = compute_metrics(body_test_output, "statear", kinpoly_cfg)
                    curr_e_root = metric_dict['root_dist']
                    curr_o_root = metric_dict['root_rot_dist']
                    curr_t_root = metric_dict['root_trans_dist']
                    curr_e_head = metric_dict['head_dist']
                    curr_o_head = metric_dict['head_rot_dist']
                    curr_t_head = metric_dict['head_trans_dist']
                    curr_mpjpe = metric_dict['mpjpe']
                    curr_mpjpe_wo_hand = metric_dict['mpjpe_wo_hand']
                    curr_single_jpe = metric_dict['single_jpe']

                    curr_pred_accl = metric_dict['accel_pred']
                    curr_gt_accl = metric_dict['accel_gt'] 
                    curr_accer = metric_dict['accel_err']
                    curr_pred_fs = metric_dict['pred_fs']
                    curr_gt_fs = metric_dict['gt_fs']

                    print("Seq name:{0}".format(seq_name))
                    print("E_root: {0}, O_root: {1}, T_root: {2}".format(curr_e_root, curr_o_root, curr_t_root))
                    print("E_head: {0}, O_head: {1}, T_head: {2}".format(curr_e_head, curr_o_head, curr_t_head))
                    print("MPJPE: {0}".format(curr_mpjpe))
                    print("MPJPE wo Hand: {0}".format(curr_mpjpe_wo_hand))
                    print("ACCEL pred: {0}".format(curr_pred_accl))
                    print("ACCEL gt: {0}".format(curr_gt_accl))
                    print("ACCER: {0}".format(curr_accer))
                    print("Foot Sliding pred: {0}".format(curr_pred_fs))
                    print("Foot Sliding gt: {0}".format(curr_gt_fs))

                    curr_head_global_jpos = pred_fk_jpos[s_idx, :, 15, :] # T X 3 

                    if curr_best_mpjpe is None:
                        curr_best_mpjpe = curr_mpjpe 
                        curr_metric_dict = metric_dict
                        curr_best_global_root_pos = ori_global_root_jpos[s_idx]
                        curr_best_local_aa_rep = ori_local_aa_rep[s_idx]
                        curr_best_head_global_pos = curr_head_global_jpos 

                    if curr_mpjpe < curr_best_mpjpe:
                        curr_best_mpjpe = curr_mpjpe 
                        curr_metric_dict = metric_dict 
                        curr_best_global_root_pos = ori_global_root_jpos[s_idx]
                        curr_best_local_aa_rep = ori_local_aa_rep[s_idx]
                        curr_best_head_global_pos = curr_head_global_jpos 

                    # if opt.gen_vis: 
                    #     dest_vis_folder = os.path.join(opt.diffusion_save_dir, "stage2_vis_on_amass_test_diversity")
                    #     curr_seq_name = seq_name.replace(" ", "") + "_try_"+ str(s_idx)
                    #     diffusion_trainer.gen_full_body_vis(ori_global_root_jpos[s_idx], ori_local_aa_rep[s_idx], dest_vis_folder, curr_seq_name)

                if try_idx == 0 or curr_best_mpjpe < mpjpe:
                    e_root = curr_metric_dict['root_dist']
                    o_root = curr_metric_dict['root_rot_dist']
                    t_root = curr_metric_dict['root_trans_dist']
                    e_head = curr_metric_dict['head_dist']
                    o_head = curr_metric_dict['head_rot_dist']
                    t_head = curr_metric_dict['head_trans_dist']
                    mpjpe = curr_metric_dict['mpjpe']
                    mpjpe_wo_hand = curr_metric_dict['mpjpe_wo_hand']
                    single_jpe = curr_metric_dict['single_jpe']

                    pred_accl = curr_metric_dict['accel_pred']
                    gt_accl = curr_metric_dict['accel_gt'] 
                    accer = curr_metric_dict['accel_err']
                    pred_fs = curr_metric_dict['pred_fs']
                    gt_fs = curr_metric_dict['gt_fs']

                    best_root_jpos = curr_best_global_root_pos 
                    best_local_aa_rep = curr_best_local_aa_rep
                    best_head_jpos = curr_best_head_global_pos 

            # if mpjpe < 300:
            e_root_list.append(e_root)
            o_root_list.append(o_root)
            t_root_list.append(t_root)
            e_head_list.append(e_head)
            o_head_list.append(o_head)
            t_head_list.append(t_head)
            mpjpe_list.append(mpjpe)
            mpjpe_wo_hand_list.append(mpjpe_wo_hand)
            single_jpe_list.append(single_jpe)

            pred_accl_list.append(pred_accl)
            gt_accl_list.append(gt_accl) 
            accer_list.append(accer)
            pred_fs_list.append(pred_fs)
            gt_fs_list.append(gt_fs)

            # continue # Tmp 
            if opt.gen_vis:
                vis_head_pose = True    

                dest_vis_folder = os.path.join(opt.diffusion_save_dir, "stage2_vis_on_amass_test")

                curr_seq_name = seq_name.replace(" ", "")

                mesh_jnts, mesh_verts = diffusion_trainer.gen_full_body_vis(best_root_jpos, best_local_aa_rep, dest_vis_folder, curr_seq_name)
                # diffusion_trainer.gen_full_body_vis(curr_gt_smpl_seq_root_trans, curr_gt_smpl_seq_joint_rot_aa, dest_vis_folder, curr_seq_name, vis_gt=True)

                if vis_head_pose:
                    vis_head_v_idx = 444 

                    align_init_head_trans = mesh_verts[0, 0:1, vis_head_v_idx, :].detach().cpu().numpy() - s1_output['head_pose'][0, 0:1, :3].detach().cpu().numpy() # 1 X 3 
                    tmp_head_trans = s1_output['head_pose'][0, :, :3].detach().cpu().numpy() + align_init_head_trans # T X 3

                    dest_head_pose_npy_path = os.path.join(dest_vis_folder, curr_seq_name+"_head_pose.npy")
                    head_save_data = np.concatenate((tmp_head_trans, \
                    s1_output['head_pose'][0, :, 3:].detach().cpu().numpy()), axis=-1) # T X 7 
                    np.save(dest_head_pose_npy_path, head_save_data)

                    dest_obj_out_folder = os.path.join(dest_vis_folder, curr_seq_name, "objs")
                    dest_out_vid_path = os.path.join(dest_vis_folder, curr_seq_name+"_human_w_head_pose.mp4")
                    run_blender_rendering_and_save2video_head_pose(dest_head_pose_npy_path, dest_obj_out_folder, \
                    dest_out_vid_path)

                    dest_out_head_only_vid_path = os.path.join(dest_vis_folder, curr_seq_name+"_head_pose_only.mp4")
                    run_blender_rendering_and_save2video_head_pose(dest_head_pose_npy_path, dest_obj_out_folder, \
                    dest_out_head_only_vid_path, vis_head_only=True) 
                
    e_root_arr = np.asarray(e_root_list)
    o_root_arr = np.asarray(o_root_list)
    t_root_arr = np.asarray(t_root_list)
    e_head_arr = np.asarray(e_head_list)
    o_head_arr = np.asarray(o_head_list)
    t_head_arr = np.asarray(t_head_list)
    mpjpe_arr = np.asarray(mpjpe_list)
    mpjpe_wo_hand_arr = np.asarray(mpjpe_wo_hand_list)
    single_jpe_arr = np.asarray(single_jpe_list)

    pred_accl_arr = np.asarray(pred_accl_list)
    gt_accl_arr = np.asarray(gt_accl_list)
    accer_arr = np.asarray(accer_list)
    pred_fs_arr = np.asarray(pred_fs_list)
    gt_fs_arr = np.asarray(gt_fs_list)

    mean_e_root = e_root_arr.mean()
    mean_o_root = o_root_arr.mean() 
    mean_t_root = t_root_arr.mean() 
    mean_e_head = e_head_arr.mean()
    mean_o_head = o_head_arr.mean()
    mean_t_head = t_head_arr.mean()
    mean_mpjpe = mpjpe_arr.mean()
    mean_mpjpe_wo_hand = mpjpe_wo_hand_arr.mean()
    mean_single_jpe = single_jpe_arr.mean(axis=0) # J 

    mean_pred_accl = pred_accl_arr.mean()
    mean_gt_accl = gt_accl_arr.mean() 
    mean_accer = accer_arr.mean() 
    mean_pred_fs = pred_fs_arr.mean() 
    mean_gt_fs = gt_fs_arr.mean()

    print("****************Full Body Estimator Evaluation Metrics*******************")
    print("The number of sequences:{0}".format(e_root_arr.shape[0]))
    print("E_root: {0}, O_root: {1}, T_root: {2}".format(mean_e_root, mean_o_root, mean_t_root))
    print("E_head: {0}, O_head: {1}, T_head: {2}".format(mean_e_head, mean_o_head, mean_t_head))
    print("MPJPE: {0}".format(mean_mpjpe))
    print("MPJPE wo Hand: {0}".format(mean_mpjpe_wo_hand))
   
    print("ACCL pred: {0}".format(mean_pred_accl))
    print("ACCL gt: {0}".format(mean_gt_accl))
    print("ACCER: {0}".format(mean_accer))
    print("Foot Sliding pred: {0}".format(mean_pred_fs))
    print("Foot Sliding gt: {0}".format(mean_gt_fs))

    print("Max seq length:{0}".format(max_len))

    res_dict = {}
    res_dict['mean_o_root'] = mean_o_root 
    res_dict['mean_t_root'] = mean_t_root 
    res_dict['mean_o_head'] = mean_o_head 
    res_dict['mean_t_head'] = mean_t_head 
    res_dict['mpjpe'] = mean_mpjpe 
    res_dict['mean_mpjpe_wo_hand'] = mean_mpjpe_wo_hand 

    res_dict['accl_pred'] = mean_pred_accl 
    res_dict['accl_gt'] = mean_gt_accl
    res_dict['accer'] = mean_accer 
    res_dict['fs_pred'] = mean_pred_fs 
    res_dict['fs_gt'] = mean_gt_fs 
    
    dest_res_path = os.path.join(opt.diffusion_save_dir, "stage2_diffusion_model_res_on_amass_test.json")
    if selected_subset_name is not None:
        dest_res_path = dest_res_path.replace(".json", "_"+selected_subset_name+".json")

    json.dump(res_dict, open(dest_res_path, 'w'))

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=0, help='the number of workers for data loading')
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--weight', default='latest')

    parser.add_argument("--gen_vis", action="store_true")

    # For AvatarPoser config 
    parser.add_argument('--kinpoly_cfg', type=str, default="", help='Path to option JSON file.')

    # Diffusion model settings
    parser.add_argument('--diffusion_window', type=int, default=80, help='horizon')

    parser.add_argument('--diffusion_batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--diffusion_learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--diffusion_n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--diffusion_n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--diffusion_d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--diffusion_d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--diffusion_d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    parser.add_argument('--diffusion_project', default='runs/train', help='project/name')
    parser.add_argument('--diffusion_exp_name', default='', help='save to project/name')

    # For data representation
    parser.add_argument("--canonicalize_init_head", action="store_true")
    parser.add_argument("--use_min_max", action="store_true")

    parser.add_argument('--data_root_folder', default='', help='')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.diffusion_save_dir = str(Path(opt.diffusion_project) / opt.diffusion_exp_name)
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    test(opt, device)
    