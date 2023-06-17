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
import time 

import imageio 

import torch
import torch.nn as nn
from torch.optim import AdamW

from mujoco_py import load_model_from_path

# import pytorch3d.transforms as transforms 

from collections import defaultdict

from egoego.data.realworld_headpose_dataset import RealWorldHeadPoseDataset
from egoego.data.ares_headpose_dataset import ARESHeadPoseDataset
from egoego.data.gimo_headpose_dataset import GIMOHeadPoseDataset

from egoego.model.head_estimation_transformer import HeadFormer 
from egoego.model.head_normal_estimation_transformer import HeadNormalFormer

from egoego.eval.head_pose_metrics import compute_head_pose_metrics 

from egoego.vis.head_motion import vis_single_head_pose_traj, vis_multiple_head_pose_traj
from egoego.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file, run_blender_rendering_and_save2video_head_pose
from egoego.vis.head_motion import vis_multiple_frames_point_only, vis_single_frame_point_only, vis_multiple_2d_traj

from utils.data_utils.process_kinpoly_qpos2smpl import qpos2smpl_vis, qpos_to_smpl_data 
from utils.data_utils.process_amass_dataset import determine_floor_height_and_contacts 

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

def images_to_video_w_imageio(img_folder, of_files, output_vid_file, kinpoly_vis=True):
    # img_files = os.listdir(img_folder)
    # img_files.sort()
    if kinpoly_vis:
        img_files = []
        for of_name in of_files:
            img_files.append(of_name.split("/")[-1].replace(".npy", ".png"))
    else:
        img_files = of_files 

    im_arr = []
    for img_name in img_files:
        if kinpoly_vis:
            img_path = os.path.join(img_folder, img_name)
        else:
            img_path = img_name
        im = imageio.imread(img_path)
        im_arr.append(im)

    im_arr = np.asarray(im_arr)
    imageio.mimwrite(output_vid_file, im_arr, fps=30, quality=8) 

def get_ego_video(seq_name, of_files, out_vid_path, opt):
    if opt.test_on_gimo:
        img_files = []
        for of_name in of_files:
            img_path = of_name.replace("raft_of_feats", "segmented_ori_data").replace(".npy", ".png")
            img_path = "/".join(img_path.split("/")[:-1]) + "/egocentric_imgs/" + img_path.split("/")[-1]

            img_files.append(img_path)

        img_folder = ""
        images_to_video_w_imageio(img_folder, img_files, out_vid_path, kinpoly_vis=False)
    elif opt.test_on_ares:
        img_files = []
        for of_name in of_files:
            img_path = of_name.replace("/viscam/u/jiamanli/datasets/egomotion_syn_dataset", \
            "/viscam/projects/egoego/datasets/ares").replace("raft_flows", \
            "observations/head").replace(".npy", ".png")

            img_files.append(img_path)

        img_folder = ""
        images_to_video_w_imageio(img_folder, img_files, out_vid_path, kinpoly_vis=False)
    else:
        img_root_folder = "/move/u/jiamanli/datasets/kin-poly/ReliveDatasetRelease/fpv_frames"
        img_folder = os.path.join(img_root_folder, "-".join(seq_name.split("-")[1:]))

        images_to_video_w_imageio(img_folder, of_files, out_vid_path)

def test(opt, device):
    # Prepare Directories
    weight_root_folder = "./pretrained_models"

    if opt.test_on_ares:
        weight_path = os.path.join(weight_root_folder, "stage1_headnet_ares_250.pt")
    elif opt.test_on_gimo:
        weight_path = os.path.join(weight_root_folder, "stage1_headnet_gimo_1000.pt")
    else:
        weight_path = os.path.join(weight_root_folder, "stage1_headnet_kinpoly_1000.pt")

    ckpt = torch.load(weight_path, map_location=device)
    print(f"Loaded weight for head scale estimation: {weight_path}")

    # Load weight for GravityNet 
    normal_weight_path = os.path.join(weight_root_folder, "stage1_gravitynet_2000.pt")
    normal_ckpt = torch.load(normal_weight_path, map_location=device)
    print(f"Loaded weight for head normal estimation: {normal_weight_path}")

    data_root_folder = opt.data_root_folder 

    # Prepare head pose data loader.
    if opt.test_on_ares:
        val_dataset = ARESHeadPoseDataset(data_root_folder, train=False, window=opt.window, for_eval=True)
    elif opt.test_on_gimo:
        val_dataset = GIMOHeadPoseDataset(data_root_folder, train=False, window=opt.window, for_eval=True)
    else:
        val_dataset = RealWorldHeadPoseDataset(data_root_folder, train=False, window=opt.window, for_eval=True, \
        eval_on_kinpoly_mocap=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=opt.workers, pin_memory=True, drop_last=False)

    use_gt_head_pose = opt.use_gt_head_pose 

    # Prepare full body ground truth data. 
    if opt.eval_on_kinpoly_mocap:
        full_body_gt_data_path = os.path.join(data_root_folder, "kinpoly-mocap", "mocap_annotations.p")
        full_body_gt_data = joblib.load(full_body_gt_data_path)

        bad_seq_path = os.path.join(data_root_folder, "failed_seq_names", "kinpoly_bad_seq_names.pkl")

    if opt.test_on_ares:
        full_body_gt_data_path = os.path.join(data_root_folder, "ares_processed_for_kinpoly", "MoCapData", "features", "mocap_annotations.p")
        full_body_gt_data = joblib.load(full_body_gt_data_path)

        bad_seq_path = os.path.join(data_root_folder, "failed_seq_names", "ares_bad_seq_names.pkl")

    if opt.test_on_gimo:
        full_body_gt_data_path = os.path.join(data_root_folder, "gimo_processed_for_kinpoly", "MoCapData", "features", "mocap_annotations.p")
        full_body_gt_data = joblib.load(full_body_gt_data_path)

        bad_seq_path = os.path.join(data_root_folder, "failed_seq_names", "gimo_bad_seq_names.pkl")

    bad_seq_names = pickle.load(open(bad_seq_path, 'rb'))['bad_seq']

    # Define HeadFormer model. (For predicting scale)
    head_transformer_encoder = HeadFormer(opt, device)
    head_transformer_encoder.load_state_dict(ckpt['transformer_encoder_state_dict'])
    head_transformer_encoder.to(device)
    head_transformer_encoder.eval()

    # Define normal prediction model. (For rotating SLAM trajectory)
    head_normal_transformer = HeadNormalFormer(opt, device, eval_whole_pipeline=True)
    head_normal_transformer.load_state_dict(normal_ckpt['transformer_encoder_state_dict'])
    head_normal_transformer.to(device)
    head_normal_transformer.eval()

    # Load weight for diffusion model. 
    # Deine diffusion model 
    diffusion_trainer = get_trainer(opt)
    diffusion_weight_path = os.path.join(weight_root_folder, "stage2_diffusion_4.pt")
    diffusion_trainer.load_weight_path(diffusion_weight_path)

    s1_e_head_list = []
    s1_o_head_list = []
    s1_t_head_list = []

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
       
    TEST_SCENES = ['office_0', 'hotel_0', 'room_2', 'frl_apartment_4', 'apartment_0'] # ARES test scene 
    gimo_test_scenes = ["storeroom0217", "classroom0219", "lab0220", "kitchen0214"]

    with torch.no_grad():
        for test_it, test_input_data_dict in enumerate(val_loader):
            seq_name = test_input_data_dict['seq_name'][0]

            # Test on ARES testing scenes. 
            if opt.test_on_ares and seq_name.split("-")[0] not in TEST_SCENES:
                continue

            # Test on kinpoly-mocap sequences without stepping motion since in our model training, we assume the human is moving on the floor, no stairs. 
            if not (opt.test_on_ares or opt.test_on_gimo) and "step" in seq_name:
                continue 

            # Test on gimo testing scenes. 
            if opt.test_on_gimo and seq_name.split("-")[0] not in gimo_test_scenes:
                continue
           
            # We do not evaluate on sequences that SLAM failed. 
            if seq_name in bad_seq_names or seq_name+".npz" in bad_seq_names:
                continue 

            if opt.test_on_ares:
                curr_seq_full_body_data = full_body_gt_data[seq_name+".npz"]
            else:
                curr_seq_full_body_data = full_body_gt_data[seq_name]

            # Predict head pose first. 
            test_output = head_transformer_encoder.forward_for_eval(test_input_data_dict)

            if use_gt_head_pose:
                # For debug 
                s1_output = defaultdict(list) 
                s1_output['head_pose'] = torch.from_numpy(curr_seq_full_body_data['head_pose'])[None].to(device)  # 1 X T X (3+4) 
                s1_output['head_vels'] = torch.from_numpy(curr_seq_full_body_data['head_vels'])[None].to(device)  # 1 X T X (3+3)

                s1_output['qpos'] = torch.from_numpy(curr_seq_full_body_data['qpos'])[None].to(device)  # 1 X T X 76 (3+4+J*3) 
                s1_output['qvel'] = torch.from_numpy(curr_seq_full_body_data['qvel'])[None].to(device)  # 1 X T X 75 
                s1_output['obj_pose'] = torch.from_numpy(curr_seq_full_body_data['obj_pose'])[None].to(device) 
                s1_output['obj_head_relative_poses'] = torch.from_numpy(curr_seq_full_body_data['obj_head_relative_poses'])[None].to(device) 
            else:
                pred_scale = test_output['pred_scale']

                normal_input_dict = defaultdict(list) 
                
                normal_input_head_trans = test_input_data_dict['ori_slam_trans'].to(device)
                normal_input_dict['head_trans'] = normal_input_head_trans - normal_input_head_trans[:, 0:1, :]
                normal_input_dict['head_rot_mat'] = test_input_data_dict['ori_slam_rot_mat'].to(device)
                
                normal_input_dict['ori_head_pose'] = torch.from_numpy(curr_seq_full_body_data['head_pose'])[None].to(device)


                normal_input_dict['seq_len']= torch.tensor(normal_input_head_trans.shape[1]).float()[None] 
                test_normal_output = head_normal_transformer.forward_for_eval(normal_input_dict, pred_scale)
            
                s1_output = defaultdict(list) 

                s1_output['qpos'] = torch.from_numpy(curr_seq_full_body_data['qpos'])[None].to(device)  # 1 X T X 76 (3+4+J*3) 
                s1_output['qvel'] = torch.from_numpy(curr_seq_full_body_data['qvel'])[None].to(device)  # 1 X T X 75 
                s1_output['obj_pose'] = torch.from_numpy(curr_seq_full_body_data['obj_pose'])[None].to(device) 
                
                head_pose = test_normal_output['head_pose'] 
                
                head_pose = torch.cat((head_pose[:, :, :3].to(device), \
                test_output['head_pose'][:, :, 3:]), dim=-1)

                head_vels = get_head_vel(head_pose[0].data.cpu().numpy())
                obj_pose = s1_output['obj_pose']
                if obj_pose.shape[1] != head_pose.shape[1]:
                    head_pose = head_pose[:, :obj_pose.shape[1]]
                    head_vels = head_vels[:obj_pose.shape[1]]

                s1_output['head_pose'] = head_pose.double() 

                obj_relative_head = get_obj_relative_pose(obj_pose[0].data.cpu().numpy(), head_pose[0].data.cpu().numpy(), num_objs=1)
            
                s1_output['head_vels'] = torch.from_numpy(head_vels)[None].to(device)
                s1_output['obj_head_relative_poses'] = torch.from_numpy(obj_relative_head)[None].to(device)

            # Compute the stage 1 head pose estimation metric
            s1_output['head_pose'][0, :, :2] -= s1_output['head_pose'][0, 0:1, :2].clone()
            
            s1_head_pred_trans = s1_output['head_pose'][0, :, :3].data.cpu().numpy()
            s1_head_pred_quat = s1_output['head_pose'][0, :, 3:]
            s1_head_pred_rot_mat = transforms.quaternion_to_matrix(s1_head_pred_quat).data.cpu().numpy()

            s1_head_gt_trans = curr_seq_full_body_data['head_pose'][:, :3].copy()
            s1_head_gt_trans[:, :2] -= s1_head_gt_trans[0:1, :2]
            s1_head_gt_quat = torch.from_numpy(curr_seq_full_body_data['head_pose'][:, 3:])
            s1_head_gt_rot_mat = transforms.quaternion_to_matrix(s1_head_gt_quat).data.cpu().numpy()
            if s1_head_gt_trans.shape[0] != s1_head_pred_trans.shape[0]:
                s1_head_gt_trans = s1_head_gt_trans[:s1_head_pred_trans.shape[0]]
                s1_head_gt_rot_mat = s1_head_gt_rot_mat[:s1_head_pred_rot_mat.shape[0]]

            s1_head_dist, s1_head_dist_rot_only, s1_head_trans_error = compute_head_pose_metrics(s1_head_pred_trans, s1_head_pred_rot_mat, \
                                                                s1_head_gt_trans, s1_head_gt_rot_mat) 
            s1_e_head_list.append(s1_head_dist)
            s1_o_head_list.append(s1_head_dist_rot_only) 
            s1_t_head_list.append(s1_head_trans_error) 
    
            print("*********************Single sequence Head Pose Evaluation********************")
            print("Stage 1 4 X 4 rot matrix dist:{0}".format(s1_head_dist))
            print("Stage 1 3 X 3 rot matrix dist:{0}".format(s1_head_dist_rot_only))
            print("Stage 1 Head trans err(mm):{0}".format(s1_head_trans_error))
          
            # Convert qpos to smpl 
            curr_gt_smpl_seq_root_trans, curr_gt_smpl_seq_joint_rot_aa = \
            qpos_to_smpl_data(torch.from_numpy(curr_seq_full_body_data['qpos']).float())
            # T X 3, T X 72 

            curr_gt_smpl_seq_joint_rot_aa = curr_gt_smpl_seq_joint_rot_aa.reshape(-1, 24, 3)[:, :22, :].cuda() # T X 22 X 3

            # fk to get global head pose 
            global_jrot, global_jpos = diffusion_trainer.ds.fk_smpl(curr_gt_smpl_seq_root_trans, \
            curr_gt_smpl_seq_joint_rot_aa)
            # T X 22 X 4, T X 22 X 3 

            floor_height, _, _ = determine_floor_height_and_contacts(global_jpos.data.cpu().numpy(), fps=30)
            global_jpos[:, :, 2] -= floor_height # Move the human to touch the floor z = 0  

            move_to_floor_trans = global_jpos[0:1, 15, :].clone() - s1_output['head_pose'][0, 0:1, :3]
            s1_output['head_pose'][0, :, :3] += move_to_floor_trans 

            # Update 
            if use_gt_head_pose:
                head_idx = 15 
                global_head_jpos = global_jpos[:, head_idx, :] # T X 3 
                global_head_jrot = global_jrot[:, head_idx, :] # T X 4 

                s1_output['head_pose'] = torch.cat((global_head_jpos, global_head_jrot), dim=-1)[None] #  1 X T X 7 

            num_try = 1 # 3
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
                curr_seq_len = curr_seq_full_body_data['head_pose'].shape[0]

                sample_bs = 1 # 100
                rep_head_pose = s1_output['head_pose'].repeat(sample_bs, 1, 1) # BS X T X 7 

                ori_local_aa_rep, ori_global_root_jpos = \
                diffusion_trainer.full_body_gen_cond_head_pose_sliding_window(\
                rep_head_pose, seq_name) 
                # BS X T X 22 X 3, BS X T X 3 
                
                # Get global joint positions using fk 
                pred_fk_jrot, pred_fk_jpos = diffusion_trainer.ds.fk_smpl(ori_global_root_jpos.reshape(-1, 3), \
                ori_local_aa_rep.reshape(-1, 22, 3))
                # (BS*T) X 22 X 4, (BS*T) X 22 X 3 
                pred_fk_jrot = pred_fk_jrot.reshape(sample_bs, -1, 22, 4) # BS X T X 22 X 4 
                pred_fk_jpos = pred_fk_jpos.reshape(sample_bs, -1, 22, 3) # BS X T X 22 X 3 

                gt_move_trans = global_jpos[0:1, 15:16, :].clone()[None].repeat(sample_bs, 1, 1, 1) # 1 X 1 X 3 
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
                curr_best_head_global_pos = None 
                for s_idx in range(sample_bs):
                    # Process Predicted data to touch floor z = 0, and move thead init head translation xy to 0. 
                    pred_floor_height, _, _ = determine_floor_height_and_contacts(pred_fk_jpos[s_idx].data.cpu().numpy(), fps=30)
                    # print("pred floor height:{0}".format(pred_floor_height)) 

                    metric_dict = compute_metrics_for_smpl(global_jrot[:pred_fk_jrot.shape[1]], \
                    rep_global_jpos[s_idx, :pred_fk_jpos.shape[1]], 0., \
                    pred_fk_jrot[s_idx], pred_fk_jpos[s_idx], pred_floor_height)

                    ori_global_root_jpos[s_idx, :, 2] -= pred_floor_height 
                
                    curr_head_global_jpos = pred_fk_jpos[s_idx, :, 15, :] # T X 3 

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

            if opt.gen_vis:
                vis_head_pose = False  

                seq_name = test_input_data_dict['seq_name'][0]

                if opt.eval_on_kinpoly_mocap:
                    dest_vis_folder = os.path.join(opt.diffusion_save_dir, "egoego_vis_on_kinpoly_mocap")
                
                if opt.test_on_gimo:
                    dest_vis_folder = os.path.join(opt.diffusion_save_dir, "egoego_vis_on_gimo")

                if opt.test_on_ares:
                    dest_vis_folder = os.path.join(opt.diffusion_save_dir, "egoego_vis_on_ares")

                if use_gt_head_pose:
                    dest_vis_folder = dest_vis_folder + "_use_gt_head"

                if not os.path.exists(dest_vis_folder):
                    os.makedirs(dest_vis_folder)

                vis_tag = "egoego"
               
                # Visualize corresponding egocentric video 
                of_files = curr_seq_full_body_data['of_files']
                curr_seq_name = seq_name.replace(" ", "")
                out_vid_path = os.path.join(dest_vis_folder, curr_seq_name+"_egocentric_view.mp4")
                get_ego_video(curr_seq_name, of_files, out_vid_path, opt)

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

    s1_e_head_arr = np.asarray(s1_e_head_list)
    s1_o_head_arr = np.asarray(s1_o_head_list)
    s1_t_head_arr = np.asarray(s1_t_head_list)

    mean_s1_e_head = s1_e_head_arr.mean()
    mean_s1_o_head = s1_o_head_arr.mean()
    mean_s1_t_head = s1_t_head_arr.mean()

    print("****************Full Head Estimator Evaluation Metrics*******************")
    print("Stage 1, E_head: {0}, O_head: {1}, T_head: {2}".format(mean_s1_e_head, mean_s1_o_head, mean_s1_t_head))

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

    if opt.eval_on_kinpoly_mocap:
        dest_res_path = os.path.join(opt.diffusion_save_dir, "diffusion_model_res_on_kinpoly_mocap.json")
                
    if opt.test_on_gimo:
        dest_res_path = os.path.join(opt.diffusion_save_dir, "diffusion_model_res_on_gimo.json")

    if opt.test_on_ares:
        dest_res_path = os.path.join(opt.diffusion_save_dir, "diffusion_model_res_on_ares.json")

    if use_gt_head_pose:
        dest_res_path = dest_res_path.replace(".json", "_use_gt_head.json")

    curr_time = time.time()
    dest_res_path = dest_res_path.replace(".json", str(curr_time)+".json")

    json.dump(res_dict, open(dest_res_path, 'w'))

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kinpoly_cfg', type=str, default="", help='Path to option JSON file.')

    parser.add_argument("--test_on_ares", action="store_true")
    parser.add_argument("--test_on_gimo", action="store_true")
    parser.add_argument("--eval_on_kinpoly_mocap", action="store_true")

    parser.add_argument("--gen_vis", action="store_true")

    parser.add_argument("--use_gt_head_pose", action="store_true")

    parser.add_argument('--data_root_folder', default='', help='folder to data')
  
    # HeadNet settings. 
    parser.add_argument('--project', default='', help='project/name')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')

    parser.add_argument('--workers', type=int, default=0, help='the number of workers for data loading')
    parser.add_argument('--device', default='0', help='cuda device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--window', type=int, default=90, help='horizon')
    parser.add_argument('--n_dec_layers', type=int, default=2, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=256, help='the dimension of intermediate representation in transformer')

    parser.add_argument('--weight', default='latest')

    parser.add_argument('--dist_scale', type=float, default=10.0, help='scale for prediction of distance scalar')

    parser.add_argument("--freeze_of_cnn", action="store_true")
    parser.add_argument("--input_of_feats", action="store_true")    

    # GravityNet settings. 
    parser.add_argument('--normal_project', default='', help='project/name')
    parser.add_argument('--normal_exp_name', default='exp', help='save to project/name')

    parser.add_argument('--normal_window', type=int, default=90, help='horizon')
    parser.add_argument('--normal_n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--normal_n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--normal_d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--normal_d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--normal_d_model', type=int, default=256, help='the dimension of intermediate representation in transformer')

    parser.add_argument('--normal_weight', default='latest')

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

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(Path(opt.project) / opt.exp_name) # For head pose estimation model 
    opt.normal_save_dir = str(Path(opt.normal_project) / opt.normal_exp_name)
    opt.diffusion_save_dir = str(Path(opt.diffusion_project) / opt.diffusion_exp_name)
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    test(opt, device)
    