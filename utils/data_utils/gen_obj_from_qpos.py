import sys, os
from turtle import degrees
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '../..'))

import glob
import os
import argparse
import time
import json 
import pickle as pkl 
import joblib 

import random 

import numpy as np
import torch

import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN

from smplpytorch.pytorch.smpl_layer import SMPL_Layer

# from body_model.body_model import BodyModel
# from body_model.utils import SMPL_JOINTS, axisangle2matrots
# from utils.torch import copy2cpu as c2c
# from utils.transforms import batch_rodrigues, compute_world2aligned_mat, compute_world2aligned_joints_mat


import TransPose.articulate as art

from transformation import quaternion_to_angle_axis, matrot2axisangle, quat2mat
from scipy.spatial.transform import Rotation as sRot

# from TIP.process_kinpoly_data import lerp_root_trajectory, slerp_rotation 

# if sequence is longer than this, splits into sequences of this size to avoid running out of memory
# ~ 4000 for 12 GB GPU, ~2000 for 8 GB
SPLIT_FRAME_LIMIT = 2000

NUM_BETAS = 10 # size of SMPL shape parameter to use

def get_body_model_sequence(smpl_path, gender, num_frames,
                  pose_body, betas, root_orient, trans):
    gender = str(gender)
    bm_path = os.path.join(smpl_path, 'SMPL_NEUTRAL.pkl')
    # SMPLH model data.files ['J_regressor_prior', 'f', 'J_regressor', 'kintree_table', 'J', 'weights_prior', 'weights', 'posedirs', 'bs_style', 'v_template', 'shapedirs', 'bs_type']
    # SMPL model data.keys() dict_keys(['J_regressor_prior', 'f', 'J_regressor', 'kintree_table', 'J', 'weights_prior', 'weights', 'posedirs', 'pose_training_info', 'bs_style', 'v_template', 'shapedirs', 'bs_type'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm = BodyModel(bm_path=bm_path, num_betas=NUM_BETAS, batch_size=num_frames).to(device)

    pose_body = torch.Tensor(pose_body).to(device)
    betas = torch.Tensor(np.repeat(betas[:NUM_BETAS][np.newaxis], num_frames, axis=0)).to(device)
    root_orient = torch.Tensor(root_orient).to(device)
    trans = torch.Tensor(trans).to(device)
    body = bm(pose_body=pose_body, betas=betas, root_orient=root_orient, trans=trans)
    return body

def write_to_obj(dest_obj_file, vertices, faces):
    # vertices: N X 3, faces: N X 3
    w_f = open(dest_obj_file, 'w')

    print("Total vertices:{0}".format(vertices.shape[0]))

    # Write vertices to file 
    for idx in range(vertices.shape[0]):
        w_f.write("v "+str(vertices[idx, 0])+" "+str(vertices[idx, 1])+" "+str(vertices[idx, 2])+"\n")

    # Write faces to file 
    for idx in range(faces.shape[0]):
        w_f.write("f "+str(faces[idx, 0]+1)+" "+str(faces[idx, 1]+1)+" "+str(faces[idx, 2]+1)+"\n")

    w_f.close() 

def gen_smpl_objs():
    # seq_name_list = ["apartment_1-Transitions_mocap_mazen_c3d_crouchwalk_turntwist180_poses_235_frames_30_fps_b890seq0_samp_0.npz", \
    # "apartment_2-MPI_HDM05_dg_HDM_dg_02-02_04_120_poses_1091_frames_30_fps_b796seq0_samp_0.npz", \
    # "frl_apartment_0-Transitions_mocap_mazen_c3d_twistdance_walk_poses_196_frames_30_fps_b842seq0_samp_0.npz", \
    # "frl_apartment_1-Transitions_mocap_mazen_c3d_walksideways_walk_poses_204_frames_30_fps_b946seq0_samp_0.npz", \
    # "frl_apartment_3-Transitions_mocap_mazen_c3d_crouchwalk_runbackwards_poses_277_frames_30_fps_b872seq0_samp_5.npz", \
    # "frl_apartment_3-Transitions_mocap_mazen_c3d_turntwist_walk_poses_218_frames_30_fps_b873seq0_samp_0.npz", \
    # "room_1-Transitions_mocap_mazen_c3d_JOOF_walk_poses_191_frames_30_fps_b802seq0_samp_6.npz", \
    # "room_2-Transitions_mocap_mazen_c3d_jumpingtwist_walk_poses_231_frames_30_fps_b797seq0_samp_2.npz"]
    
    # Random sample some sequences 
    tmp_data_path = "/viscam/u/jiamanli/results/kinpoly/all/statear/baseline_posereg_of_only_on_syn_amass_v3/results/iter_0100_test_mocap_annotations.p"
    tmp_data = joblib.load(tmp_data_path) 
    k_list = list(tmp_data.keys()) 
    seq_k_pkl_path = "./sampled_seq_names.pkl"
    if os.path.exists(seq_k_pkl_path):
        tmp_dict = pkl.load(open(seq_k_pkl_path, 'rb'))
        seq_name_list = tmp_dict['sampled_seq_names'] 
    else:
        tmp_dict = {}
        seq_name_list = random.sample(k_list, 100) 
        tmp_dict['sampled_seq_names'] = seq_name_list 
        pkl.dump(tmp_dict, open(seq_k_pkl_path, 'wb'))

    # Tmp use!!!
    seq_name_list = ["frl_apartment_3-Transitions_mocap_mazen_c3d_crouchwalk_runbackwards_poses_277_frames_30_fps_b872seq0_samp_5.npz", \
    "room_1-Transitions_mocap_mazen_c3d_JOOF_walk_poses_191_frames_30_fps_b802seq0_samp_6.npz", \
    "room_2-Transitions_mocap_mazen_c3d_jumpingtwist_walk_poses_231_frames_30_fps_b797seq0_samp_2.npz"]

    # data_path = "/viscam/u/jiamanli/results/kinpoly/all/statear/our_combined_headformer_rot_and_scale_eval_on_syn/results/iter_1600_test_mocap_annotations.p"
    # data_path = "/viscam/u/jiamanli/results/kinpoly/all/statear/baseline_kinpoly_of_only_on_syn_amass_v1/results/iter_1200_test_mocap_annotations.p"
    # data_path = "/viscam/u/jiamanli/results/kinpoly/all/statear/our_droid_slam_kinpoly_amass_eval/results/iter_1600_test_mocap_annotations.p"
    data_path = "/viscam/u/jiamanli/results/kinpoly/all/statear/baseline_posereg_of_only_on_syn_amass_v3/results/iter_0100_test_mocap_annotations.p"
    all_data = joblib.load(data_path)

    # obj_folder = "/viscam/u/jiamanli/vis_res_for_paper/our_egoego_hybrid_objs"
    # obj_folder = "/viscam/u/jiamanli/vis_res_for_paper/baseline_kinpoly_of_only_objs"
    # obj_folder = "/viscam/u/jiamanli/vis_res_for_paper/our_egoego_slam_objs"
    # obj_folder = "/viscam/u/jiamanli/vis_res_for_paper/gt_objs"
    obj_folder = "/viscam/u/jiamanli/vis_res_for_paper/baseline_posereg_objs"

    # smpl_path = "/viscam/u/jiamanli/github/egomotion/body_models/smpl"
    MODEL_PATH = "/viscam/u/jiamanli/github/smpl/models"
    # SMPL model joints
    # 0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee
    # 5: R_Knee, 6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3
    # 10: L_Foot, 11: R_Foot, 12: Neck, 13: L_Collar, 14: R_Collar
    # 15: Head, 16: L_Shoulder, 17: R_Shoulder, 18: L_Elbow, 19: R_Elbow
    # 20: L_Wrist, 21: R_Wrist, 22(25): L_Index1, 23(40): R_Index1

    # smpl_xml_joint_idx = np.array([0,1,4,7,10,2,5,8,11,3,6,  9,12,15,13,16,18,20,22,14,17,19,21,23]) # From SMPL to mujoco joints
    #                                0,1,2,3,4, 5,6,7,8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 
    # 0: Pelvis, 1: L_Hip, 2: L_Knee, 3: L_Ankle, 4: L_Foot, 5: R_Hip, 6: R_Knee, 
    # 7: R_Ankle, 8: R_foot, 9: Spine1, 10: Spine2, 11: Spine3, 12: Neck, 13: Head, 
    # 14: L_Collar, 15: L_Shoulder, 16: L_Elbow, 17: L_Wrist, 18: L_Index1, 
    # 19: R_Collar, 20: R_Shoulder, 21: R_Elbow, 22: R_Wrist, 23: R_Index1.  

    # mujoco joints
    # 0: Pelvis 1: L_Hip 2: L_foot 3: R_ankle 4:Spine2  5: L_Knee  6: R_Hip 7: R_foot   8: Spine 3  9: L_ankle  10: R_knee 11:Spine1
    # 12: Neck   13:  L_shoulder 14: Head  15: L_elbow   16: L_index1   17: R_shoulder 18:R_wrist  19: L_collar     
    # 20: L_wrist  21: R_collar   22: R_elbow  23:  R_index1 
    
    mujoco2smpl_joint_idx = np.asarray([0,1,5,9,2,6,10,3,7,11,4,8,12,14,19,13,15,20,16,21,17,22,18,23]) # From mujoco joints to SMPL
    # wrong: mujoco2smpl_joint_idx = np.asarray([0,1,5,9,2,6,10,3,7,11,4,8,12,19,14,13,20,15,21,16,22,17,23,18]) # From mujoco joints to SMPL

    k_list = list(all_data.keys()) 
    new_k_list = []
    for seq_name in k_list:
        if seq_name in seq_name_list:
            new_k_list.append(seq_name)
    for k_name in new_k_list: 

        seq_data = all_data[k_name]['qpos']
        # seq_data = all_data[k_name]['qpos_gt']
        # quat_data = all_data[k_name]['bquat'] # T X (24*4)

        root_trans = seq_data[:, :3] # T X 3  
        root_quat = seq_data[:, 3:7] # T X 4
        joint_euler = seq_data[:, 7:] # T X (23*3) 

        poses = torch.from_numpy(joint_euler).float() # T X (23*3)
        trans = torch.from_numpy(root_trans).float() # T X 3 
        num_frames = poses.shape[0] 
        rep_betas = torch.zeros(num_frames, 10).float() # T X 10 

        # curr_pose_mat = angle_axis_to_rotation_matrix(pose.reshape(-1, 3)).reshape(pose.shape[0], -1, 4, 4)
        # curr_spose = sRot.from_matrix(curr_pose_mat[:,:,:3,:3].reshape(-1, 3, 3).numpy())
        # curr_spose_euler = curr_spose.as_euler(euler_order, degrees=False).reshape(curr_pose_mat.shape[0], -1)
        # curr_spose_euler = curr_spose_euler.reshape(-1, 24, 3)[:, smpl_2_mujoco, :].reshape(-1, 72)
        # root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:,0,:3,:])
        # curr_qpos = np.concatenate((trans, root_quat, curr_spose_euler[:,3:]), axis = 1)

        
        root_aa = quaternion_to_angle_axis(torch.from_numpy(root_quat).float()) # T X 3 

        # Convert euler angle to axis angle representation.
        poses = poses.view(-1, 3) # (T*23) X 3  
        r = sRot.from_euler('ZYX', poses.data.cpu().numpy(), degrees=False) # zyx is different from ZYX!!
        rot_mats = r.as_matrix() # (T*23) X 3 X 3 
        joint_aa = matrot2axisangle(rot_mats) # (T*23) X 1 X 3 
        joint_aa = torch.from_numpy(joint_aa).float().squeeze(1) # (T*23) X 3 
        joint_aa = joint_aa.view(-1, 23, 3) # T X 23 X 3 

        aa_rep = torch.cat((root_aa[:, None, :], joint_aa), dim=1) # T X 24 X 3 
        aa_rep = aa_rep.data.cpu().numpy() 

        new_poses = aa_rep[:, mujoco2smpl_joint_idx] # T X 24 x 3 
        new_poses = torch.from_numpy(new_poses).float().reshape(-1, 72) # T X 72 

        new_poses = new_poses.view(-1, 24, 3) # T X 24 X 3

        # align AMASS global fame with DIP
        # amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]]).to(trans.device)
        # trans = amass_rot.matmul(trans.unsqueeze(-1)).view_as(trans) # T X 3 
        # new_poses[:, 0] = art.math.rotation_matrix_to_axis_angle(
        #     amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(new_poses[:, 0])))

        new_poses = new_poses.view(-1, 72) # T X 72

        # Make the first frame's trans to 0 
        init_trans = trans[0:1, :].clone() # 1 X 3 
        trans -= init_trans 

        # Define SMPL layer
        smpl_layer = SMPL_Layer(
            center_idx=0,
            gender="male",
            model_root=MODEL_PATH)
    
        faces = smpl_layer.th_faces
        smpl_layer.cuda()

        # SMPL forward to generate joints and vertices 
        body_vtx_seq = []
        process_inds = [0, min([num_frames, SPLIT_FRAME_LIMIT])]
        while process_inds[0] < num_frames:
            print(process_inds)
            sidx, eidx = process_inds
            # body = get_body_model_sequence(smpl_path, gender, process_inds[1] - process_inds[0],
            #                     poses[sidx:eidx, 3:], betas, poses[sidx:eidx, :3], trans[sidx:eidx])


            smpl_verts, _, orientations = smpl_layer(
                th_pose_axisang=new_poses[sidx:eidx].cuda(),
                th_betas=rep_betas[sidx:eidx].cuda()
                )

            rep_trans = trans[sidx:eidx, :].unsqueeze(1).repeat(1, 6890, 1)
            smpl_verts = smpl_verts + rep_trans.cuda()
 
            body_vtx_seq.append(smpl_verts.data.cpu().numpy()) 

            process_inds[0] = process_inds[1]
            process_inds[1] = min([num_frames, process_inds[1] + SPLIT_FRAME_LIMIT])

        vtx_seq = np.concatenate(body_vtx_seq, axis=0)
        print(vtx_seq.shape)

        # Save to .obj
        dest_obj_folder = os.path.join(obj_folder, k_name)
        if not os.path.exists(dest_obj_folder):
            os.makedirs(dest_obj_folder) 
        for t_idx in range(num_frames):
            dest_obj_path = os.path.join(dest_obj_folder, ('%05d'%t_idx)+".obj")
            write_to_obj(dest_obj_path, vtx_seq[t_idx], faces.data.cpu().numpy())

            # break 
        
        # break 


if __name__ == "__main__":
    gen_smpl_objs()  

