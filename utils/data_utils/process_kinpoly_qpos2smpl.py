import sys, os
from turtle import degrees
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '../..'))
sys.path.append("./")

import glob
import os
import argparse
import time
import json 
import pickle as pkl 
import joblib 
import numpy as np
import yaml 
import trimesh 

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

from scipy.spatial.transform import Rotation as sRot

import torch
import pytorch3d.transforms as transforms 

from egoego.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file

from body_model.body_model import BodyModel
from body_model.utils import SMPL_JOINTS

from utils.data_utils.process_amass_dataset import local2global_pose, get_head_vel  

from utils.data_utils.transformation import quaternion_to_angle_axis, matrot2axisangle, quat2mat

# if sequence is longer than this, splits into sequences of this size to avoid running out of memory
# ~ 4000 for 12 GB GPU, ~2000 for 8 GB
SPLIT_FRAME_LIMIT = 2000

NUM_BETAS = 10 # size of SMPL shape parameter to use

# for determining floor height
FLOOR_VEL_THRESH = 0.005
FLOOR_HEIGHT_OFFSET = 0.01
# for determining contacts
CONTACT_VEL_THRESH = 0.005 #0.015
CONTACT_TOE_HEIGHT_THRESH = 0.04
CONTACT_ANKLE_HEIGHT_THRESH = 0.08
# for determining terrain interaction
TERRAIN_HEIGHT_THRESH = 0.04 # if static toe is above this height
ROOT_HEIGHT_THRESH = 0.04 # if maximum "static" root height is more than this + root_floor_height
CLUSTER_SIZE_THRESH = 0.25 # if cluster has more than this faction of fps (30 for 120 fps)

#
# Processing
#
def qpos_to_smpl_data(seq_data):
    # seq_data: T X 76 

    # poses: T X 23 X 3 torch tensor 
    # root_aa: T X 4 torch tensor 
    # trans: T X 3 torch tensor 

    mujoco2smpl_joint_idx = np.asarray([0,1,5,9,2,6,10,3,7,11,4,8,12,14,19,13,15,20,16,21,17,22,18,23])
    smplh_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh_amass" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_data = seq_data.to(device)
    
    root_quat = seq_data[:, 3:7] # T X 4 
    poses = seq_data[:, 7:].reshape(-1, 23, 3) # T X 23 X 3 
    trans = seq_data[:, :3] # T X 3 

    root_aa = quaternion_to_angle_axis(root_quat.float()) # T X 3 

    # Convert euler angle to axis angle representation.
    poses = poses.reshape(-1, 3) # (T*23) X 3  
    r = sRot.from_euler('ZYX', poses.data.cpu().numpy(), degrees=False) # zyx is different from ZYX!!
    rot_mats = r.as_matrix() # (T*23) X 3 X 3 
    joint_aa = matrot2axisangle(rot_mats) # (T*23) X 1 X 3 
    joint_aa = torch.from_numpy(joint_aa).float().squeeze(1).to(device) # (T*23) X 3 
    joint_aa = joint_aa.reshape(-1, 23, 3) # T X 23 X 3 

    aa_rep = torch.cat((root_aa[:, None, :].to(device), joint_aa), dim=1) # T X 24 X 3 
    aa_rep = aa_rep.data.cpu().numpy() 

    new_poses = aa_rep[:, mujoco2smpl_joint_idx] # T X 24 x 3 
    new_poses = torch.from_numpy(new_poses).float().reshape(-1, 72) # T X 72 
    new_poses = new_poses.view(-1, 24, 3) # T X 24 X 3
    new_poses = new_poses.view(-1, 72) # T X 72, axis angle representation 

    return trans, new_poses  
    # T X 3, T X 72 

def qpos2smpl_vis(poses, root_quat, trans, vis_res_folder, k_name, vis_gt=False):
    # poses: T X 23 X 3 torch tensor 
    # root_aa: T X 4 torch tensor 
    # trans: T X 3 torch tensor 

    mujoco2smpl_joint_idx = np.asarray([0,1,5,9,2,6,10,3,7,11,4,8,12,14,19,13,15,20,16,21,17,22,18,23])
    smplh_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh_amass" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root_aa = quaternion_to_angle_axis(root_quat.float()) # T X 3 

    # Convert euler angle to axis angle representation.
    poses = poses.reshape(-1, 3) # (T*23) X 3  
    r = sRot.from_euler('ZYX', poses.data.cpu().numpy(), degrees=False) # zyx is different from ZYX!!
    rot_mats = r.as_matrix() # (T*23) X 3 X 3 
    joint_aa = matrot2axisangle(rot_mats) # (T*23) X 1 X 3 
    joint_aa = torch.from_numpy(joint_aa).float().squeeze(1).to(device) # (T*23) X 3 
    joint_aa = joint_aa.reshape(-1, 23, 3) # T X 23 X 3 

    aa_rep = torch.cat((root_aa[:, None, :], joint_aa), dim=1) # T X 24 X 3 
    aa_rep = aa_rep.data.cpu().numpy() 

    new_poses = aa_rep[:, mujoco2smpl_joint_idx] # T X 24 x 3 
    new_poses = torch.from_numpy(new_poses).float().reshape(-1, 72) # T X 72 
    new_poses = new_poses.view(-1, 24, 3) # T X 24 X 3
    new_poses = new_poses.view(-1, 72) # T X 72, axis angle representation 

    root_orient = new_poses[:, :3].data.cpu().numpy()
    pose_body = new_poses[:, 3:-6].data.cpu().numpy() 

    num_frames = new_poses.shape[0]

    # SMPL forward to generate joints and vertices 
    body_jnt_seq = [] 
    body_vtx_seq = []
    process_inds = [0, min([num_frames, SPLIT_FRAME_LIMIT])]

    gender = "male"
    use_smplh = True 
    betas = np.zeros(10) 
    while process_inds[0] < num_frames:
        print(process_inds)
        sidx, eidx = process_inds
        
        if use_smplh:
            body, kintree_table = get_body_model_sequence(smplh_path, gender, process_inds[1] - process_inds[0], \
                        new_poses[sidx:eidx, 3:-6].to(device), None, betas, \
                        new_poses[sidx:eidx, :3].to(device), trans[sidx:eidx].to(device))
        else:
            body = get_body_model_sequence(smpl_path, gender, process_inds[1] - process_inds[0], \
                        new_poses[sidx:eidx, 3:].to(device), None, betas, \
                        new_poses[sidx:eidx, :3].to(device), trans[sidx:eidx].to(device), use_smplh=False)

        smpl_jnts = body.Jtr
        smpl_jnts = smpl_jnts[:, :len(SMPL_JOINTS), :]

        smpl_verts = body.v # T X Nv X 3
        faces = body.f.data.cpu().numpy()  # Nf X 3

        body_vtx_seq.append(smpl_verts.data.cpu().numpy()) 
        body_jnt_seq.append(smpl_jnts.data.cpu().numpy()) # T X 24 X 3 

        process_inds[0] = process_inds[1]
        process_inds[1] = min([num_frames, process_inds[1] + SPLIT_FRAME_LIMIT])

    vtx_seq = np.concatenate(body_vtx_seq, axis=0)
    joint_seq = np.concatenate(body_jnt_seq, axis=0)

    # determine floor height and foot contacts 
    fps = 30 
    floor_height, contacts, discard_seq = determine_floor_height_and_contacts(joint_seq, fps)
    print('Floor height: %f' % (floor_height))
    
    # translate so floor is at z=0
    trans[:,2] -= floor_height
    joint_seq[:,:,2] -= floor_height
    vtx_seq[:,:,2] -= floor_height

    # Save to .obj
    vis_obj = True 
    if vis_obj:
        dest_mesh_vis_folder = os.path.join(vis_res_folder, k_name)
        if not os.path.exists(dest_mesh_vis_folder):
            os.makedirs(dest_mesh_vis_folder)

        if vis_gt:
            gt_mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                            "objs_gt")
            gt_out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                            "imgs_gt")
            gt_out_vid_file_path = os.path.join(vis_res_folder, \
                            k_name+"_gt.mp4")
        else:
            gt_mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                            "objs_pred")
            gt_out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                            "imgs_pred")
            gt_out_vid_file_path = os.path.join(vis_res_folder, \
                            k_name+"_pred.mp4")
        
        if not os.path.exists(gt_out_vid_file_path):
            # For visualizating both object and human mesh 
            save_verts_faces_to_mesh_file(vtx_seq[:150], faces, gt_mesh_save_folder)
            run_blender_rendering_and_save2video(gt_mesh_save_folder, gt_out_rendered_img_folder, gt_out_vid_file_path)

def determine_floor_height_and_contacts(body_joint_seq, fps):
    '''
    Input: body_joint_seq N x 21 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    '''
    num_frames = body_joint_seq.shape[0]

    # compute toe velocities
    root_seq = body_joint_seq[:, SMPL_JOINTS['hips'], :]
    left_toe_seq = body_joint_seq[:, SMPL_JOINTS['leftToeBase'], :]
    right_toe_seq = body_joint_seq[:, SMPL_JOINTS['rightToeBase'], :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_inds = np.append(left_static_inds, right_static_inds)

    discard_seq = False
    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)
      
        min_median = min_root_median = float('inf')
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
          
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        floor_height = min_median 
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET # toe joint is actually inside foot mesh a bit
    else:
        floor_height = offset_floor_height = 0.0

    # now find contacts (feet are below certain velocity and within certain range of floor)
    # compute heel velocities
    left_heel_seq = body_joint_seq[:, SMPL_JOINTS['leftFoot'], :]
    right_heel_seq = body_joint_seq[:, SMPL_JOINTS['rightFoot'], :]
    left_heel_vel = np.linalg.norm(left_heel_seq[1:] - left_heel_seq[:-1], axis=1)
    left_heel_vel = np.append(left_heel_vel, left_heel_vel[-1])
    right_heel_vel = np.linalg.norm(right_heel_seq[1:] - right_heel_seq[:-1], axis=1)
    right_heel_vel = np.append(right_heel_vel, right_heel_vel[-1])

    left_heel_contact = left_heel_vel < CONTACT_VEL_THRESH
    right_heel_contact = right_heel_vel < CONTACT_VEL_THRESH
    left_toe_contact = left_toe_vel < CONTACT_VEL_THRESH
    right_toe_contact = right_toe_vel < CONTACT_VEL_THRESH

    # compute heel heights
    left_heel_heights = left_heel_seq[:, 2] - floor_height
    right_heel_heights = right_heel_seq[:, 2] - floor_height
    left_toe_heights =  left_toe_heights - floor_height
    right_toe_heights =  right_toe_heights - floor_height

    left_heel_contact = np.logical_and(left_heel_contact, left_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    right_heel_contact = np.logical_and(right_heel_contact, right_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    left_toe_contact = np.logical_and(left_toe_contact, left_toe_heights < CONTACT_TOE_HEIGHT_THRESH)
    right_toe_contact = np.logical_and(right_toe_contact, right_toe_heights < CONTACT_TOE_HEIGHT_THRESH)

    contacts = np.zeros((num_frames, len(SMPL_JOINTS)))
    contacts[:,SMPL_JOINTS['leftFoot']] = left_heel_contact
    contacts[:,SMPL_JOINTS['leftToeBase']] = left_toe_contact
    contacts[:,SMPL_JOINTS['rightFoot']] = right_heel_contact
    contacts[:,SMPL_JOINTS['rightToeBase']] = right_toe_contact

    # hand contacts
    left_hand_contact = detect_joint_contact(body_joint_seq, 'leftHand', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    right_hand_contact = detect_joint_contact(body_joint_seq, 'rightHand', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:,SMPL_JOINTS['leftHand']] = left_hand_contact
    contacts[:,SMPL_JOINTS['rightHand']] = right_hand_contact

    # knee contacts
    left_knee_contact = detect_joint_contact(body_joint_seq, 'leftLeg', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    right_knee_contact = detect_joint_contact(body_joint_seq, 'rightLeg', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:,SMPL_JOINTS['leftLeg']] = left_knee_contact
    contacts[:,SMPL_JOINTS['rightLeg']] = right_knee_contact

    return offset_floor_height, contacts, discard_seq

def detect_joint_contact(body_joint_seq, joint_name, floor_height, vel_thresh, height_thresh):
    # calc velocity
    joint_seq = body_joint_seq[:, SMPL_JOINTS[joint_name], :]
    joint_vel = np.linalg.norm(joint_seq[1:] - joint_seq[:-1], axis=1)
    joint_vel = np.append(joint_vel, joint_vel[-1])
    # determine contact by velocity
    joint_contact = joint_vel < vel_thresh
    # compute heights
    joint_heights = joint_seq[:, 2] - floor_height
    # compute contact by vel + height
    joint_contact = np.logical_and(joint_contact, joint_heights < height_thresh)

    return joint_contact

def get_body_model_sequence(smplh_path, gender, num_frames,
                  pose_body, pose_hand, betas, root_orient, trans, use_smplh=True):
    gender = str(gender)
    # bm_path = os.path.join(smplh_path, gender + '/model.npz')
    if use_smplh:
        # bm_path = os.path.join(smplh_path, "SMPLH_"+gender+".pkl")
        bm_path = os.path.join(smplh_path, gender + '/model.npz')
        model_type = "smplh"
    else: # Use SMPL
        bm_path = os.path.join(smplh_path, "basicModel_m_lbs_10_207_0_v1.0.0.pkl")
        model_type = "smpl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm = BodyModel(bm_path=bm_path, num_betas=NUM_BETAS, batch_size=num_frames, model_type=model_type).to(device)

    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 
    kintree_table = ori_kintree_table[0, :22] # 22 
    kintree_table[0] = -1 # Assign -1 for the root joint's parent idx. 

    pose_body = pose_body.to(device)
    if pose_hand is not None:
        pose_hand = pose_hand.to(device)
    betas = torch.Tensor(np.repeat(betas[:NUM_BETAS][np.newaxis], num_frames, axis=0)).to(device).float()
    root_orient = root_orient.to(device)
    trans = trans.to(device)

    body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, root_orient=root_orient, trans=trans)
    
    return body, kintree_table 

def process_kinpoly_seq(data_path, vis_res_folder, out_file_folder):
    use_smplh = True  

    all_data = joblib.load(data_path)

    fps = 30 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_smplh:
        smplh_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh_amass" 
        gender = "male"
    else:
        # smpl_path = "/viscam/u/jiamanli/github/egomotion/body_models/smpl"
        smpl_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smpl" 
        gender = "male"
    
    # SMPL model joints
    # 0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee
    # 5: R_Knee, 6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3
    # 10: L_Foot, 11: R_Foot, 12: Neck, 13: L_Collar, 14: R_Collar
    # 15: Head, 16: L_Shoulder, 17: R_Shoulder, 18: L_Elbow, 19: R_Elbow
    # 20: L_Wrist, 21: R_Wrist, 22(25): L_Index1, 23(40): R_Index1

    # smpl_xml_joint_idx = np.array([0,1,4,7,10,2,5,8,11,3,6,  9,12,15,13,16,18,20,22,14,17,19,21,23]) # From SMPL to mujoco joints
    #                                0,1,2,3,4, 5,6,7,8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 

    # Mujoco model joints
    # 0: Pelvis, 1: L_Hip, 2: L_Knee, 3: L_Ankle, 4: L_Foot, 5: R_Hip, 6: R_Knee, 
    # 7: R_Ankle, 8: R_foot, 9: Spine1, 10: Spine2, 11: Spine3, 12: Neck, 13: Head, 
    # 14: L_Collar, 15: L_Shoulder, 16: L_Elbow, 17: L_Wrist, 18: L_Index1, 
    # 19: R_Collar, 20: R_Shoulder, 21: R_Elbow, 22: R_Wrist, 23: R_Index1.  
    
    mujoco2smpl_joint_idx = np.asarray([0,1,5,9,2,6,10,3,7,11,4,8,12,14,19,13,15,20,16,21,17,22,18,23]) # From mujoco joints to SMPL

    k_list = list(all_data.keys()) 
    print("Total sequences:{0}".format(len(k_list)))    
    for k_name in k_list: 
        start_t = time.time()

        seq_data = all_data[k_name]['qpos'] # T X 76 
       
        root_trans = seq_data[:, :3] # T X 3  
        root_quat = seq_data[:, 3:7] # T X 4
        joint_euler = seq_data[:, 7:] # T X (23*3) 

        poses = torch.from_numpy(joint_euler).float() # T X (23*3)
        trans = torch.from_numpy(root_trans).float() # T X 3 

        num_frames = poses.shape[0] 

        betas = np.zeros(10) # 10  
        rep_betas = torch.zeros(num_frames, 10).float() # T X 10 
        
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
        new_poses = new_poses.view(-1, 72) # T X 72, axis angle representation 

        root_orient = new_poses[:, :3].data.cpu().numpy()
        pose_body = new_poses[:, 3:-6].data.cpu().numpy() 

        # SMPL forward to generate joints and vertices 
        body_jnt_seq = [] 
        body_vtx_seq = []
        process_inds = [0, min([num_frames, SPLIT_FRAME_LIMIT])]
        while process_inds[0] < num_frames:
            print(process_inds)
            sidx, eidx = process_inds
            
            if use_smplh:
                body, kintree_table = get_body_model_sequence(smplh_path, gender, process_inds[1] - process_inds[0], \
                            new_poses[sidx:eidx, 3:-6].to(device), None, betas, \
                            new_poses[sidx:eidx, :3].to(device), trans[sidx:eidx].to(device))
            else:
                body = get_body_model_sequence(smpl_path, gender, process_inds[1] - process_inds[0], \
                            new_poses[sidx:eidx, 3:].to(device), None, betas, \
                            new_poses[sidx:eidx, :3].to(device), trans[sidx:eidx].to(device), use_smplh=False)

            smpl_jnts = body.Jtr
            smpl_jnts = smpl_jnts[:, :len(SMPL_JOINTS), :]

            smpl_verts = body.v # T X Nv X 3
            faces = body.f.data.cpu().numpy()  # Nf X 3

            body_vtx_seq.append(smpl_verts.data.cpu().numpy()) 
            body_jnt_seq.append(smpl_jnts.data.cpu().numpy()) # T X 24 X 3 

            process_inds[0] = process_inds[1]
            process_inds[1] = min([num_frames, process_inds[1] + SPLIT_FRAME_LIMIT])

        vtx_seq = np.concatenate(body_vtx_seq, axis=0)
        joint_seq = np.concatenate(body_jnt_seq, axis=0)

        # determine floor height and foot contacts 
        floor_height, contacts, discard_seq = determine_floor_height_and_contacts(joint_seq, fps)
        # print('Floor height: %f' % (floor_height))
        
        # translate so floor is at z=0
        trans[:,2] -= floor_height
        joint_seq[:,:,2] -= floor_height
        vtx_seq[:,:,2] -= floor_height

        # Save to .obj
        vis_obj = False
        if vis_obj:
            dest_mesh_vis_folder = os.path.join(vis_res_folder, k_name)
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)

            gt_mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                            "objs_gt")
            gt_out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                            "imgs_gt")
            gt_out_vid_file_path = os.path.join(vis_res_folder, \
                            k_name+".mp4")
            
            if not os.path.exists(gt_out_vid_file_path):
                # For visualizating both object and human mesh 
                save_verts_faces_to_mesh_file(vtx_seq[:120], faces, gt_mesh_save_folder)
                run_blender_rendering_and_save2video(gt_mesh_save_folder, gt_out_rendered_img_folder, gt_out_vid_file_path)

        # Prepare global head pose
        local_joint_aa = torch.cat((torch.from_numpy(root_orient).float(), torch.from_numpy(pose_body).float()), dim=1) # T X (3+21*3)
        local_joint_aa = local_joint_aa.reshape(-1, 22, 3) # T X 22 X 3  
        local_joint_rot_mat = transforms.axis_angle_to_matrix(local_joint_aa) # T X 22 X 3 X 3 
        global_joint_rot_mat = local2global_pose(local_joint_rot_mat, kintree_table) # T X 22 X 3 X 3 

        head_idx = 15 
        global_head_rot_mat = global_joint_rot_mat[:, head_idx, :] # T X 3 X 3 
        global_head_trans = torch.from_numpy(joint_seq[:, head_idx, :]).float() # T X 3 

        global_head_rot_mat_diff = torch.matmul(torch.inverse(global_head_rot_mat[:-1]), global_head_rot_mat[1:]) # (T-1) X 3 X 3 
        global_head_trans_diff = global_head_trans[1:, :] - global_head_trans[:-1, :] # (T-1) X 3

        global_head_rot_6d = transforms.matrix_to_rotation_6d(global_head_rot_mat) # T X 6
        global_head_rot_6d_diff = transforms.matrix_to_rotation_6d(global_head_rot_mat_diff) # (T-1) X 6   

        global_head_quat = transforms.matrix_to_quaternion(global_head_rot_mat) # T X 4 
        head_qpos = torch.cat((global_head_trans, global_head_quat), dim=-1) # T X 7
        head_vels = get_head_vel(head_qpos) # The head vels representation used in kinpoly. T X 6 

        # save
        # add number of frames and framrate to file path for each of loading
        if not os.path.exists(out_file_folder):
            os.makedirs(out_file_folder)
        output_file_path = k_name + '_%d_frames_%d_fps.npz' % (num_frames, int(fps))
        output_file_path = os.path.join(out_file_folder, output_file_path)
        np.savez(output_file_path, fps=fps,
                                seq_name=k_name,
                                gender=str(gender),
                                floor_height=floor_height,
                                contacts=contacts,
                                trans=trans,
                                root_orient=root_orient,
                                pose_body=pose_body,
                                betas=betas,
                                joints=joint_seq, 
                                head_qpos=head_qpos,
                                head_vels=head_vels,
                                global_head_rot_6d=global_head_rot_6d, 
                                global_head_trans=global_head_trans, 
                                global_head_rot_6d_diff=global_head_rot_6d_diff, 
                                global_head_trans_diff=global_head_trans_diff)

        print('Seq process time: %f s' % (time.time() - start_t))

def prep_smpl_to_single_data(ori_seq_folder, smpl_output_filename):
    smplh_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh_amass"

    # Convert the data format 
    smpl_seq_data = {} 

    total_frame_cnt = 0
    counter = 0

    npz_files = os.listdir(ori_seq_folder)
    for npz_name in npz_files:
        motion_path = os.path.join(ori_seq_folder, npz_name) 
        ori_npz_data = np.load(motion_path)

        pose_body = ori_npz_data['pose_body'] # T X 63 
    
        seq_length = pose_body.shape[0]
        total_frame_cnt += seq_length 

        dest_seq_name = str(ori_npz_data['seq_name'])
        smpl_seq_data[dest_seq_name] = {
            "root_orient": ori_npz_data['root_orient'],
            "body_pose": ori_npz_data['pose_body'],
            'trans': ori_npz_data['trans'],
            'beta': ori_npz_data['betas'],
            "seq_name": dest_seq_name,
            "gender": ori_npz_data['gender'],
            "head_qpos": ori_npz_data['head_qpos'], 
            "head_vels": ori_npz_data['head_vels'], 
            "global_head_trans": ori_npz_data['global_head_trans'],
            "global_head_rot_6d": ori_npz_data['global_head_rot_6d'],
            "global_head_rot_6d_diff": ori_npz_data['global_head_rot_6d_diff'],
            "global_head_trans_diff": ori_npz_data['global_head_trans_diff'],
        }

        # import pdb 
        # pdb.set_trace()
        counter += 1

    print("Total number of sequences:{0}".format(len(smpl_seq_data)))
    print("Total number of frames:{0}".format(total_frame_cnt))

    joblib.dump(smpl_seq_data, open(smpl_output_filename, "wb"))

def reorganize_data(ori_data_file):
    data_dict = joblib.load(ori_data_file)

    train_data_path = ori_data_file.replace(ori_data_file.split("/")[-1], "train_"+ori_data_file.split("/")[-1])
    test_data_path = ori_data_file.replace(ori_data_file.split("/")[-1], "test_"+ori_data_file.split("/")[-1])

    train_data_dict = {}
    test_data_dict = {}

    train_cnt = 0
    test_cnt = 0

    for seq_name in data_dict:
        test_data_dict[test_cnt] = {}
        test_data_dict[test_cnt] = data_dict[seq_name]
        test_cnt += 1 
  
    # joblib.dump(train_data_dict, open(train_data_path, 'wb'))
    joblib.dump(test_data_dict, open(test_data_path, 'wb'))

    # print("Total training sequences:{0}".format(len(train_data_dict))) # 
    print("Total testing sequences:{0}".format(len(test_data_dict))) # 

if __name__ == "__main__":
    data_path = "/viscam/u/jiamanli/datasets/kin-poly/MoCapData/features/mocap_annotations.p"
    vis_res_folder = "/viscam/u/jiamanli/vis_egoego_processed_data/kinpoly_smpl_vis"
    out_file_folder = "/move/u/jiamanli/datasets/egoego_processed_data/processed_kinpoly_mocap" 
    # process_kinpoly_seq(data_path, vis_res_folder, out_file_folder) 

    smpl_output_folder = "/move/u/jiamanli/datasets/egoego_processed_data/kinpoly_mocap_same_shape_egoego_processed"
    if not os.path.exists(smpl_output_folder):
        os.makedirs(smpl_output_folder)
    smpl_output_filename = os.path.join(smpl_output_folder, "kinpoly_mocap_smplh_motion.p")    
    prep_smpl_to_single_data(out_file_folder, smpl_output_filename)
    reorganize_data(smpl_output_filename)

'''
Total number of sequences:266
Total number of frames:148044
Total testing sequences:266
'''
