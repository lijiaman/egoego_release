# This code is modified from https://github.com/davrempe/humor and https://github.com/KlabCMU/kin-poly .
import sys
sys.path.append("../../")
import glob
import os
import argparse
import time
import math 
import joblib 

import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from body_model.body_model import BodyModel
from body_model.utils import SMPL_JOINTS, KEYPT_VERTS

import torch 
import pytorch3d.transforms as transforms 

#
# Processing options
#

OUT_FPS = 30
DISCARD_TERRAIN_SEQUENCES = True # throw away sequences where the person steps onto objects (determined by a heuristic)

DISCARD_SHORTER_THAN = 1.0 # seconds

# optional viz during processing
VIZ_PLOTS = False
VIZ_SEQ = False

ALL_DATASETS = ['ACCAD', 'BMLmovi', 'BioMotionLab_NTroje', 'BMLhandball', 'CMU', 'DanceDB', 'DFaust_67', 
                'EKUT', 'Eyes_Japan_Dataset', 'HumanEva', 'KIT', 'MPI_HDM05', 
                'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap', 
                'TotalCapture', 'Transitions_mocap'] # everything in AMASS
TRAIN_DATASETS = ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje', 'BMLmovi', 
                    'EKUT', 'ACCAD'] # training dataset
TEST_DATASETS = ['Transitions_mocap', 'HumanEva'] # test datasets
VAL_DATASETS = ['MPI_HDM05', 'SFU', 'MPI_mosh'] # validation datasets

# if sequence is longer than this, splits into sequences of this size to avoid running out of memory
# ~ 4000 for 12 GB GPU, ~2000 for 8 GB
SPLIT_FRAME_LIMIT = 2000

NUM_BETAS = 16 # size of SMPL shape parameter to use

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

def local2global_pose(local_pose, kintree):
    # local_pose: T X J X 3 X 3 
    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose

def transform_vec(v, q, trans='root'):
    if trans == 'root':
        rot = transforms.quaternion_to_matrix(q) # 3 X 3 
    elif trans == 'heading':
        hq = q.clone()
        hq[1] = 0.0
        hq[2] = 0.0
        hq /= torch.linalg.norm(hq)
        rot = transforms.quaternion_to_matrix(hq)
    else:
        assert False

    rot = rot.data.cpu().numpy()

    v = rot.T.dot(v[:, None]).ravel()
    return v

def rotation_from_quaternion(quaternion, separate=False):
    # if 1.0 - quaternion[0] < 1e-8:
    if np.abs(1.0 - quaternion[0]) < 1e-6 or np.abs(1  + quaternion[0]) < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        angle = 0.0
    else:
        angle = 2 * math.acos(quaternion[0])
        axis = quaternion[1:4] / math.sin(angle/2.0)
        axis /= np.linalg.norm(axis)
        
    return (axis, angle) if separate else axis * angle

def get_head_vel(head_pose, dt = 1/30):
    # get head velocity 
    head_vels = []
    head_qpos = head_pose[0]

    for i in range(head_pose.shape[0] - 1):
        curr_qpos = head_pose[i, :]
        next_qpos = head_pose[i + 1, :] 
        v = (next_qpos[:3] - curr_qpos[:3]) / dt
        v = transform_vec(v.data.cpu().numpy(), curr_qpos[3:7], 'heading') # get velocity within the body frame
        
        qrel = transforms.quaternion_multiply(next_qpos[3:7], transforms.quaternion_invert(curr_qpos[3:7]))
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

def get_body_model_sequence(smplh_path, gender, num_frames,
                  pose_body, pose_hand, betas, root_orient, trans):
    gender = str(gender)
    bm_path = os.path.join(smplh_path, gender + '/model.npz')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm = BodyModel(bm_path=bm_path, num_betas=NUM_BETAS, batch_size=num_frames).to(device)

    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 
    kintree_table = ori_kintree_table[0, :22] # 22 
    kintree_table[0] = -1 # Assign -1 for the root joint's parent idx. 

    pose_body = torch.Tensor(pose_body).to(device)
    if pose_hand is not None:
        pose_hand = torch.Tensor(pose_hand).to(device)
    betas = torch.Tensor(np.repeat(betas[:NUM_BETAS][np.newaxis], num_frames, axis=0)).to(device)
    root_orient = torch.Tensor(root_orient).to(device)
    trans = torch.Tensor(trans).to(device)
    body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, root_orient=root_orient, trans=trans)
    return body, kintree_table 

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

    if VIZ_PLOTS:
        fig = plt.figure()
        steps = np.arange(num_frames)
        plt.plot(steps, left_toe_vel, '-r', label='left vel')
        plt.plot(steps, right_toe_vel, '-b', label='right vel')
        plt.legend()
        plt.show()
        plt.close()

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]

    if VIZ_PLOTS:
        fig = plt.figure()
        steps = np.arange(num_frames)
        plt.plot(steps, left_toe_heights, '-r', label='left toe height')
        plt.plot(steps, right_toe_heights, '-b', label='right toe height')
        plt.plot(steps, root_heights, '-g', label='root height')
        plt.legend()
        plt.show()
        plt.close()

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_inds = np.append(left_static_inds, right_static_inds)

    if VIZ_PLOTS:
        fig = plt.figure()
        steps = np.arange(left_static_foot_heights.shape[0])
        plt.plot(steps, left_static_foot_heights, '-r', label='left static height')
        plt.legend()
        plt.show()
        plt.close()

    # fig = plt.figure()
    # plt.hist(all_static_foot_heights)
    # plt.show()
    # plt.close()

    discard_seq = False
    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)
        # print(all_labels)
        if VIZ_PLOTS:
            plt.figure()
        min_median = min_root_median = float('inf')
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
            if VIZ_PLOTS:
                plt.scatter(cur_clust, np.zeros_like(cur_clust), label='foot %d' % (cur_label))
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)
            if VIZ_PLOTS:
                plt.scatter(cur_root_clust, np.zeros_like(cur_root_clust), label='root %d' % (cur_label))

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        # print(cluster_heights)
        # print(cluster_root_heights)
        # print(cluster_sizes)
        if VIZ_PLOTS:
            plt.show()
            plt.close()

        floor_height = min_median 
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET # toe joint is actually inside foot mesh a bit

        if DISCARD_TERRAIN_SEQUENCES:
            # print(min_median + TERRAIN_HEIGHT_THRESH)
            # print(min_root_median + ROOT_HEIGHT_THRESH)
            for cluster_root_height, cluster_height, cluster_size in zip (cluster_root_heights, cluster_heights, cluster_sizes):
                root_above_thresh = cluster_root_height > (min_root_median + ROOT_HEIGHT_THRESH)
                toe_above_thresh = cluster_height > (min_median + TERRAIN_HEIGHT_THRESH)
                cluster_size_above_thresh = cluster_size > int(CLUSTER_SIZE_THRESH*fps)
                if root_above_thresh and toe_above_thresh and cluster_size_above_thresh:
                    discard_seq = True
                    print('DISCARDING sequence based on terrain interaction!')
                    break
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

def process_seq(data_paths):
    start_t = time.time()

    input_file_path, output_file_path, smplh_path = data_paths
    print(input_file_path)

    bdata = np.load(input_file_path)
    # gender = np.array(bdata['gender'], ndmin=1)[0]
    # gender = str(gender, 'utf-8') if isinstance(gender, bytes) else str(gender)
    gender = "male" # For using the same skeleton. 
    fps = bdata['mocap_framerate']
    num_frames = bdata['poses'].shape[0]
    trans = bdata['trans'][:]               # global translation
    root_orient = bdata['poses'][:, :3]     # global root orientation (1 joint)
    pose_body = bdata['poses'][:, 3:66]     # body joint rotations (21 joints)
    pose_hand = bdata['poses'][:, 66:]      # finger articulation joint rotations
    # betas = bdata['betas'][:]               # body shape parameters
    betas = np.zeros(NUM_BETAS) # For using the same skeleton 

    # correct mislabeled data
    if input_file_path.find('BMLhandball') >= 0:
        fps = 240
    if input_file_path.find('20160930_50032') >= 0 or input_file_path.find('20161014_50033') >= 0:
        fps = 59

    print(gender)
    print('fps: %d' % (fps))
    print(trans.shape)
    print(root_orient.shape)
    print(pose_body.shape)
    print(pose_hand.shape)
    print(betas.shape)

    # only keep middle 80% of sequences to avoid redundanct static poses
    trim_data = [trans, root_orient, pose_body, pose_hand]
    for i, data_seq in enumerate(trim_data):
        trim_data[i] = data_seq[int(0.1*num_frames):int(0.9*num_frames)]
    trans, root_orient, pose_body, pose_hand = trim_data
    num_frames = trans.shape[0]

    # discard if shorter than threshold
    if num_frames < DISCARD_SHORTER_THAN*fps:
        print('Sequence shorter than %f s, discarding...' % (DISCARD_SHORTER_THAN))
        return

    print(trans.shape)
    print(root_orient.shape)
    print(pose_body.shape)
    print(pose_hand.shape)
    print(betas.shape)

    # must do SMPL forward pass to get joints
    # split into manageable chunks to avoid running out of GPU memory for SMPL
    body_joint_seq = []
    body_vtx_seq = []
    process_inds = [0, min([num_frames, SPLIT_FRAME_LIMIT])]
    while process_inds[0] < num_frames:
        print(process_inds)
        sidx, eidx = process_inds
        body, kintree_table = get_body_model_sequence(smplh_path, gender, process_inds[1] - process_inds[0],
                            pose_body[sidx:eidx], pose_hand[sidx:eidx], betas, root_orient[sidx:eidx], trans[sidx:eidx])
        cur_joint_seq = body.Jtr.data.cpu()
        cur_body_joint_seq = cur_joint_seq[:, :len(SMPL_JOINTS), :]
        body_joint_seq.append(cur_body_joint_seq)

        cur_vtx_seq = body.v.data.cpu()
        body_vtx_seq.append(cur_vtx_seq)

        process_inds[0] = process_inds[1]
        process_inds[1] = min([num_frames, process_inds[1] + SPLIT_FRAME_LIMIT])

    joint_seq = np.concatenate(body_joint_seq, axis=0)
    print(joint_seq.shape)

    vtx_seq = np.concatenate(body_vtx_seq, axis=0)
    print(vtx_seq.shape)

    # determine floor height and foot contacts 
    floor_height, contacts, discard_seq = determine_floor_height_and_contacts(joint_seq, fps)
    # print('Floor height: %f' % (floor_height))
    
    # translate so floor is at z=0
    trans[:,2] -= floor_height
    joint_seq[:,:,2] -= floor_height   
    vtx_seq[:,:,2] -= floor_height

    # downsample before saving
    if OUT_FPS != fps:
        if OUT_FPS > fps:
            print('Cannot supersample data, saving at data rate!')
        else:
            fps_ratio = float(OUT_FPS) / fps
            print('Downsamp ratio: %f' % (fps_ratio))
            new_num_frames = int(fps_ratio*num_frames)
            print('Downsamp num frames: %d' % (new_num_frames))
            # print(cur_num_frames)
            # print(new_num_frames)
            downsamp_inds = np.linspace(0, num_frames-1, num=new_num_frames, dtype=int)
            # print(downsamp_inds)

            # update data to save
            fps = OUT_FPS
            num_frames = new_num_frames
            contacts = contacts[downsamp_inds]
            trans = trans[downsamp_inds]
            root_orient = root_orient[downsamp_inds]
            pose_body = pose_body[downsamp_inds]
            pose_hand = pose_hand[downsamp_inds]
            joint_seq = joint_seq[downsamp_inds]

    if discard_seq:
        print('Terrain interaction detected, discarding...')
        return

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
    output_file_path = output_file_path[:-4] + '_%d_frames_%d_fps.npz' % (num_frames, int(fps))
    np.savez(output_file_path, fps=fps,
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

def prep_smpl_to_single_data(smplh_path, ori_amass_root_folder, smpl_output_filename):
    # Convert the data format 
    smpl_seq_data = {} 

    total_frame_cnt = 0
    counter = 0

    subset_folders = os.listdir(ori_amass_root_folder)
    for subset_name in subset_folders: 
        if ".log" not in subset_name and "script" not in subset_name:   
            subset_folder_path = os.path.join(ori_amass_root_folder, subset_name)
            seq_folders = os.listdir(subset_folder_path)
            for seq_name in seq_folders:
                seq_folder_path = os.path.join(subset_folder_path, seq_name) 

                npz_files = os.listdir(seq_folder_path)

                for npz_name in npz_files:
                    motion_path = os.path.join(seq_folder_path, npz_name) 
                    ori_npz_data = np.load(motion_path)

                    pose_body = ori_npz_data['pose_body'] # T X 63 
                
                    seq_length = pose_body.shape[0]
                    total_frame_cnt += seq_length 

                    dest_seq_name = subset_name+"-"+seq_name+"-"+npz_name

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
    TRAIN_DATASETS = ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje', 'BMLmovi', 
                    'EKUT', 'ACCAD'] # training dataset
    TEST_DATASETS = ['Transitions_mocap', 'HumanEva'] # test datasets
    VAL_DATASETS = ['MPI_HDM05', 'SFU', 'MPI_mosh'] # validation datasets

    data_dict = joblib.load(ori_data_file)

    train_data_path = ori_data_file.replace(ori_data_file.split("/")[-1], "train_"+ori_data_file.split("/")[-1])
    test_data_path = ori_data_file.replace(ori_data_file.split("/")[-1], "test_"+ori_data_file.split("/")[-1])

    train_data_dict = {}
    test_data_dict = {}

    train_cnt = 0
    test_cnt = 0

    for seq_name in data_dict:
        for train_name in TRAIN_DATASETS:
            if train_name in seq_name:
                train_data_dict[train_cnt] = {}
                train_data_dict[train_cnt] = data_dict[seq_name]
                train_cnt += 1 
                break 
        for val_name in VAL_DATASETS:
            if val_name in seq_name:
                test_data_dict[test_cnt] = {}
                test_data_dict[test_cnt] = data_dict[seq_name]
                test_cnt += 1
                break 
        for test_name in TEST_DATASETS:
            if test_name in seq_name:
                test_data_dict[test_cnt] = {}
                test_data_dict[test_cnt] = data_dict[seq_name]
                test_cnt += 1 
                break 
  
    joblib.dump(train_data_dict, open(train_data_path, 'wb'))
    joblib.dump(test_data_dict, open(test_data_path, 'wb'))

    print("Total training sequences:{0}".format(len(train_data_dict))) # 
    print("Total testing sequences:{0}".format(len(test_data_dict))) # 

'''
Total number of sequences:13106
Total number of frames:3877903
Total training sequences:11655
Total testing sequences:462
'''

def main(config):
    start_time = time.time()
    out_folder = config.out
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # get all available datasets
    all_dataset_dirs = [os.path.join(config.amass_root, f) for f in sorted(os.listdir(config.amass_root)) if f[0] != '.']
    all_dataset_dirs = [f for f in all_dataset_dirs if os.path.isdir(f)]
    print('Found %d available datasets from raw AMASS data source.' % (len(all_dataset_dirs)))
    all_dataset_names = [f.split('/')[-1] for f in all_dataset_dirs]
    print(all_dataset_names)

    # requested datasets
    dataset_dirs = [os.path.join(config.amass_root, f) for f in config.datasets]
    dataset_names = config.datasets
    print('Requested datasets:')
    print(dataset_dirs)
    print(dataset_names)

    # go through each dataset to set up directory structure before processing
    all_seq_in_files = []
    all_seq_out_files = []
    for data_dir, data_name in zip(dataset_dirs, dataset_names):
        if not os.path.exists(data_dir):
            print('Could not find dataset %s in available raw AMASS data!' % (data_name))
            return

        cur_output_dir = os.path.join(out_folder, data_name)
        if not os.path.exists(cur_output_dir):
            os.mkdir(cur_output_dir)

        # first create subject structure in output
        cur_subject_dirs = [f for f in sorted(os.listdir(data_dir)) if f[0] != '.' and os.path.isdir(os.path.join(data_dir, f))]
        print(cur_subject_dirs)
        for subject_dir in cur_subject_dirs:
            cur_subject_out = os.path.join(cur_output_dir, subject_dir)
            if not os.path.exists(cur_subject_out):
                os.mkdir(cur_subject_out)

        # then collect all sequence input files
        input_seqs = glob.glob(os.path.join(data_dir, '*/*_poses.npz'))
        print(len(input_seqs))

        # and create output sequence file names
        output_file_names = ['/'.join(f.split('/')[-2:]) for f in input_seqs]
        output_seqs = [os.path.join(cur_output_dir, f) for f in output_file_names]
        print(len(output_seqs))

        # already_processed = [i for i in range(len(output_seqs)) if len(glob.glob(output_seqs[i][:-4] + '*.npz')) == 1]
        # already_processed_output_names =  [output_file_names[i] for i in already_processed]
        # print('Already processed these sequences, skipping:')
        # print(already_processed_output_names)
        # not_already_processed = [i for i in range(len(output_seqs)) if len(glob.glob(output_seqs[i][:-4] + '*.npz')) == 0]
        # input_seqs = [input_seqs[i] for i in not_already_processed]
        # output_seqs = [output_seqs[i] for i in not_already_processed]

        all_seq_in_files += input_seqs
        all_seq_out_files += output_seqs
    
    smplh_paths = [config.smplh_root]*len(all_seq_in_files)
    data_paths = list(zip(all_seq_in_files, all_seq_out_files, smplh_paths))

    for data_in in data_paths:
        process_seq(data_in)

    total_time = time.time() - start_time
    print('TIME TO PROCESS: %f min' % (total_time / 60.0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--amass-root', type=str, default='data/amass', help='Root directory of raw AMASS dataset.')
    parser.add_argument('--datasets', type=str, nargs='+', default=ALL_DATASETS, help='Which datasets to process. By default processes all.')
    parser.add_argument('--out', type=str, default='data/processed_amass_same_shape_seq', help='Root directory to save processed output to.')
    parser.add_argument('--smpl_output_folder', type=str, default='data/amass_same_shape_egoego_processed', help='Root directory to save processed output to.')
    parser.add_argument('--smplh-root', type=str, default='smpl_models/smplh_amass', help='Root directory of the SMPL+H body model.')

    config = parser.parse_known_args()
    config = config[0]

    main(config)

    smpl_output_folder = config.smpl_output_folder
    if not os.path.exists(smpl_output_folder):
        os.makedirs(smpl_output_folder)
    smpl_output_filename = os.path.join(smpl_output_folder, "amass_smplh_motion.p")
    prep_smpl_to_single_data(config.smplh_root, config.out, smpl_output_filename)
    reorganize_data(smpl_output_filename) 

'''
By running this code, we can get the following files. 
amass_smplh_motion.p contains all the sequences in AMASS. 
train_amass_smplh_motion.p contains training sequences. 
test_amass_smplh_motion.p contains testing sequences. 
'''