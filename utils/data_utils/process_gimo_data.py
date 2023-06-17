from inspect import trace
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
sys.path.append("../../..")

import glob
import os
import argparse
import time

import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from body_model.body_model import BodyModel
from body_model.utils import SMPL_JOINTS
from viz.utils import viz_smpl_seq
from utils.torch import copy2cpu as c2c
from utils.transforms import batch_rodrigues, compute_world2aligned_mat, compute_world2aligned_joints_mat

from scipy.spatial.transform import Rotation
# from utils_common import show3Dpose_animation 

#
# Processing options
#

OUT_FPS = 30
SAVE_MOJO_VERTS = False # save MOJO vertex locations
SAVE_HAND_POSE = False # save joint angles for the hand
SAVE_VELOCITIES = True # save all parameter velocities available
SAVE_ALIGN_ROT = True # save rot mats that go from world root orient to aligned root orient
SAVE_JOINT_FRAME = True # saves various information to compute the local frame based on the body joints rather than SMPL root orient/trans
DISCARD_TERRAIN_SEQUENCES = False # throw away sequences where the person steps onto objects (determined by a heuristic)

VIZ_PLOTS = False
VIZ_SEQ = False
# ACCAD  BMLmovi  BMLrub  CMU  DFaust  EKUT  Eyes_Japan_Dataset  
# HDM05  HumanEva  KIT  MoSh  PosePrior  SFU  SSM  TCDHands  TotalCapture  Transitions
ALL_DATASETS = ['ACCAD', 'BMLmovi', 'BMLrub', 'CMU', 'DFaust', 
                'EKUT', 'Eyes_Japan_Dataset', 'HumanEva', 'KIT', 'HDM05', 
                'PosePrior', 'MoSh', 'SFU', 'SSM', 'TCDHands', 
                'TotalCapture', 'Transitions'] # everything in AMASS
TRAIN_DATASETS = ['CMU', 'PosePrior', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLrub', 'BMLmovi', 
                    'EKUT', 'ACCAD'] # HuMoR training dataset
TEST_DATASETS = ['Transitions', 'HumanEva'] # HuMoR test datasets
VAL_DATASETS = ['HDM05', 'SFU', 'MoSh'] # HuMoR validation datasets

# chosen virtual mocap markers to save
MOJO_VERTS = [4404, 920, 3076, 3169, 823, 4310, 1010, 1085, 4495, 4569, 6615, 3217, 3313, 6713,
            6785, 3383, 6607, 3207, 1241, 1508, 4797, 4122, 1618, 1569, 5135, 5040, 5691, 5636,
            5404, 2230, 2173, 2108, 134, 3645, 6543, 3123, 3024, 4194, 1306, 182, 3694, 4294, 744]

# if sequence is longer than this, splits into sequences of this size to avoid running out of memory
# ~ 4000 for 12 GB GPU, ~2000 for 8 GB
SPLIT_FRAME_LIMIT = 2000

NUM_BETAS = 10 # size of SMPL shape parameter to use

DISCARD_SHORTER_THAN = 1.0 # seconds

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

def debug_viz_seq(body, fps, contacts=None):
    viz_smpl_seq(body, imw=1080, imh=1080, fps=fps, contacts=contacts,
            render_body=False, render_joints=True, render_skeleton=False, render_ground=True)

def get_body_model_sequence(smplh_path, gender, num_frames,
                  pose_body, pose_hand, betas, root_orient, trans):
    gender = str(gender)
    # bm_path = os.path.join(smplh_path, gender + '/model.npz')
    # bm_path = os.path.join(smplh_path, 'SMPLX_MALE.npz')
    bm_path = os.path.join(smplh_path, 'SMPLH_male.pkl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm = BodyModel(bm_path=bm_path, num_betas=NUM_BETAS, batch_size=num_frames).to(device)

    pose_body = torch.Tensor(pose_body).to(device)
    # pose_hand = torch.Tensor(pose_hand).to(device)
    betas = torch.Tensor(np.repeat(betas[:NUM_BETAS][np.newaxis], num_frames, axis=0)).to(device)
    root_orient = torch.Tensor(root_orient).to(device)
    trans = torch.Tensor(trans).to(device)
    body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, root_orient=root_orient, trans=trans)
    return body

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

def compute_align_mats(root_orient):
    '''   compute world to canonical frame for each timestep (rotation around up axis) '''
    num_frames = root_orient.shape[0]
    # convert aa to matrices
    root_orient_mat = batch_rodrigues(torch.Tensor(root_orient).reshape(-1, 3)).numpy().reshape((num_frames, 9))

    # return compute_world2aligned_mat(torch.Tensor(root_orient_mat).reshape((num_frames, 3, 3))).numpy()

    # rotate root so that forward is facing +y by aligning local body right vector (-x) with world right vector (+x)
    #       with a rotation around the up axis (+z)
    body_right = -root_orient_mat.reshape((num_frames, 3, 3))[:,:,0] # in body coordinates body x-axis is to the left
    world2aligned_mat, world2aligned_aa = compute_align_from_right(body_right)

    return world2aligned_mat

def compute_joint_align_mats(joint_seq):
    '''
    Compute world to canonical frame for each timestep (rotation around up axis)
    from the given joint seq (T x J x 3)
    '''
    left_idx = SMPL_JOINTS['leftUpLeg']
    right_idx = SMPL_JOINTS['rightUpLeg']

    body_right = joint_seq[:, right_idx] - joint_seq[:, left_idx]
    body_right = body_right / np.linalg.norm(body_right, axis=1)[:,np.newaxis]

    world2aligned_mat, world2aligned_aa = compute_align_from_right(body_right)

    return world2aligned_mat

def compute_align_from_right(body_right):
    world2aligned_angle = np.arccos(body_right[:,0] / (np.linalg.norm(body_right[:,:2], axis=1) + 1e-8)) # project to world x axis, and compute angle
    body_right[:,2] = 0.0
    world2aligned_axis = np.cross(body_right, np.array([[1.0, 0.0, 0.0]]))

    world2aligned_aa = (world2aligned_axis / (np.linalg.norm(world2aligned_axis, axis=1)[:,np.newaxis]+ 1e-8)) * world2aligned_angle[:,np.newaxis]
    world2aligned_mat = batch_rodrigues(torch.Tensor(world2aligned_aa).reshape(-1, 3)).numpy()

    return world2aligned_mat, world2aligned_aa

def estimate_velocity(data_seq, h):
    '''
    Given some data sequence of T timesteps in the shape (T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    - h : step size
    '''
    data_tp1 = data_seq[2:]
    data_tm1 = data_seq[0:-2]
    data_vel_seq = (data_tp1 - data_tm1) / (2*h)
    return data_vel_seq

def estimate_angular_velocity(rot_seq, h):
    '''
    Given a sequence of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_velocity(rot_seq, h)
    R = rot_seq[1:-1]
    RT = np.swapaxes(R, -1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = np.matmul(dRdt, RT) 

    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)

    return w

def process_seq(data_paths):
    start_t = time.time()

    input_file_path, output_file_path, smplh_path = data_paths
    print(input_file_path)

    # load in input data
    # we leave out "dmpls" and "marker_data"/"marker_label" which are not present in all datasets
    bdata = np.load(input_file_path)
    # gender = np.array(bdata['gender'], ndmin=1)[0]
    # gender = str(gender, 'utf-8') if isinstance(gender, bytes) else str(gender)
    gender = "male"
    fps = 30
    num_frames = bdata['poses'].shape[0]
    trans = bdata['root_trans'][:]               # global translation
    root_orient = bdata['root_orient'][:, :3]     # global root orientation (1 joint)
    pose_body = bdata['poses'][:, :, :].reshape(-1, 21*3)     # body joint rotations (21 joints)
    # pose_hand = bdata['poses'][:, 66:]      # finger articulation joint rotations
    # betas = bdata['beta'][:]               # body shape parameters
    betas = np.zeros(10)

    print(gender)
    print('fps: %d' % (fps))
    print(trans.shape)
    print(root_orient.shape)
    print(pose_body.shape)
    # print(pose_hand.shape)
    print(betas.shape)

    # random_rotation = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()

    # R_s = Rotation.from_rotvec(root_orient).as_matrix() # T X 3 X 3 

    # ori_s = Rotation.from_matrix(random_rotation @ R_s).as_rotvec() # T X 3 
    # trans_s = (random_rotation @ trans[:, :, np.newaxis]) # T X 3 X 1 

    # root_orient = ori_s.copy() # T X 3 
    # trans = trans_s[:, :, 0].copy()  # T X 3 

    # must do SMPL forward pass to get joints
    # split into manageable chunks to avoid running out of GPU memory for SMPL
    body_joint_seq = []
    process_inds = [0, min([num_frames, SPLIT_FRAME_LIMIT])]
    while process_inds[0] < num_frames:
        print(process_inds)
        sidx, eidx = process_inds
        body = get_body_model_sequence(smplh_path, gender, process_inds[1] - process_inds[0],
                            pose_body[sidx:eidx], None, betas, \
                            root_orient[sidx:eidx], trans[sidx:eidx])
        cur_joint_seq = c2c(body.Jtr)
        cur_body_joint_seq = cur_joint_seq[:, :len(SMPL_JOINTS), :]
        body_joint_seq.append(cur_body_joint_seq)

        process_inds[0] = process_inds[1]
        process_inds[1] = min([num_frames, process_inds[1] + SPLIT_FRAME_LIMIT])

    joint_seq = np.concatenate(body_joint_seq, axis=0)
    print(joint_seq.shape)

    # determine floor height and foot contacts 
    floor_height, contacts, discard_seq = determine_floor_height_and_contacts(joint_seq, fps)
    print('Floor height: %f' % (floor_height))
    # translate so floor is at z=0
    trans[:,2] -= floor_height
    joint_seq[:,:,2] -= floor_height

    # Visualization for debugging 
    debug = False 
    if debug:
        dest_folder = "./tmp_debug_vis"
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        dest_vis_path = os.path.join(dest_folder, "gimo_seq_vis.gif")
        show3Dpose_animation(joint_seq[np.newaxis][:, :30, :, :], dest_vis_path) 
        
    
    pose_hand = None

    # save
    # add number of frames and framrate to file path for each of loading
    output_file_path = output_file_path[:-4] + '_%d_frames_%d_fps.npz' % (num_frames, int(fps))
    np.savez(output_file_path, fps=fps,
                               gender=str(gender),
                               floor_height=floor_height, # float number 
                               contacts=contacts, # T X 22 
                               trans=trans, # T X 3
                               root_orient=root_orient, # T X 3 
                               pose_body=pose_body, # T X 63
                               pose_hand=pose_hand,
                               betas=betas, # 16
                               joints=joint_seq,) # T X 22 X 3) # T X 3 X 3: Based on joint positions with trans, calculate the rotation matrix of the right vector to canonical axis. 

    print('Seq process time: %f s' % (time.time() - start_t))


def main(smplh_path, ori_data_folder, dest_data_folder):
    all_seq_in_files = []
    all_seq_out_files = []

    # smplh_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh" 
    # dest_data_folder = "/move/u/jiamanli/datasets/gimo_processed/smplx_npz_processed"
    # ori_data_folder = "/move/u/jiamanli/datasets/gimo_processed/smplx_npz"
    scene_files = os.listdir(ori_data_folder)
    for scene_name in scene_files:
        scene_folder = os.path.join(ori_data_folder, scene_name)

        dest_scene_folder = os.path.join(dest_data_folder, scene_name) 
        if not os.path.exists(dest_scene_folder):
            os.makedirs(dest_scene_folder) 

        npy_files = os.listdir(scene_folder)
        for npy_name in npy_files:
            ori_npy_path = os.path.join(scene_folder, npy_name)
            dest_npy_path = os.path.join(dest_scene_folder, npy_name) 

            all_seq_in_files.append(ori_npy_path)
            all_seq_out_files.append(dest_npy_path)
    
    smplh_paths = [smplh_path]*len(all_seq_in_files)
    data_paths = list(zip(all_seq_in_files, all_seq_out_files, smplh_paths))

    for data_in in data_paths:
        process_seq(data_in)
    
if __name__ == "__main__":
    smplh_path = "smpl_models/smplh_amass"
    ori_data_folder = "data/gimo/smplx_npz"
    dest_data_folder = "data/gimo/smplx_npz_processed" 
    main()
