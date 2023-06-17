import os
import sys
sys.path.append(os.getcwd())
sys.path.append("../../")
import numpy as np
import glob
import pickle as pk 
import joblib
import yaml 

from tqdm import tqdm

from scipy.spatial.transform import Rotation as sRot

from collections import defaultdict

import torch 
import pytorch3d.transforms as transforms 

from process_amass_dataset import get_body_model_sequence, local2global_pose, get_head_vel, determine_floor_height_and_contacts 

SPLIT_FRAME_LIMIT = 2000

def prep_smpl_to_single_data(smplh_path, ori_amass_root_folder, smpl_output_folder):
    # smplh_path = "smpl_all_models/smplh_amass"
    bm_path = os.path.join(smplh_path, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 
    kintree_table = ori_kintree_table[0, :22] # 22 
    kintree_table[0] = -1 # Assign -1 for the root joint's parent idx. 

    # Convert the data format 
    smpl_seq_data = {} 
    
    if not os.path.exists(smpl_output_folder):
        os.makedirs(smpl_output_folder)
    smpl_output_filename = os.path.join(smpl_output_folder, "ares_smplh_motion.p")

    # ori_amass_root_folder = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/habitat_rendering_replica_all" 

    total_frame_cnt = 0
    counter = 0

    subset_folders = os.listdir(ori_amass_root_folder)
    for subset_name in subset_folders: 
        if ".log" not in subset_name and "script" not in subset_name:   
            subset_folder_path = os.path.join(ori_amass_root_folder, subset_name)
            seq_folders = os.listdir(subset_folder_path)
            for seq_name in seq_folders:
                seq_folder_path = os.path.join(subset_folder_path, seq_name) 
                
                # Add of path 
                flow_folder = os.path.join(seq_folder_path, "raft_flows")
                if os.path.exists(flow_folder):
                    of_files = os.listdir(flow_folder)
                    of_files.sort() 
                    of_path_list = []
                    for of_name in of_files:
                        if ".npy" in of_name and ".png" not in of_name: 
                            of_path = os.path.join(flow_folder, of_name)
                            of_path_list.append(of_path)

                    motion_path = os.path.join(seq_folder_path, "ori_motion_seq.npz") 
                    ori_npz_data = np.load(motion_path)

                    root_orient = ori_npz_data['root_orient'] # T X 3 
                    pose_body = ori_npz_data['pose_body'] # T X 63 
                    trans = ori_npz_data['trans'] # T X 3 
                    betas = ori_npz_data['betas'] # 16 
                    gender = ori_npz_data['gender']

                    seq_length = pose_body.shape[0]
                    total_frame_cnt += seq_length 

                    # must do SMPL forward pass to get joints
                    # split into manageable chunks to avoid running out of GPU memory for SMPL
                    body_joint_seq = []
                    body_vtx_seq = []
                    process_inds = [0, min([seq_length, SPLIT_FRAME_LIMIT])]
                    while process_inds[0] < seq_length:
                        print(process_inds)
                        sidx, eidx = process_inds
                        body, kintree_table = get_body_model_sequence(smplh_path, gender, process_inds[1] - process_inds[0],
                                            pose_body[sidx:eidx], None, betas, root_orient[sidx:eidx], trans[sidx:eidx])
                        
                        cur_joint_seq = body.Jtr.data.cpu()
                        cur_body_joint_seq = cur_joint_seq[:, :22, :]
                        body_joint_seq.append(cur_body_joint_seq)

                        cur_vtx_seq = body.v.data.cpu()
                        body_vtx_seq.append(cur_vtx_seq)

                        process_inds[0] = process_inds[1]
                        process_inds[1] = min([seq_length, process_inds[1] + SPLIT_FRAME_LIMIT])

                    joint_seq = np.concatenate(body_joint_seq, axis=0)
                    print(joint_seq.shape)

                    vtx_seq = np.concatenate(body_vtx_seq, axis=0)
                    print(vtx_seq.shape)

                    # determine floor height and foot contacts 
                    fps = 30 
                    floor_height, contacts, discard_seq = determine_floor_height_and_contacts(joint_seq, fps)
                    # print('Floor height: %f' % (floor_height))
                    
                    # translate so floor is at z=0
                    trans[:,2] -= floor_height
                    joint_seq[:,:,2] -= floor_height   
                    vtx_seq[:,:,2] -= floor_height

                    # Convert local joint rotation to global joint rotation
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
                    
                    dest_seq_name = subset_name+"-"+seq_name

                    smpl_seq_data[dest_seq_name] = {
                        "root_orient": root_orient,
                        "body_pose": pose_body,
                        'trans': trans,
                        'beta': betas,
                        "seq_name": dest_seq_name,
                        "gender": gender,
                        "of_files": of_path_list,
                        "head_qpos": head_qpos.data.cpu().numpy(), 
                        "head_vels": head_vels, 
                        "global_head_trans": global_head_trans.data.cpu().numpy(),
                        "global_head_rot_6d": global_head_rot_6d.data.cpu().numpy(),
                        "global_head_rot_6d_diff": global_head_rot_6d_diff.data.cpu().numpy(),
                        "global_head_trans_diff": global_head_trans_diff.data.cpu().numpy(),
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

    TEST_SCENES = ['frl_apartment_4', 'office_0', 'hotel_0', 'room_2', 'apartment_0']

    data_dict = joblib.load(ori_data_file)

    train_data_path = ori_data_file.replace(ori_data_file.split("/")[-1], "train_"+ori_data_file.split("/")[-1])
    test_data_path = ori_data_file.replace(ori_data_file.split("/")[-1], "test_"+ori_data_file.split("/")[-1])

    train_data_dict = {}
    test_data_dict = {}

    train_cnt = 0
    test_cnt = 0

    for seq_name in data_dict:
        for train_name in TRAIN_DATASETS:
            curr_scene = seq_name.split("-")[0]
            if train_name in seq_name and curr_scene not in TEST_SCENES:
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

    print("Total training sequences:{0}".format(len(train_data_dict))) # 9162
    print("Total testing sequences:{0}".format(len(test_data_dict))) # 1213


'''
After processing split scenes:
Train scenes:{'apartment_2': 100946, 'frl_apartment_1': 107733, 'frl_apartment_5': 101530, 'office_4': 103673, 'apartment_1': 104556, 'office_3': 94984, 'frl_apartment_2': 80124, 'room_0': 78280, 'frl_apartment_0': 91350, 'office_1': 63388, 'office_2': 92103, 'frl_apartment_3': 91445, 'room_1': 71205}
Test scenes:{'apartment_2': 9028, 'frl_apartment_1': 10656, 'office_0': 690, 'hotel_0': 6278, 'frl_apartment_5': 13457, 'office_4': 7796, 'apartment_1': 6004, 'office_3': 11120, 'frl_apartment_2': 8238, 'room_0': 6816, 'frl_apartment_0': 11674, 'office_1': 2096, 'room_2': 4180, 'office_2': 7268, 'frl_apartment_3': 11289, 'room_1': 4542, 'apartment_0': 10920, 'frl_apartment_4': 12782}
Train scenes number:13
Test scene number:18
Total training frames:1181317
Total training duration:10.938120370370369 hours
Total testing frames:144834
Total testing duration:1.3410555555555557 hours
Total Unseen Scene testing frames:34850
Total Unseen Scene testing duration:0.3226851851851852 hours
'''

if __name__ == "__main__":
    smplh_path = "smpl_models/smplh_amass"
    ori_amass_data_folder = "data/ares/ares_ego_videos"
    smpl_output_folder = "data/ares_egoego_processed"
    prep_smpl_to_single_data(smplh_path, ori_amass_data_folder, smpl_output_folder)
    # w of 
    # Total number of sequences:12977
    # Total number of frames:1664504

    # Convert the data to the same organized folder as original kinpoly data 
    ori_data_file = os.path.join(smpl_output_folder, "ares_smplh_motion.p")
    reorganize_data(ori_data_file)
