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
import shutil 

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
from copycat.utils.tools import get_expert_v2
from copycat.data_loaders.dataset_amass_single import DatasetAMASSSingle
from copycat.data_loaders.dataset_smpl_obj import DatasetSMPLObj
from copycat.envs.humanoid_im import HumanoidEnv as CC_HumanoidEnv

from copycat.utils.config import Config as CC_Config

from collections import defaultdict

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

np.random.seed(1)
left_right_idx = [ 0,  2,  1,  3,  5,  4,  6,  8,  7,  9, 11, 10, 12, 14, 13, 15, 17,16, 19, 18, 21, 20, 23, 22]

def merge_droid_slam_res():
    ori_res_folder = "/viscam/u/jiamanli/datasets/kin-poly/OriData/droid_slam_res_ori"
    dest_res_folder = ori_res_folder.replace("_ori", "")
    if not os.path.exists(dest_res_folder):
        os.makedirs(dest_res_folder)

    seq_dict = {}

    npy_files = os.listdir(ori_res_folder)
    for f_name in npy_files:
        if "_b" in f_name:
            seq_name = f_name.split("_b")[0]
        else:
            seq_name = f_name.replace(".npy", "")

        if seq_name not in seq_dict:
            seq_dict[seq_name] = []
        seq_dict[seq_name].append(f_name)

    for seq_k in seq_dict:
        f_names = seq_dict[seq_k]
        num_f = len(f_names)

        if "_b" in f_names[0]:
            seq_npy_data = []
            for f_idx in range(num_f):
                tmp_name = seq_k + "_b_" + str(f_idx) + ".npy"
                npy_path = os.path.join(ori_res_folder, tmp_name)
                curr_npy_data = np.load(npy_path)

                seq_npy_data.append(curr_npy_data)

            seq_npy_data = np.vstack(seq_npy_data)

            dest_npy_path = os.path.join(dest_res_folder, seq_k+".npy")
            np.save(dest_npy_path, seq_npy_data) 
        else:
            ori_npy_path = os.path.join(ori_res_folder, seq_k+".npy")
            dest_npy_path = os.path.join(dest_res_folder, seq_k+".npy")
            shutil.copy(ori_npy_path, dest_npy_path)
      
def insert_data():
    data_folder = "/viscam/u/jiamanli/datasets/kin-poly/MoCapData/features"
    ori_data_path = os.path.join(data_folder, "mocap_annotations_ori.p")
    dest_data_path = os.path.join(data_folder, "mocap_annotations.p")

    of_feats_folder = "/viscam/u/jiamanli/datasets/kin-poly/ReliveDatasetRelease/fpv_raft_of_feats"
    droid_slam_folder = "/viscam/u/jiamanli/datasets/kin-poly/OriData/droid_slam_res"

    data = joblib.load(ori_data_path)
    data_dict = {}
    for seq_name in data:
        seq_data = data[seq_name]

        ori_of_files = seq_data['of_files']
        dest_of_files = []
        for of_name in ori_of_files:
            of_folder_name = of_name.split("/")[-2] 
            of_npy_name = of_name.split("/")[-1]

            dest_of_path = os.path.join(of_feats_folder, of_folder_name, of_npy_name)
            dest_of_files.append(dest_of_path)

       
        seq_data['of_files'] = dest_of_files 
        
        data_dict[seq_name] = seq_data 
        # import pdb 
        # pdb.set_trace()
        # break 
    # import pdb 
    # pdb.set_trace() 

    joblib.dump(data_dict, dest_data_path)

def divide_data_blocks(max_timesteps):
    droid_slam_res_folder = "/viscam/u/jiamanli/datasets/kin-poly/OriData/droid_slam_res_ori_selected"
    dest_droid_slam_res_folder = "/viscam/u/jiamanli/datasets/kin-poly/OriData/droid_slam_res_blocks_max_" + str(max_timesteps)
    if not os.path.exists(dest_droid_slam_res_folder):
        os.makedirs(dest_droid_slam_res_folder)

    data_folder = "/viscam/u/jiamanli/datasets/kin-poly/MoCapData"
    dest_data_folder = "/viscam/u/jiamanli/datasets/kin-poly/MoCapData_blocks" + "_max_" + str(max_timesteps)

    dest_data_mocap_folder = os.path.join(dest_data_folder, "features")
    if not os.path.exists(dest_data_mocap_folder):
        os.makedirs(dest_data_mocap_folder)

    dest_data_path = os.path.join(dest_data_mocap_folder, "mocap_annotations.p")

    # Read meta file 
    meta_folder = os.path.join(data_folder, "meta")
    meta_path = os.path.join(meta_folder, "mocap_meta.yml")

    meta_list = yaml.load(open(meta_path, 'r'), Loader=yaml.FullLoader)
    
    dest_meta_folder = os.path.join(dest_data_folder, "meta")
    if not os.path.exists(dest_meta_folder):
        os.makedirs(dest_meta_folder)
    dest_meta_path = os.path.join(dest_meta_folder, "mocap_meta.yml")

    video_mocap_sync_dict = meta_list['video_mocap_sync'] 

    data_path = os.path.join(data_folder, "features", "mocap_annotations.p")

    new_data_dict = {} 

    ori_data = joblib.load(data_path) 
    for seq_name in ori_data:
        seq_data = ori_data[seq_name]

        curr_qpos = seq_data['qpos']
        curr_of_files_list = seq_data['of_files'] 

        # Make sure each of_feats path exists
        missing_of = False 
        for tmp_of_path in curr_of_files_list:
            if not os.path.exists(tmp_of_path):
                print("Of missing, Skip sequence:{0}".format(seq_name))
                missing_of = True 
                break 

        missing_slam = False 

        curr_droid_slam_path = os.path.join(droid_slam_res_folder, seq_name.split("-")[1]+".npy")
        if not os.path.exists(curr_droid_slam_path):
            missing_slam = True 

        if not missing_of and not missing_slam:
            curr_slam_data = np.load(curr_droid_slam_path)[::2] # Include all the images, including those without gt motion 

            sync_list = video_mocap_sync_dict[seq_name]
            im_offset, lb, ub = sync_list

            curr_slam_data = curr_slam_data[lb + im_offset: ub + im_offset]

            if curr_slam_data.shape[0] != curr_qpos.shape[0]:
                curr_qpos = curr_qpos[:curr_slam_data.shape[0]]

            assert curr_slam_data.shape[0] == curr_qpos.shape[0]

            # Divide into blocks if the whole sequence length is too long 
            curr_seq_len = curr_qpos.shape[0]
            num_blocks = curr_seq_len // max_timesteps + 1
            for b_idx in range(num_blocks):
                curr_dest_seq_name = seq_name + "_b_" + str(b_idx)

                # Save corresponding slam res 
                curr_dest_slam_path = os.path.join(dest_droid_slam_res_folder, \
                    curr_dest_seq_name.split("-")[1]+".npy")
                if curr_slam_data[b_idx*max_timesteps:(b_idx+1)*max_timesteps].shape[0] != max_timesteps:
                    break 
                np.save(curr_dest_slam_path, curr_slam_data[b_idx*max_timesteps:(b_idx+1)*max_timesteps])

                # dict_keys(['qpos', 'meta', 'wbquat', \
                # 'wbpos', 'ee_pos', 'ee_wpos', 'bquat', \
                # 'com', 'body_com', 'head_pose', 'rq_rmh', \
                # 'qvel', 'rlinv', 'rlinv_local', 'rangv', \
                # 'bangvel', 'len', 'height_lb', 'head_height_lb', \
                # 'of_files', 'action', 'obj_pose', 'action_one_hot', \
                # 'head_vels', 'obj_head_relative_poses', 'obj_root_relative_poses'])

                new_data_dict[curr_dest_seq_name] = {}

                for k in seq_data:
                    if "meta" not in k and "len" not in k:
                        if "height" in k:
                            new_data_dict[curr_dest_seq_name][k] = seq_data[k]
                        elif "of_file" in k:
                            new_data_dict[curr_dest_seq_name][k] = seq_data[k][b_idx*max_timesteps:(b_idx+1)*max_timesteps]
                        elif "action" in k:
                            new_data_dict[curr_dest_seq_name][k] = seq_data[k]
                        else:
                            new_data_dict[curr_dest_seq_name][k] = seq_data[k][b_idx*max_timesteps:(b_idx+1)*max_timesteps]
                            # if seq_data[k].shape[0] == curr_seq_len:
                            #     new_data_dict[curr_dest_seq_name][k] = seq_data[k][b_idx:(b_idx+1)*max_timesteps]
                            # else:
                            #     new_data_dict[curr_dest_seq_name][k] = seq_data[k]

                
                # if new_data_dict[curr_dest_seq_name]['head_pose'].shape[0] != \
                # curr_slam_data[b_idx*max_timesteps:(b_idx+1)*max_timesteps].shape[0]:
                #     import pdb 
                #     pdb.set_trace() 
                # assert new_data_dict[curr_dest_seq_name]['head_pose'].shape[0] == \
                # curr_slam_data[b_idx*max_timesteps:(b_idx+1)*max_timesteps].shape[0]

                new_data_dict[curr_dest_seq_name]['meta'] = seq_data['meta']
                new_data_dict[curr_dest_seq_name]['meta']['seq_name'] = curr_dest_seq_name
                new_data_dict[curr_dest_seq_name]['len'] = new_data_dict[curr_dest_seq_name]['qpos'].shape[0]

    joblib.dump(new_data_dict, dest_data_path)
           
    # Update meta file
    new_meta_dict = {}
    new_meta_dict['action_type'] = {}
    for tmp_seq_name in new_data_dict:
        new_meta_dict['action_type'][tmp_seq_name] = "none"

    new_meta_dict['capture'] = {}
    new_meta_dict['capture'] = meta_list['capture']
    new_meta_dict['object'] = {}
    new_meta_dict['object'] = meta_list['object']
    new_meta_dict['offset_z'] = meta_list['offset_z']
    # Set train and test 
    new_meta_dict['train'] = []
    new_meta_dict['test'] = []
    for tmp_seq_name in new_data_dict:
        new_meta_dict['train'].append(tmp_seq_name)
        new_meta_dict['test'].append(tmp_seq_name)
               
    # Set video mocap sync, not used now since currently haven't processed image features 
    new_meta_dict['video_mocap_sync'] = {} 
    for tmp_seq_name in new_data_dict:
        new_meta_dict['video_mocap_sync'][tmp_seq_name] = [0, 0, \
            new_data_dict[tmp_seq_name]['qpos'].shape[0]]

    documents = yaml.dump(new_meta_dict, open(dest_meta_path, 'w'))


if __name__ == "__main__":
    # Convert the data to the same organized folder as original kinpoly data 
    # merge_droid_slam_res()

    # insert_data() 

    divide_data_blocks(max_timesteps=150)
  
    # 10 seconds max: 300 timesteps 
    divide_data_blocks(max_timesteps=300)

    # # 20 seconds max: 600 timesteps 
    divide_data_blocks(max_timesteps=600) 

'''
elta_head = np.repeat(np.array([[0, 0.2, 0.2]]), head_pose.shape[0], axis = 0)
        delta_head = quat_mul_vec_batch(head_pose[:, 3:7], delta_head)
        head_pose[:, :3] -= delta_head
        head_pose[:, 2] +=   throat_level - head_pose[0, 2]# Adjusting for the head height
'''

