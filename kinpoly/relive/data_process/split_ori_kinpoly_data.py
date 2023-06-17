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


def split_ori_seq():
    droid_slam_res_folder = "/viscam/u/jiamanli/datasets/kin-poly/OriData/droid_slam_res_ori_selected"

    data_folder = "/viscam/u/jiamanli/datasets/kin-poly/MoCapData"
    dest_data_folder = "/viscam/u/jiamanli/datasets/kin-poly/MoCapData_split_by_slam_res"

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

        sync_list = video_mocap_sync_dict[seq_name]
        im_offset, lb, ub = sync_list

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
                np.save(curr_dest_slam_path, curr_slam_data[b_idx:(b_idx+1)*max_timesteps])

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
                            new_data_dict[curr_dest_seq_name][k] = seq_data[k][b_idx:(b_idx+1)*max_timesteps]
                        elif "action" in k:
                            new_data_dict[curr_dest_seq_name][k] = seq_data[k]
                        else:
                            if seq_data[k].shape[0] == curr_seq_len:
                                new_data_dict[curr_dest_seq_name][k] = seq_data[k][b_idx:(b_idx+1)*max_timesteps]
                            else:
                                new_data_dict[curr_dest_seq_name][k] = seq_data[k]

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
   split_ori_seq() 

