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
from copycat.utils.tools import get_expert
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

def get_head_vel(head_pose, dt = 1/30):
    # get head velocity 
    head_vels = []
    head_qpos = head_pose[0]

    for i in range(head_pose.shape[0] - 1):
        curr_qpos = head_pose[i, :]
        next_qpos = head_pose[i + 1, :] 
        v = (next_qpos[:3] - curr_qpos[:3]) / dt
        v = transform_vec(v, curr_qpos[3:7], 'heading') # get velocity within the body frame
        
        qrel = quaternion_multiply(next_qpos[3:7], quaternion_inverse(curr_qpos[3:7]))
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

def get_root_relative_head(root_poses, head_poses):
    res_root_poses = []
    
    for idx in range(head_poses.shape[0]):
        head_qpos = head_poses[idx]
        root_qpos = root_poses[idx]
        
        head_pos = head_qpos[:3]
        head_rot = head_qpos[3:7]
        q_heading = get_heading_q(head_rot).copy()
        obs = []

        root_pos  = root_qpos[:3].copy()
        diff = root_pos - head_pos
        diff_loc = transform_vec(diff, head_rot, "heading")

        root_quat = root_qpos[3:7].copy()
        root_quat_local = quaternion_multiply(quaternion_inverse(head_rot), root_quat) # ???? Should it be flipped?
        axis, angle = rotation_from_quaternion(root_quat_local, True) 
        
        if angle > np.pi: # -180 < angle < 180
            angle -= 2 * np.pi # 
        elif angle < -np.pi:
            angle += 2 * np.pi
        
        rv = axis * angle 
        rv = transform_vec(rv, head_rot, 'root') # root 2 head diff in head's frame

        root_pose = np.concatenate((diff_loc, rv))
        res_root_poses.append(root_pose)

    res_root_poses = np.array(res_root_poses)
    return res_root_poses

def root_from_relative_head(root_relative, head_poses):
    assert(root_relative.shape[0] == head_poses.shape[0])
    root_poses = []
    for idx in range(root_relative.shape[0]):
        head_pos = head_poses[idx][:3]
        head_rot = head_poses[idx][3:7]
        q_heading = get_heading_q(head_rot).copy()

        root_pos_delta = root_relative[idx][:3]
        root_rot_delta = root_relative[idx][3:]

        root_pos = quat_mul_vec(q_heading, root_pos_delta) + head_pos
        root_rot_delta  = quat_mul_vec(head_rot, root_rot_delta)
        root_rot = quaternion_multiply(head_rot, quat_from_expmap(root_rot_delta))
        root_pose = np.hstack([root_pos, root_rot])
        root_poses.append(root_pose)
    return np.array(root_poses)

def get_obj_relative_pose(obj_poses, ref_poses, num_objs = 1):
    # get object pose relative to the head
    res_obj_poses = []
    
    for idx in range(ref_poses.shape[0]):
        ref_qpos = ref_poses[idx]
        obj_qpos = obj_poses[idx]
        
        ref_pos = ref_qpos[:3]
        ref_rot = ref_qpos[3:7]
        q_heading = get_heading_q(ref_rot).copy()
        obs = []
        for oidx in range(num_objs):
            obj_pos  = obj_qpos[oidx*7:oidx*7+3].copy()
            diff = obj_pos - ref_pos
            diff_loc = transform_vec(diff, ref_rot, "heading")

            obj_quat = obj_qpos[oidx*7+3:oidx*7+7].copy()
            obj_quat_local = quaternion_multiply(quaternion_inverse(q_heading), obj_quat)
            obj_pose = np.concatenate((diff_loc, obj_quat_local))
            obs.append(obj_pose)

        res_obj_poses.append(np.concatenate(obs))
    
    res_obj_poses = np.array(res_obj_poses)
    return res_obj_poses

def post_process_expert(expert, obj_pose, num_objs = 1):
    head_pose = expert['head_pose'] 
    orig_traj = expert['qpos']
    root_pose = orig_traj[:, :7]
    head_vels = get_head_vel(head_pose)
    obj_relative_head = get_obj_relative_pose(obj_pose, head_pose, num_objs = num_objs)
    obj_relative_root = get_obj_relative_pose(obj_pose, root_pose, num_objs = num_objs)
    root_relative_2_head = get_root_relative_head(root_pose, head_pose)
    expert['head_vels'] = head_vels
    expert['obj_head_relative_poses'] = obj_relative_head
    expert['obj_root_relative_poses'] = obj_relative_root
    # root_recover = root_from_relative_head(root_relative_2_head, head_pose)
    # print(np.abs(np.sum(root_recover - root_pose)))
    return expert

def string_to_one_hot(all_classes, class_name):
    hot = np.zeros(len(all_classes))
    if class_name in all_classes:
        index = all_classes.index(class_name)
        hot[index] = 1
        
    return hot[None, ]

def prep_smpl_to_qpos_expert():
    # Initialize config, humanoid simulator 
    cc_cfg = CC_Config("copycat", "/viscam/u/jiamanli/github/egomotion/kin-poly", create_dirs=False)
    # cc_cfg = CC_Config("copycat", "/Users/jiamanli/github/egomotion/kin-poly", create_dirs=False)

    # cc_cfg.data_specs['test_file_path'] = "/Users/jiamanli/github/kin-polysample_data/h36m_train_no_sit_30_qpos.pkl"
    cc_cfg.data_specs['test_file_path'] = "/viscam/u/jiamanli/github/egomotion/kin-poly/sample_data/h36m_test.pkl"
    # cc_cfg.data_specs['test_file_path'] = "/Users/jiamanli/github/egomotion/kin-poly/sample_data/h36m_test.pkl"
    cc_cfg.data_specs['neutral_path'] = "/viscam/u/jiamanli/github/egomotion/kin-poly/sample_data/standing_neutral.pkl"

    cc_cfg.mujoco_model_file = "/viscam/u/jiamanli/github/egomotion/kin-poly/assets/mujoco_models/humanoid_smpl_neutral_mesh_all_step.xml"
    
    data_loader = DatasetSMPLObj(cc_cfg.data_specs, data_mode="test")
    init_expert = data_loader.sample_seq()
    env = CC_HumanoidEnv(cc_cfg, init_expert = init_expert, data_specs = cc_cfg.data_specs, mode="test")

    model_file = f'/viscam/u/jiamanli/github/egomotion/kin-poly/assets/mujoco_models/humanoid_smpl_neutral_mesh.xml'
    humanoid_model =load_model_from_path(model_file)

    # Convert the data format 
    smpl_seq_data = {} 
    # smpl_output_folder = "/viscam/u/jiamanli/datasets/amass_processed_for_kinpoly"
    # smpl_output_folder = "/viscam/u/jiamanli/datasets/amass_processed_locomotion_for_kinpoly"
    smpl_output_folder = "/viscam/u/jiamanli/datasets/gimo_processed/gimo_motion_for_kinpoly"
    if not os.path.exists(smpl_output_folder):
        os.makedirs(smpl_output_folder)
    smpl_output_filename = os.path.join(smpl_output_folder, "gimo_kinpoly_motion.p")

    # ori_amass_root_folder = "/viscam/u/jiamanli/datasets/amass_processed" 
    # ori_amass_root_folder = "/viscam/u/jiamanli/datasets/amass_processed_locomotion" 
    # ori_amass_root_folder = "/viscam/u/jiamanli/datasets/amass_processed_all_locomotion" 
    ori_amass_root_folder = "/viscam/u/jiamanli/datasets/gimo_processed/smplx_npz_processed"
    ori_flow_folder = ori_amass_root_folder.replace("smplx_npz_processed", "raft_of_feats")

    total_frame_cnt = 0
    counter = 0

    subset_folders = os.listdir(ori_amass_root_folder)
    for subset_name in subset_folders:    
        subset_folder_path = os.path.join(ori_amass_root_folder, subset_name)
        
        npz_files = os.listdir(subset_folder_path)
        for npz_name in npz_files:
            # Add of path 
            flow_folder =os.path.join(ori_flow_folder, subset_name, \
                    "_".join(npz_name.split("_")[:3]))
            if os.path.exists(flow_folder):
                of_files = os.listdir(flow_folder)
                of_files.sort() 
                of_path_list = []
                for of_name in of_files:
                    if ".npy" in of_name and ".png" not in of_name: 
                        of_path = os.path.join(flow_folder, of_name)
                        of_path_list.append(of_path)

                motion_path = os.path.join(subset_folder_path, npz_name) 
                ori_npz_data = np.load(motion_path)

                amass_root_pose = ori_npz_data['root_orient'] # T X 3 
                amass_pose = ori_npz_data['pose_body'] # T X 63 
                amass_trans = ori_npz_data['trans'] # T X 3 
                betas = ori_npz_data['betas'] # 10
                gender = "male"

                seq_length = amass_pose.shape[0]
                total_frame_cnt += seq_length 
                
                pose_aa = torch.cat((torch.tensor(amass_root_pose), torch.tensor(amass_pose), torch.zeros(seq_length, 6)), dim=-1) # T X 72       
                curr_trans = amass_trans
                # pose_seq_6d = convert_aa_to_orth6d(pose_aa).reshape(-1, 144)
                qpos = smpl_to_qpose(pose = pose_aa, model = humanoid_model, trans = curr_trans)
                
                dest_seq_name = subset_name+"-"+npz_name.replace(".npz", "")

                expert_meta = {
                    "cyclic": False,
                    "seq_name": dest_seq_name
                }
                expert_res = get_expert(qpos, expert_meta, env)
                expert_res['action'] = "none"
                num_frames = expert_res['qpos'].shape[0]
                
                expert_res['obj_pose'] = np.repeat(np.array([0,0,0,1,0,0,0])[None, ], num_frames, axis=0).astype(np.float)
                expert_res['action_one_hot'] = np.repeat(np.array([0,0,0,0])[None, ], num_frames, axis=0).astype(np.float)
                expert_res = post_process_expert(expert_res, expert_res['obj_pose'], int(expert_res['obj_pose'].shape[1]/7))

                expert_res['of_files'] = of_path_list
                if not expert_res is None:
                    smpl_seq_data[dest_seq_name] = {
                        "pose_aa": pose_aa.numpy(),
                        # "pose_6d":pose_seq_6d.numpy(),
                        "qpos": qpos,
                        'trans': curr_trans,
                        'beta': betas[:10],
                        "seq_name": dest_seq_name,
                        "gender": gender,
                        "expert": expert_res
                    }

                counter += 1
   
    print("Total number of sequences:{0}".format(len(smpl_seq_data)))
    print("Total number of frames:{0}".format(total_frame_cnt))

    joblib.dump(smpl_seq_data, open(smpl_output_filename, "wb"))

def reorganize_data(ori_data_file, dest_data_folder):
    if not os.path.exists(dest_data_folder):
        os.makedirs(dest_data_folder)

    data_dict = joblib.load(ori_data_file)

    anno_folder = os.path.join(dest_data_folder, "features")
    if not os.path.exists(anno_folder):
        os.makedirs(anno_folder)
    
    anno_path = os.path.join(anno_folder, "mocap_annotations.p")
    
    meta_folder = os.path.join(dest_data_folder, "meta")
    if not os.path.exists(meta_folder):
        os.makedirs(meta_folder)
    meta_path = os.path.join(meta_folder, "mocap_meta.yml")

    # Prepare annotation.p
    new_data_dict = {}
    for seq_name in data_dict:
        new_data_dict[seq_name] = {}
        expert_data = data_dict[seq_name]['expert']

        new_data_dict[seq_name] = expert_data
        # break

    joblib.dump(new_data_dict, open(anno_path, 'wb'))

    # Write meta 
    template_meta_file = "/viscam/u/jiamanli/datasets/kin-poly/MoCapData/meta/mocap_meta.yml"
    meta_list = yaml.load(open(template_meta_file, 'r'), Loader=yaml.FullLoader)

    new_meta_dict = {}
    # Set all the sequences' action type to None.
    new_meta_dict['action_type'] = {}
    for seq_name in data_dict:
        new_meta_dict['action_type'][seq_name] = "none"
    new_meta_dict['capture'] = {}
    new_meta_dict['capture'] = meta_list['capture']
    new_meta_dict['object'] = {}
    new_meta_dict['object'] = meta_list['object']
    new_meta_dict['offset_z'] = meta_list['offset_z']
    # Set train and test 
    new_meta_dict['train'] = []
    new_meta_dict['val'] = []
    new_meta_dict['test'] = []
    for seq_name in data_dict:
        new_meta_dict['train'].append(seq_name)
        new_meta_dict['val'].append(seq_name)
        new_meta_dict['test'].append(seq_name)
    # Set video mocap sync, not used now since currently haven't processed image features 
    new_meta_dict['video_mocap_sync'] = {} 
    for seq_name in data_dict:
        new_meta_dict['video_mocap_sync'][seq_name] = [0, 0, data_dict[seq_name]['trans'].shape[0]]

    documents = yaml.dump(new_meta_dict, open(meta_path, 'w'))

if __name__ == "__main__":
    prep_smpl_to_qpos_expert()

    # Total number of sequences:215
    # Total number of frames:60217

    # Convert the data to the same organized folder as original kinpoly data 
    ori_data_file = "/viscam/u/jiamanli/datasets/gimo_processed/gimo_motion_for_kinpoly/gimo_kinpoly_motion.p"
    dest_data_folder = "/viscam/u/jiamanli/datasets/gimo_processed/gimo_motion_for_kinpoly/MoCapData"
    reorganize_data(ori_data_file, dest_data_folder)

'''
elta_head = np.repeat(np.array([[0, 0.2, 0.2]]), head_pose.shape[0], axis = 0)
        delta_head = quat_mul_vec_batch(head_pose[:, 3:7], delta_head)
        head_pose[:, :3] -= delta_head
        head_pose[:, 2] +=   throat_level - head_pose[0, 2]# Adjusting for the head height
'''

