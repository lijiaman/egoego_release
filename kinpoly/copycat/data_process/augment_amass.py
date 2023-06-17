import os
import sys
import pdb
sys.path.append(os.getcwd())

import numpy as np
import glob
import pickle as pk 
import joblib
import torch 

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

np.random.seed(1)
left_right_idx = [ 0,  2,  1,  3,  5,  4,  6,  8,  7,  9, 11, 10, 12, 14, 13, 15, 17,16, 19, 18, 21, 20, 23, 22]

def left_to_rigth_euler(pose_euler):
    pose_euler[:,:, 0] = pose_euler[:,:,0] *  -1
    pose_euler[:,:,2] = pose_euler[:,:,2] * -1
    pose_euler = pose_euler[:,left_right_idx,:]
    return pose_euler

def flip_smpl(pose, trans = None):
    '''
        Pose input batch * 72
    '''
    curr_spose = sRot.from_rotvec(pose.reshape(-1, 3))
    curr_spose_euler = curr_spose.as_euler('ZXY', degrees=False).reshape(pose.shape[0], 24, 3)
    curr_spose_euler = left_to_rigth_euler(curr_spose_euler)
    curr_spose_rot = sRot.from_euler("ZXY", curr_spose_euler.reshape(-1, 3), degrees = False)
    curr_spose_aa = curr_spose_rot.as_rotvec().reshape(pose.shape[0], 24, 3)
    if trans != None:
        pass
        # target_root_mat = curr_spose.as_matrix().reshape(pose.shape[0], 24, 3, 3)[:, 0]
        # root_mat = curr_spose_rot.as_matrix().reshape(pose.shape[0], 24, 3, 3)[:, 0]
        # apply_mat = np.matmul(target_root_mat[0], np.linalg.inv(root_mat[0]))

    return curr_spose_aa.reshape(-1, 72)


def sample_random_hemisphere_root():
    rot = np.random.random() * np.pi * 2
    pitch =  np.random.random() * np.pi/3 + np.pi
    r = sRot.from_rotvec([pitch, 0, 0])
    r2 = sRot.from_rotvec([0, rot, 0])
    root_vec = (r * r2).as_rotvec()
    return root_vec

def sample_seq_length(seq, tran, seq_length = 150):
    if seq_length != -1:
        num_possible_seqs = seq.shape[0] // seq_length
        max_seq = seq.shape[0]

        start_idx = np.random.randint(0, 10)
        start_points = [max(0, max_seq - (seq_length + start_idx))]

        for i in range(1, num_possible_seqs - 1):
            start_points.append(i * seq_length + np.random.randint(-10, 10))

        if num_possible_seqs >= 2:
            start_points.append(max_seq - seq_length - np.random.randint(0, 10))

        seqs = [seq[i:(i + seq_length)] for i in start_points]
        trans = [tran[i:(i + seq_length)] for i in start_points]
    else:
        seqs = [seq]
        trans = [tran]
        start_points = []
    return seqs, trans, start_points

def get_random_shape(batch_size):
    shape_params = torch.rand(1, 10).repeat(batch_size, 1)
    s_id = torch.tensor(np.random.normal(scale = 1.5, size = (3)))
    shape_params[:,:3] = s_id
    return shape_params


def fix_height(expert, expert_meta, env):
    wbpos = expert['wbpos']
    wbpos = wbpos.reshape(wbpos.shape[0], 24, 3)
    begin_feet = min(wbpos[0, 4, 2],  wbpos[0, 8, 2])
    begin_root = wbpos[0, 0, 2]
    if begin_root < 0.3 and begin_feet > -0.1:
        print(f"Crawling: {expert_meta['seq_name']}")
        return expert
    
    begin_feet -= 0.015 # Hypter parameter to tune
    qpos =expert['qpos']
    qpos[:, 2] -= begin_feet
    new_expert = get_expert(qpos, expert_meta, env)
    new_wpos = new_expert['wbpos']
    new_wpos = new_wpos.reshape(new_wpos.shape[0], 24, 3)
    ground_pene = min(np.min(new_wpos[:, 4, 2]),  np.min(new_wpos[:, 8, 2]))
    if ground_pene < -0.15:
        print(f"{expert_meta['seq_name']} negative sequence invalid for copycat: {ground_pene}")
        return None
    return new_expert
        


def process_qpos_list(qpos_list):
    amass_res = {}
    pbar = tqdm(qpos_list)
    for (k, v) in pbar:
        pbar.set_description(k)
        amass_pose = v['pose_aa']
        amass_trans = v['trans']
        betas = v['beta']
        gender = v['gender']
        bound = amass_pose.shape[0]

        if k in amass_occlusion:
            issue = amass_occlusion[k]['issue']
            if issue == "sitting" or issue == "airborne" :
                bound = amass_occlusion[k]['idxes'][0] # This bounded is calucaled assuming 30 FPS.....
                if bound < 10:
                    print("bound too small", k, bound)
                    continue
            else:
                # print("issue irrecoverable", k, issue)
                continue

        seq_length = amass_pose.shape[0]
        if seq_length < 10:
            continue

        pose_aa = torch.tensor(amass_pose)[:bound] # After sampling the bound
        curr_trans = amass_trans[:bound]           # After sampling the bound

        seq_name =  k 
        expert_meta = {
                "cyclic": False,
                "seq_name": seq_name
            }

        if bound == seq_length:
            expert_res = v['expert']
            pose_seq_6d = v['pose_6d']
            qpos = v['qpos']
            expert_res = fix_height(expert_res, expert_meta, env)
        else:
            pose_seq_6d = convert_aa_to_orth6d(torch.tensor(pose_aa)).reshape(-1, 144).numpy()
            qpos = smpl_to_qpose(pose = pose_aa, model = humanoid_model, trans = curr_trans)
            
            expert_res = get_expert(qpos, expert_meta, env)
            expert_res = fix_height(expert_res, expert_meta, env)
            pose_aa = pose_aa.numpy()
        


        if not expert_res is None:
            amass_res[seq_name] = {
                "pose_aa": pose_aa,
                "pose_6d":pose_seq_6d,
                "qpos": expert_res['qpos'], # need to use the updated qpos from fix_height
                'trans': expert_res['qpos'][:3],
                'beta': betas[:10],
                "seq_name": seq_name,
                "gender": gender,
                "expert": expert_res
            }
            
    return amass_res

if __name__ == "__main__":
    amass_base = "/insert_directory_here/"
    take_num = "copycat_take4"
    # amass_cls_data = pk.load(open(os.path.join(amass_base, "amass_class.pkl"), "rb"))
    amass_seq_data = {}
    seq_length = -1
    cfg = Config("copycat_5", None, create_dirs=False)

    data_loader = DatasetAMASSSingle(cfg.data_specs, data_mode="test")
    init_expert = data_loader.sample_seq()
    env = HumanoidEnv(cfg, init_expert = init_expert, data_specs = cfg.data_specs, mode="test")

    # target_frs = [20,30,40] # target framerate
    target_frs = [30] # target framerate
    video_annot = {}
    counter = 0 
    seq_counter = 0
    amass_db = joblib.load("/insert_directory_here/amass_qpos_30.pkl")
    amass_occlusion = joblib.load("/insert_directory_here/amass_copycat_occlusion.pkl")

    model_file = f'assets/mujoco_models/humanoid_smpl_neutral_mesh.xml'
    humanoid_model =load_model_from_path(model_file)
    qpos_list = list(amass_db.items())
    np.random.seed(0)
    np.random.shuffle(qpos_list)


    from multiprocessing import Pool
    num_jobs = 20
    jobs = qpos_list
    chunk = np.ceil(len(jobs)/num_jobs).astype(int)
    jobs= [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    job_args = [(jobs[i],) for i in range(len(jobs))]
    print(len(job_args))

    try:
        pool = Pool(num_jobs)   # multi-processing
        job_res = pool.starmap(process_qpos_list, job_args)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

    [amass_seq_data.update(j) for j in job_res]
    

    # amass_output_file_name = "/insert_directory_here/amass_{}_test.pkl".format(take_num)
    amass_output_file_name = "/insert_directory_here/amass_{}.pkl".format(take_num)
    print(amass_output_file_name, len(amass_seq_data))
    joblib.dump(amass_seq_data, open(amass_output_file_name, "wb"))
