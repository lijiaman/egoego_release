import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import sys
import time
import pickle
import glob
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from relive.utils import *
from tqdm import tqdm
import joblib
import yaml

from relive.utils.egomimic_config import Config as EgoConfig
from relive.utils import get_qvel_fd, de_heading
from relive.utils.torch_humanoid import Humanoid
from relive.utils.transformation import quaternion_multiply, quaternion_inverse,  rotation_from_quaternion
from relive.utils.transform_utils import (
    convert_6d_to_mat, compute_orth6d_from_rotation_matrix, convert_quat_to_6d
)



def get_expert(expert_qpos, lb, ub, cfg, env, obj_name = "chair"):
    off_obj_qpos = env.off_obj_qpos
    off_obj_qvel = env.off_obj_qvel
    expert = {'qpos': expert_qpos}
    feat_keys = {'qvel', 'rlinv', 'rlinv_local', 'rangv', 'hvel','hvel_local', 'rq_rmh',
                 'com', 'head_pos', 'head_info', 'obs', 'ee_pos', 'ee_wpos', 'bquat', 'wbquat', 'wbpos', 'bangvel', 'heading_q'}
    for key in feat_keys:
        expert[key] = []

    for i in range(expert_qpos.shape[0]):
        qpos = expert_qpos[i]
        # qpos[off_obj_qpos+2] += cfg.offset_z # we don't have offset_z anymore
        # remove noisy hand data
        qpos[slice(*env.body_qposaddr['LeftHand'])]  = 0.0
        qpos[slice(*env.body_qposaddr['RightHand'])] = 0.0
        

        env.data.qpos[:] = qpos
        env.sim.forward()
        # if args.render:
            # env.render()
        rq_rmh = de_heading(qpos[off_obj_qpos+3:off_obj_qpos+7])
        obs = env.get_obs()
        ee_pos = env.get_ee_pos(env.cfg.obs_coord)
        ee_wpos = env.get_ee_pos()
        bpos = env.get_body_pos()
        bquat = env.get_body_quat()
        wbquat = env.get_world_body_quat()
        com = env.get_com()
        heading_q = get_heading_q(qpos[off_obj_qpos+3:off_obj_qpos+7])
        head_pos = env.get_head()[:3].copy()
        head_info = env.get_head().copy()
        if i > 0:
            prev_head_info = expert['head_info'][i - 1]
            prev_qpos = expert_qpos[i - 1]
            # ignore chair offset to compute humanoid velocity
            qvel = get_qvel_fd(prev_qpos[off_obj_qpos:], qpos[off_obj_qpos:], env.dt)
            rlinv = qvel[:3].copy()
            rlinv_local = transform_vec(qvel[:3].copy(), qpos[off_obj_qpos+3:off_obj_qpos+7], env.cfg.obs_coord)
            rangv = qvel[3:6].copy()
            qvel_obj = np.zeros(off_obj_qvel)
            qvel = np.concatenate([qvel_obj, qvel])
            hpvel = (head_info[:3] - prev_head_info[:3]) / env.dt
            hpvel_local = transform_vec(hpvel.copy(), prev_head_info[3:], 'heading')

            hqvel = get_angvel_fd(prev_head_info[3:], head_info[3:], env.dt)

            expert['qvel'].append(qvel)
            expert['rlinv'].append(rlinv)
            expert['rlinv_local'].append(rlinv_local)
            expert['rangv'].append(rangv)
            expert['hvel'].append(np.concatenate((np.array(hpvel), np.array(hqvel))))
            expert['hvel_local'].append(hpvel_local)

        expert['obs'].append(obs)
        expert['ee_pos'].append(ee_pos)
        expert['ee_wpos'].append(ee_wpos)
        expert['bquat'].append(bquat)
        expert['wbquat'].append(wbquat)
        expert['wbpos'].append(bpos)
        expert['com'].append(com)
        expert['head_pos'].append(head_pos)
        expert['head_info'].append(head_info)
        expert['rq_rmh'].append(rq_rmh)
        expert['heading_q'].append(heading_q)

    expert['qvel'].append(expert['qvel'][-1].copy())
    expert['rlinv'].append(expert['rlinv'][-1].copy())
    expert['rlinv_local'].append(expert['rlinv_local'][-1].copy())
    expert['rangv'].append(expert['rangv'][-1].copy())
    expert['hvel'].append(expert['hvel'][-1].copy())
    expert['hvel_local'].append(expert['hvel_local'][-1].copy())

    # get expert body quaternions
    for i in range(expert_qpos.shape[0]):
        if i > 0:
            bangvel = get_angvel_fd(expert['bquat'][i - 1], expert['bquat'][i], env.dt)
            expert['bangvel'].append(bangvel)
    expert['bangvel'].append(expert['bangvel'][-1].copy())

    expert['qpos'] = expert['qpos'][lb:ub, :]
    for key in feat_keys:
        expert[key] = np.vstack(expert[key][lb:ub])
    expert['len'] = expert['qpos'].shape[0]
    expert['height_lb'] = expert['qpos'][:, off_obj_qpos+2].min()
    expert['head_height_lb'] = expert['head_pos'][:, 2].min()


    expert = post_process_expert(expert, int(off_obj_qpos/7))
    return expert


def get_head_vel(head_pose, dt = 1/30):
    # get head velocity  in body frame!!!
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

def get_obj_relative_pose(obj_poses, head_poses, num_objs = 1):
    # get object pose relative to the head
    res_obj_poses = []
    
    for idx in range(head_poses.shape[0]):
        head_qpos = head_poses[idx]
        obj_qpos = obj_poses[idx]
        
        head_pos = head_qpos[:3]
        head_rot = head_qpos[3:7]
        q_heading = get_heading_q(head_rot).copy()
        obs = []
        for oidx in range(num_objs):
            obj_pos  = obj_qpos[oidx*7:oidx*7+3].copy()
            diff = obj_pos - head_pos
            diff_loc = transform_vec(diff, head_rot, "heading")

            obj_quat = obj_qpos[oidx*7+3:oidx*7+7].copy()
            obj_quat_local = quaternion_multiply(quaternion_inverse(q_heading), obj_quat)
            obj_pose = np.concatenate((diff_loc, obj_quat_local))
            obs.append(obj_pose)

        res_obj_poses.append(np.concatenate(obs))
    
    res_obj_poses = np.array(res_obj_poses)

    return res_obj_poses

def post_process_expert(expert, num_objs = 1):
    head_pose = expert['head_info'] 
    orig_traj = expert['qpos']
    root_pose = orig_traj[:, num_objs * 7:num_objs * 7 + 7 ]
    obj_pose = orig_traj[:, : num_objs * 7 ]

    head_vels = get_head_vel(head_pose)
    obj_relative_poses = get_obj_relative_pose(obj_pose, head_pose, num_objs = num_objs)
    root_relative_2_head = get_root_relative_head(root_pose, head_pose)
    expert['head_vels'] = head_vels
    expert['obj_head_relative_poses'] = obj_relative_poses # only taking the first object's relative position
    expert['root_relative_2_head'] = root_relative_2_head
    # root_recover = root_from_relative_head(root_relative_2_head, head_pose)
    # print(np.abs(np.sum(root_recover - root_pose)))
    return expert




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='all_traj_3')
    parser.add_argument('--render', action='store_true', default=False)
    args = parser.parse_args()


    actions = ["sit", "push", "step", "avoid"]
    meta_id = "meta_all"
    # actions = ["sit"]
    expert_dict = {a:{} for a in actions}
    for action in actions:
        cfg = EgoConfig(action, args.cfg, create_dirs=False)
        env = HumanoidEnv(cfg)
        of_folder = os.path.join(cfg.data_dir, 'fpv_of')
        meta_file = osp.join(cfg.data_dir, "meta", f"{meta_id}.yml") 
        meta = yaml.load(open(meta_file, 'r'), Loader=yaml.FullLoader)
        msync = meta['video_mocap_sync']

        off_obj_qpos = env.off_obj_qpos
        off_obj_qvel = env.off_obj_qvel
        print(f"action: {action}", off_obj_qpos)

        num_sample = 0
        takes = cfg.takes['train'] + cfg.takes['test']
        pbar = tqdm(range(len(takes)))
        for i in pbar:
            take = takes[i]
            traj_path = os.path.join(cfg.data_dir, 'traj_norm', take + "_traj.p") # ZL:use traj_norm
            expert_qpos = pickle.load(open(traj_path, "rb"))
            im_offset, lb, ub = msync[take]
            # lb = 0
            # ub = expert_qpos.shape[0]

            expert = get_expert(expert_qpos, lb, ub, cfg, env = env)
            of_files = np.array(sorted(glob.glob(osp.join(of_folder, take, "*.npy"))))[lb + im_offset: ub + im_offset]
            expert['of_files'] = of_files
            

            expert_dict[action][take] = expert
            num_sample += expert['len']

            pbar.set_description(take)
            # print(take, expert['len'], expert['qvel'].min(), expert['qvel'].max(), expert['head_height_lb'])
            # if args.render:
            #     head_display = expert['qpos'].copy()
            #     head_display[:,7:14] = expert['head_info']
            #     head_display[:,14:] = 0
            #     from relive.utils.egomimic_config import Config as EgoConfig
            #     from relive.envs.visual.humanoid_vis import HumanoidVisEnv
            #     cfg = EgoConfig(action, args.cfg, create_dirs=False)
            #     env = HumanoidEnv(cfg)
            #     for i in range(head_display.shape[0]):
            #         env.data.qpos[:] = head_display[i]
            #         env.sim.forward()
            #         env.render()

    path = osp.join(cfg.data_dir, "features", f"expert_all.p" )
    print(f"{path}")
    joblib.dump(expert_dict, open(path, 'wb'))

    path = osp.join(cfg.data_dir, "features", f"traj_all.p" )
    print(f"{path}")
    joblib.dump(expert_dict, open(path, 'wb'))


