import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
import numpy as np
import math
sys.path.append(os.getcwd())

from relive.utils import *
from relive.utils.transformation import quaternion_from_euler
from mujoco_py import load_model_from_path, MjSim
from relive.envs.common.mjviewer import MjViewer
from relive.mocap.skeleton import Skeleton
from relive.mocap.pose import load_bvh_file, load_obj_bvh_file, interpolated_traj, load_obj_position
import pickle
import glob
import argparse
import cv2

parser = argparse.ArgumentParser()


parser.add_argument('--model-id', type=str, default='1213')
parser.add_argument('--mocap-id', type=str, default='1001')
parser.add_argument('--range', type=int, default=(5, 20))
parser.add_argument('--dt', type=float, default=1/30)
parser.add_argument('--dataset-path', type=str, default='datasets')
parser.add_argument('--action', type=str, default='sit') # action is paired to the object
parser.add_argument('--del-rot', action='store_true', default=False) #Default should be on
args = parser.parse_args()

#object_list = args.object.split(',')

objects = {'sit':['chair'], 'avoid':['Can'], 'push':['box', 'table'], 'step':['step']}
object_list = objects[args.action]

model_file = 'assets/mujoco_models/%s/%s/humanoid.xml' % (args.model_id, args.action)
model = load_model_from_path(model_file)
body_qposaddr = get_body_qposaddr(model)

traj_dir = '%s/traj' % args.dataset_path
traj_offset = args.mocap_id + "_take"



def get_qpos_obj(pose, obj_pose, bone_addr):
    qpos = np.zeros(model.nq)
    for bone_name, ind2 in body_qposaddr.items():
        #print('name: {}, ind2: {}'.format(bone_name, ind2))
        if not bone_name in bone_addr.keys(): ## object
            obj_idx = object_list.index(bone_name)
            trans  = obj_pose[obj_idx][:3]
            #trans[1] += 0.15
            angles = obj_pose[obj_idx][3:]
            if args.del_rot:
                #quat = quaternion_from_euler(0.0, 0.0, angles[2] - math.pi / 2.0, 'rxyz')
                quat = quaternion_from_euler(0.0, 0.0, angles[2], 'rxyz')
            else:
                quat = quaternion_from_euler(angles[0], angles[1], angles[2], 'rxyz')
            qpos[ind2[0]:ind2[0] + 3] = trans
            qpos[ind2[0] + 3:ind2[1]] = quat
            continue
        ind1 = bone_addr[bone_name]
        #print('ind1: {}'.format(ind1))
        if ind1[0] == 0: ## Hips
            trans  = pose[ind1[0]:ind1[0] + 3].copy()
            angles = pose[ind1[0] + 3:ind1[1]].copy()
            quat = quaternion_from_euler(angles[0], angles[1], angles[2], 'rxyz')
            qpos[ind2[0]:ind2[0] + 3] = trans
            qpos[ind2[0] + 3:ind2[1]] = quat
        else:
            qpos[ind2[0]:ind2[1]] = pose[ind1[0]:ind1[1]]
    return qpos


def get_poses_obj(bvh_file, bvh_obj_files):
    poses, bone_addr, skel_fr = load_bvh_file(bvh_file, skeleton)
    poses = interpolated_traj(poses, args.dt, mocap_fr=skel_fr)
    frame_num = poses.shape[0]

    obj_poses = []
    for obj_file in bvh_obj_files:
        if os.path.isfile(obj_file):
            obj_pose, obj_fr = load_obj_bvh_file(obj_file)
            obj_pose = interpolated_traj(obj_pose, args.dt, mocap_fr=obj_fr)
            if obj_pose.shape[0] > frame_num:
                obj_pose = obj_pose[:frame_num, :]
            elif frame_num > obj_pose.shape[0]:
                obj_pose = np.concatenate((obj_pose, np.tile(obj_pose[-1, :], (frame_num - obj_pose.shape[0], 1))))
            obj_poses.append(obj_pose)
        else:
            null_pose = np.array([0]*6, dtype=np.float32)
            null_pose = np.tile(null_pose[np.newaxis, :], (frame_num, 1))
            obj_poses.append(null_pose)
    obj_poses = np.array(obj_poses)
    qpos_traj = []
    for i in range(frame_num):
        cur_pose = poses[i, :]
        cur_obj_pose = obj_poses[:, i, :]
        cur_qpos = get_qpos_obj(cur_pose, cur_obj_pose, bone_addr)
        qpos_traj.append(cur_qpos)
    qpos_traj = np.vstack(qpos_traj)
    print(qpos_traj.shape)
    return qpos_traj


bvh_files = []

bvh_files = glob.glob('%s/bvh_raw/%s_%s_*actor.bvh' % (args.dataset_path, args.mocap_id, args.action))


#skt_bvh = os.path.expanduser('%s/bvh_raw/%s_%s_25actor.bvh' % (args.dataset_path, args.action, args.mocap_id))
skt_bvh = bvh_files[0]
exclude_bones = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'End', 'Toe'}
spec_channels = {'LeftForeArm': ['Zrotation'], 'RightForeArm': ['Zrotation'],
                 'LeftLeg': ['Xrotation'], 'RightLeg': ['Xrotation']}
skeleton = Skeleton()
skeleton.load_from_bvh(skt_bvh, exclude_bones, spec_channels)

bvh_files.sort()
for _file in bvh_files:
    print('extracting from %s' % _file)

    obj_files = [_file.replace('actor', _object) for _object in object_list]
    qpos_traj = get_poses_obj(_file, obj_files)
    name = os.path.splitext(os.path.basename(_file))[0].split('_')[-1]
    name = name[:name.find('actor')]
    
    bvh_dir = os.path.dirname(_file)
    
    traj_file = '%s/%s_%s_traj.p' % (traj_dir, traj_offset, name)
    
    print('saving to %s' % traj_file)
    pickle.dump(qpos_traj, open(traj_file, 'wb'))
