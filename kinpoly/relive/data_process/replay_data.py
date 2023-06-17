import os
import sys
import numpy as np
import math
sys.path.append(os.getcwd())

from relive.utils import *
from mujoco_py import load_model_from_path, MjSim
from relive.envs.common.mjviewer import MjViewer
import pickle
import glob
import argparse
import glfw
import yaml
import cv2
from relive.utils.transformation import quaternion_from_euler, quaternion_multiply

parser = argparse.ArgumentParser()
parser.add_argument('--model-id', type=str, default='1213')
parser.add_argument('--mocap-id', type=str, default='1001')
parser.add_argument('--meta-id', type=str, default='meta_subject_01')
parser.add_argument('--action', type=str, default='sit')
parser.add_argument('--take-ind', type=int, default=0)
parser.add_argument('--multi', action='store_true', default=False)
parser.add_argument('--dataset-path', type=str, default='datasets')
parser.add_argument('--show-noise', action='store_true', default=False)

args = parser.parse_args()


if args.meta_id is not None:
    meta_file = os.path.expanduser('%s/meta/%s.yml' % (args.dataset_path, args.meta_id))
    meta = yaml.safe_load(open(meta_file, 'r'))
    object_num = len(meta['object'][args.action])
    offset_z = meta['offset_z'][args.action]
    off_obj_pos = 7 * object_num

model_file = 'assets/mujoco_models/%s/%s/humanoid_vis_single.xml' % (args.model_id, args.action)
print(model_file)
model = load_model_from_path(model_file)


sim = MjSim(model)
viewer = MjViewer(sim)

take_folders = glob.glob(os.path.expanduser('%s/fpv_frames/%s_*' % (args.dataset_path, args.mocap_id)))
take_folders.sort()
takes = []
for x in take_folders:
    take_name = os.path.splitext(os.path.basename(x))[0]
    if not take_name in meta['action_type']:
        continue
    if meta['action_type'][take_name] == args.action:
        takes.append(take_name)
print(takes)
bvh_files = glob.glob(os.path.expanduser('%s/bvh_raw/*actor.bvh' % args.dataset_path))

_takes = []
for x in takes:
    contain_list = [y for y in bvh_files if x.split('_')[-1] in y]
    if len(contain_list) > 0:
        _takes.append(x)
takes = _takes
print(takes)
def key_callback(key, action, mods):
    global T, fr, paused, stop, im_offset, offset_z, take_ind, reverse

    if action != glfw.RELEASE:
        return False
    elif key == glfw.KEY_D:
        T *= 1.5
    elif key == glfw.KEY_F:
        T = max(1, T / 1.5)
    elif key == glfw.KEY_R:
        stop = True
    elif key == glfw.KEY_Q:
        if fr + im_offset > 0:
            im_offset -= 1
        update_fpv()
    elif key == glfw.KEY_E:
        if fr + im_offset < len(images) - 1:
            im_offset += 1
        update_fpv()
    elif key == glfw.KEY_W:
        fr = max(0, -im_offset)
        update_all()
    elif key == glfw.KEY_S:
        reverse = not reverse
    elif key == glfw.KEY_C:
        take_ind = (take_ind + 1) % len(takes)
        load_take()
        update_all()
    elif key == glfw.KEY_Z:
        take_ind = (take_ind - 1) % len(takes)
        load_take()
        update_all()
    elif key == glfw.KEY_RIGHT:
        if fr + im_offset < len(images) - 1 and fr < qpos_traj.shape[0] - 1:
            fr += 1
        update_all()
    elif key == glfw.KEY_LEFT:
        if fr + im_offset > 0 and fr > 0:
            fr -= 1
        update_all()
    elif key == glfw.KEY_UP:
        offset_z += 0.001
        update_mocap()
    elif key == glfw.KEY_DOWN:
        offset_z -= 0.001
        update_mocap()
    elif key == glfw.KEY_SPACE:
        paused = not paused
    else:
        return False
    return True


def update_mocap():
    sim.data.qpos[:] = qpos_traj[fr]
    sim.data.qpos[off_obj_pos + 2] += offset_z

    sim.forward()


def update_fpv():
    print('take: %s  fr: %d  im_fr: %d  offset: %d  fr_boundary: (%d, %d)  dz: %.3f' %
          (takes[take_ind], fr, fr + im_offset, im_offset, max(-im_offset, 0),
           min(qpos_traj.shape[0], len(images) - im_offset - 1), offset_z))
    cv2.imshow('FPV', images[fr + im_offset])
    cv2.waitKey(1)


def update_all():
    update_mocap()
    update_fpv()


def load_take():
    global qpos_traj, images, im_offset, fr
    take = takes[take_ind]
    print(take)
    im_offset = 0 if not take in meta['video_mocap_sync'].keys() else meta['video_mocap_sync'][take][0]
    #im_offset = 0 if args.meta_id is None else meta['video_mocap_sync'][take][0]
    fr = max(0, -im_offset)
    traj_file = os.path.expanduser('%s/traj/%s_traj.p' % (args.dataset_path, take))
    qpos_traj = pickle.load(open(traj_file, "rb"))
    '''
    _noise = np.random.normal(loc=0.0, scale=1.e-2, size=2)
    _ang_noise = np.random.normal(loc=0.0, scale=np.deg2rad(5.0), size=1)
    quat_noise = quaternion_from_euler(0.0, 0.0, _ang_noise, axes='rxyz')
    print(_noise)
    qpos_traj[:, :2] +=_noise[np.newaxis, :]
    qpos_traj[:, 3:7] = [quaternion_multiply(quat_noise, quat[3:7]) for quat in qpos_traj]
    '''
    frame_folder = os.path.expanduser('%s/fpv_frames/%s' % (args.dataset_path, take))
    frame_files = glob.glob(os.path.join(frame_folder, '*.png'))
    frame_files.sort()
    images = [cv2.imread(file) for file in frame_files]
    print('traj len: %d,  image num: %d,  dz: %.3f' % (qpos_traj.shape[0], len(images), offset_z))


qpos_traj = None
images = None
take_ind = args.take_ind

T = 10
im_offset = 0
fr = max(0, -im_offset)
paused = False
stop = False
reverse = False


cv2.namedWindow('FPV')
cv2.moveWindow('FPV', 150, 200)

viewer._hide_overlay = True
viewer.cam.distance = 10
viewer.cam.elevation = -20
viewer.cam.azimuth = 90
viewer.custom_key_callback = key_callback
glfw.set_window_size(viewer.window, 1000, 720)
glfw.set_window_pos(viewer.window, 400, 0)
load_take()
#update_fpv()
update_mocap()
t = 0

while not stop:

    if t >= math.floor(T):
        if not reverse and fr + im_offset < len(images) - 1 and fr < qpos_traj.shape[0] - 1:
            fr += 1
            update_all()
        elif reverse and fr + im_offset > 0 and fr > 0:
            fr -= 1
            update_all()
        t = 0

    viewer.render()
    if not paused:
        t += 1

