import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
import cv2
import glfw
import math
import threading

from copycat.khrylib.rl.envs.common.mjviewer import MjViewer
from copycat.smpllib.smpl_parser import SMPL_Parser
from copycat.smpllib.smpl_parser import SMPL_BONE_NAMES

from copycat.khrylib.utils import get_body_qposaddr
from copycat.utils.torch_geometry_transforms import angle_axis_to_rotation_matrix, rotation_matrix_to_quaternion
from mujoco_py import load_model_from_path, MjSim, MjRenderContextOffscreen
from copycat.utils.image_utils import write_frames_to_video
from copycat.utils.transform_utils import (
    convert_aa_to_orth6d, convert_orth_6d_to_aa, vertizalize_smpl_root,
    rotation_matrix_to_angle_axis, convert_orth_6d_to_mat
)

class SMPL_M_Renderer(object):
    def __init__(self, model_file = "", render_size = (960, 480) \
        ):
        self.model = load_model_from_path(model_file)
        self.sim = MjSim(self.model)
        self.viewer = MjRenderContextOffscreen(self.sim, 0)
        self.render_size = render_size

        self.viewer._hide_overlay = True
        self.viewer.cam.distance = 6
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 90

        self.T = 10
        self.qpos_traj = None
        self.offset_z = 0

    def render_smpl(self, body_pose, tran = None, output_name = None, size = (960, 480), frame_rate = 30, add_text = None, offset_z = 0):
        qpose = smpl_to_qpose(body_pose, self.model, tran)
        images = self.render_qpose(qpose, size = self.render_size, frame_rate = frame_rate, add_text = add_text, offset_z = offset_z)

        if output_name is None:
            return images
        else:
            write_frames_to_video(images, out_file_name = output_name, frame_rate = frame_rate, add_text = add_text)

    def render_qpose_and_write(self, qpos, output_name = None, size = (960, 480), frame_rate = 30, add_text = None, offset_z = 0):
        
        images = self.render_qpose(qpos, size = self.render_size, frame_rate = frame_rate, add_text = add_text, offset_z = offset_z)

        if output_name is None:
            return images
        else:
            write_frames_to_video(images, out_file_name = output_name, frame_rate = frame_rate, add_text = add_text)
            

    def render_qpose(self, qpose, size = (960, 480), frame_rate = 30, add_text = None, offset_z = 0, follow = False):
        images = []
        print("Rendering: ", qpose.shape)
        for fr in range(qpose.shape[0]):
            # import pdb
            # pdb.set_trace()
            self.sim.data.qpos[:] = qpose[fr]
            self.sim.data.qpos[2] += offset_z
            self.sim.forward()
            if follow:
                self.viewer.cam.lookat[:2]  = qpose[fr, :2]
            self.viewer.render(size[0], size[1])
            data = np.asarray(self.viewer.read_pixels(size[0], size[1], depth=False)[::-1, :, :], dtype=np.uint8)
            images.append(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        return images
    
    def show_pose(self, size = (960, 480), loop = False):
        fr = 0
        t = 0
        
        while True: 
            if t >= math.floor(self.T) and not self.qpos_traj is None:
                self.sim.data.qpos[:] = self.qpos_traj[fr % self.qpos_traj.shape[0]]
                self.sim.data.qpos[2] += self.offset_z
                # self.sim.forward()
                fr += 1
                t = 0
                
            self.sim.forward()
            self.viewer.render(size[0], size[1])
            # data = np.asarray(self.viewer.read_pixels(size[0], size[1], depth=False)[::-1, :, :], dtype=np.uint8)
            t += 1
            if not loop and not self.qpos_traj is None and fr >= self.qpos_traj.shape[0]:
                break


    def set_smpl_pose(self, pose, tran = None, offset_z = 0):
        self.offset_z = offset_z
        qpose = smpl_to_qpose(pose, self.model, tran)
        self.set_qpose(qpose)

    def set_smpl_pose_6d(self, full_pose, tran = None, offset_z = 0):
        self.offset_z = offset_z
        qpose = smpl_6d_to_qpose(full_pose, self.model)
        self.set_qpose(qpose)

    def set_qpose(self, qpose):
        self.qpos_traj = qpose

    def show_pose_thread(self, return_img = False):
        try:
            t = threading.Thread(target = self.show_pose, args=())
            t.start()
        except e as E:
            print(E, "gg")
        
class SMPL_M_Viewer(object):
    def __init__(self, model_file = "", render_size = (960, 480) \
        ):
        self.model = load_model_from_path(model_file)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.render_size = render_size
        glfw.set_window_size(self.viewer.window, 960, 540)
        glfw.set_window_pos(self.viewer.window, 40, 0)

        self.viewer._hide_overlay = True
        self.viewer.cam.distance = 6
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 90
        self.T = 10
        self.qpos_traj = None
        self.offset_z = 0
        self.curr_img = None


    def render_qpose(self, qpose, follow = False):
        images = []
        t = 0
        fr = 0
        print("Rendering: ", qpose.shape)
        while fr < len(qpose):
            if t >= math.floor(self.T):
                self.sim.data.qpos[:] = qpose[fr]
                self.sim.data.qpos[2] += self.offset_z
                self.sim.forward()

                if follow:
                    self.viewer.cam.lookat[:2]  = qpose[fr, :2]

                fr += 1
                t = 0

            self.viewer.render()
            t += 1

    def show_pose(self, return_img = False, size = (1920, 1080), loop = False):
        fr = 0
        t = 0
        max_len = self.qpos_traj.shape[0]
        while loop or fr <= max_len: 
            if t >= math.floor(self.T) and not self.qpos_traj is None:
                self.sim.data.qpos[:] = self.qpos_traj[fr% self.qpos_traj.shape[0]]
                self.sim.data.qpos[2] += self.offset_z
                self.sim.forward()
                fr += 1
                t = 0

            self.viewer.render()
            if return_img:
                self.curr_img = np.asarray(self.viewer.read_pixels(size[0], size[1], depth=False)[::-1, :, :], dtype=np.uint8)
            t += 1
            
    def show_pose_in_thread(self, return_img = False, size = (1920, 1080)):
        fr = 0
        t = 0
        while True: 
            if t >= math.floor(self.T) and not self.qpos_traj is None:
                self.sim.data.qpos[:] = self.qpos_traj[fr% self.qpos_traj.shape[0]]
                self.sim.data.qpos[2] += self.offset_z
                self.sim.forward()
                fr += 1
                t = 0

            self.viewer.render()
            if return_img:
                self.curr_img = np.asarray(self.viewer.read_pixels(size[0], size[1], depth=False)[::-1, :, :], dtype=np.uint8)
            t += 1

    def show_pose_thread(self, return_img = False):
        try:
            t = threading.Thread(target = self.show_pose_in_thread, args=(return_img, ))
            t.start()
        except e as E:
            print(E, "gg")

    def set_smpl_pose(self, pose, trans = None, offset_z = 0):
        self.offset_z = offset_z
        qpose = smpl_to_qpose(pose, self.model, trans)
        self.set_qpose(qpose)

    def set_smpl_pose_6d(self, full_pose, offset_z = 0):
        self.offset_z = offset_z
        qpose = smpl_6d_to_qpose(full_pose, self.model)
        self.set_qpose(qpose)
        
    def set_qpose(self, qpose):
        self.qpos_traj = qpose


def smpl_to_qpose(pose, model, trans = None, normalize = False, random_root = False, euler_order = "ZYX"):
    '''
        Expect pose to be batch_size x 72
        trans to be batch_size x 3
    '''
    if trans is None:
        trans = np.zeros((pose.shape[0], 3))
        trans[:,2] = 0.91437225
    if normalize:
        pose, trans = normalize_smpl_pose(pose, trans, random_root=random_root)
    
    smpl_2_mujoco = [ SMPL_BONE_NAMES.index(q) for q in list(get_body_qposaddr(model).keys()) if q in SMPL_BONE_NAMES]

    if not torch.is_tensor(pose):
        pose = torch.tensor(pose)
    
    pose = pose.reshape(-1, 24, 3).reshape(-1, 72)
    
    curr_pose_mat = angle_axis_to_rotation_matrix(pose.reshape(-1, 3)).reshape(pose.shape[0], -1, 4, 4)
    curr_spose = sRot.from_matrix(curr_pose_mat[:,:,:3,:3].reshape(-1, 3, 3).numpy())
    curr_spose_euler = curr_spose.as_euler(euler_order, degrees=False).reshape(curr_pose_mat.shape[0], -1)
    curr_spose_euler = curr_spose_euler.reshape(-1, 24, 3)[:, smpl_2_mujoco, :].reshape(-1, 72)
    root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:,0,:3,:])
    curr_qpos = np.concatenate((trans, root_quat, curr_spose_euler[:,3:]), axis = 1)
    return curr_qpos


def smpl_6d_to_qpose(full_pose, model, normalize = False):
    pose_aa = convert_orth_6d_to_aa(torch.tensor(full_pose[:,3:]))
    trans = full_pose[:,:3]
    curr_qpose = smpl_to_qpose(pose_aa, model, trans, normalize = normalize)
    return curr_qpose

def normalize_smpl_pose(pose_aa, trans = None, random_root = False):
    root_aa = pose_aa[0,:3]
    root_rot = sRot.from_rotvec(np.array(root_aa))
    root_euler = np.array(root_rot.as_euler("xyz", degrees = False))
    target_root_euler = root_euler.copy() # take away Z axis rotation so the human is always facing same direction
    if random_root:
        target_root_euler[2] = np.random.random(1) * np.pi * 2
    else:
        target_root_euler[2] = -1.57
    target_root_rot = sRot.from_euler("xyz", target_root_euler, degrees=False)
    target_root_aa = target_root_rot.as_rotvec()

    target_root_mat = target_root_rot.as_matrix()
    root_mat = root_rot.as_matrix()
    apply_mat = np.matmul(target_root_mat, np.linalg.inv(root_mat))

    if torch.is_tensor(pose_aa):
        pose_aa = vertizalize_smpl_root(pose_aa, root_vec = target_root_aa)
    else:
        pose_aa = vertizalize_smpl_root(torch.from_numpy(pose_aa), root_vec = target_root_aa)
        

    if not trans is None:
        trans[:,[0,1]] -= trans[0, [0,1]]
        trans[:,[2]] = trans[:,[2]] - trans[0, [2]] + 0.91437225
        trans = np.matmul(apply_mat, trans.T).T
    return pose_aa, trans

    
    
