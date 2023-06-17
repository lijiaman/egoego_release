import numpy as np
import os
import shutil
from os import path
from os import listdir
from PIL import Image
from OpenGL import GL
from gym.envs.mujoco.mujoco_env import MujocoEnv
from relive.utils.math_utils import *
import glfw
import cv2
from skimage.util.shape import view_as_windows




def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))


def out_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../out'))


def log_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../logs'))


def recreate_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)


def load_img(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        I = Image.open(f)
        img = I.resize((224, 224), Image.ANTIALIAS).convert('RGB')
        return img


def save_screen_shots(window, file_name, transparent=False):
    import pyautogui
    xpos, ypos = glfw.get_window_pos(window)
    width, height = glfw.get_window_size(window)
    width = 1920
    height = 1080
    #image = pyautogui.screenshot(region=(xpos*2, ypos*2, width*2, height*2))
    #image = pyautogui.screenshot(region=(xpos*2, ypos*2, width, height))
    image = pyautogui.screenshot(region=(xpos, ypos, width, height))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA if transparent else cv2.COLOR_RGB2BGR)
    if transparent:
        image[np.all(image >= [240, 240, 240, 240], axis=2)] = [255, 255, 255, 0]
    cv2.imwrite(file_name, image)


"""mujoco helper"""


def get_body_qposaddr(model):
    body_qposaddr = dict()
    for i, body_name in enumerate(model.body_names):
        start_joint = model.body_jntadr[i]
        if start_joint < 0:
            continue
        end_joint = start_joint + model.body_jntnum[i]
        start_qposaddr = model.jnt_qposadr[start_joint]
        if end_joint < len(model.jnt_qposadr):
            end_qposaddr = model.jnt_qposadr[end_joint]
        else:
            end_qposaddr = model.nq
        body_qposaddr[body_name] = (start_qposaddr, end_qposaddr)
    return body_qposaddr


def align_human_state(qpos, qvel, ref_qpos, oq):
    qpos[oq:oq+2] = ref_qpos[oq:oq+2]
    hq = get_heading_q(ref_qpos[oq+3:oq+7])
    qpos[oq+3:oq+7] = quaternion_multiply(hq, qpos[oq+3:oq+7])
    qvel[:3] = quat_mul_vec(hq, qvel[:3])
    return qpos, qvel




def get_chunk_selects(chunk_idxes, last_chunk, window_size = 80, overlap = 10):
    shift = window_size - int(overlap/2)
    chunck_selects = []
    for i in range(len(chunk_idxes)):
        chunk_idx = chunk_idxes[i]
        if i == 0:
            chunck_selects.append((0, shift))
        elif i == len(chunk_idxes) - 1:
            chunck_selects.append((-last_chunk, window_size))
        else:
            chunck_selects.append((int(overlap/2), shift))
    return chunck_selects 

def get_chunk_with_overlap(num_frames, window_size = 80, overlap = 10):
    step = window_size - overlap 
    
    chunk_idxes = view_as_windows(np.array(range(num_frames)), window_size, step= step)
    chunk_supp = np.linspace(num_frames - window_size, num_frames-1, num = window_size).astype(int)
    chunk_idxes = np.concatenate((chunk_idxes, chunk_supp[None, ]))
    last_chunk = chunk_idxes[-1][:step][-1] - chunk_idxes[-2][:step][-1] + int(overlap/2)
    chunck_selects = get_chunk_selects(chunk_idxes, last_chunk, window_size= window_size, overlap=overlap)
    
    return chunk_idxes, chunck_selects


def fix_height(expert):
    wbpos = expert['wbpos']
    wbpos = wbpos.reshape(wbpos.shape[0], 24, 3)
    begin_feet = min(wbpos[0, 4, 2],  wbpos[0, 8, 2])
    # if begin_feet > 0.2:
        # print(expert_meta['seq_name'], "sequence invalid for copycat")
        # return None`
    
    begin_feet -= 0.015 # Hypter parameter to tune
    qpos = expert['qpos']
    qpos[:, 2] -= begin_feet
    new_expert = get_expert(qpos, expert_meta, env)
    new_wpos = new_expert['wbpos']
    new_wpos = new_wpos.reshape(new_wpos.shape[0], 24, 3)
    ground_pene = min(np.min(new_wpos[:, 4, 2]),  np.min(new_wpos[:, 8, 2]))
    if ground_pene < -0.15:
        print(expert_meta['seq_name'], "negative sequence invalid for copycat", ground_pene)
        return None
    return new_expert