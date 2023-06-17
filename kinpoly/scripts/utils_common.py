# Added by Jiaman
import os 
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import yaml
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils

import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import shutil 

import torch.utils.data as data

import scipy.io
import scipy.ndimage

import tqdm

import cv2
import math
import time
import torch

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from matplotlib.animation import FuncAnimation

import colorsys

def show3Dpose(channels, ax, radius=40, lcolor='#ff0000', rcolor='#0000ff', mask=None, only_show_points=False, use_joint12=False):
    # mask for missing joints, if joints are missing, visualize with another color
    lcolor = '#E76F51'
    rcolor = '#F4A261'
    vals = channels # 24 X 3
    # 0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee
    # 5: R_Knee, 6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3
    # 10: L_Foot, 11: R_Foot, 12: Neck, 13: L_Collar, 14: R_Collar
    # 15: Head, 16: L_Shoulder, 17: R_Shoulder, 18: L_Elbow, 19: R_Elbow
    # 20: L_Wrist, 21: R_Wrist, 22(25): L_Index1, 23(40): R_Index1

    # connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
    #                [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
    #                [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
    if use_joint12:
        connections = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]
    else:
        connections = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                    [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]

    # LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    LR = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)

    if not only_show_points:
        for ind, (i,j) in enumerate(connections):
            x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
            ax.plot(x, y, z, lw=2, c=lcolor if LR[ind] else rcolor)
        
    if mask is None:
        ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], marker='o')
    else: # mask: 24
        if only_show_points:
            ax.scatter(vals[mask==1, 0], vals[mask==1, 1], vals[mask==1, 2], marker='o')
        else:
            ax.scatter(vals[mask==1, 0], vals[mask==1, 1], vals[mask==1, 2], marker='o')
            ax.scatter(vals[mask==0, 0], vals[mask==0, 1], vals[mask==0, 2], c="#FF0000", marker='o')

    RADIUS = radius  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_axis_off()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

def vis_single_frame(pose_data, dest_img_path, mask=None, only_show_points=False, use_joint12=False):
    # pose_data: 24 X 3 numpy array
    fig = plt.figure(figsize=(12, 7))
    # fig = plt.figure(figsize=(5, 5))

    # ax = fig.add_subplot('111', projection='3d', aspect=1)
    ax = Axes3D(fig)
    show3Dpose(pose_data, ax, radius=1, mask=mask, only_show_points=only_show_points, use_joint12=use_joint12)
    # ax.view_init(-90, 90) # For rest pose visualization (3dpw original data)
    # ax.view_init(0, 90)
    ax.view_init(0, 120)
    # ax.view_init(-90, -90) # For 3dpw processed data (the processing includes the camera rotation)

    plt.draw()
    plt.savefig(dest_img_path)
    plt.cla()
    plt.close()

def vis_multiple_frames(pose_data, dest_img_path, mask=None, only_show_points=False, use_joint12=False):
    # pose_data: N X 24 X 3 numpy array
    fig = plt.figure(figsize=(12, 7))
    # fig = plt.figure(figsize=(5, 5))

    # ax = fig.add_subplot('111', projection='3d', aspect=1)
    ax = Axes3D(fig)
    for idx in range(pose_data.shape[0]):
        show3Dpose(pose_data[idx], ax, radius=1, mask=mask, only_show_points=only_show_points, use_joint12=use_joint12)

    ax.view_init(-90, 90) # For rest pose visualization 
    # ax.view_init(0, 90)
    # ax.view_init(0, 120)

    plt.draw()
    plt.savefig(dest_img_path)
    plt.cla()
    plt.close()

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def make_result_folders(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory

def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if ((not callable(getattr(trainer, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr))]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)

def write_images(pose, image_directory, iterations, tag, mask=None, only_show_points=False, use_joint12=False):
    # pose: bs X 24 X 3
    dest_folder = os.path.join(image_directory, str(iterations), tag)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    bs = pose.size()[0]
    for bs_idx in range(bs):
        single_pose = pose[bs_idx, :, :].data.cpu().numpy() # 24 X 3
        if mask is not None:
            curr_mask = mask[bs_idx, :]
        else:
            curr_mask = None
        dest_path = os.path.join(dest_folder, str(bs_idx)+".png")
        vis_single_frame(single_pose, dest_path, curr_mask, only_show_points, use_joint12=use_joint12)

def write_multiple_images(pose, image_directory, iterations, tag, mask=None, only_show_points=False, use_joint12=False):
    # pose: bs X 24 X 3
    dest_folder = os.path.join(image_directory, str(iterations), tag)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    dest_path = os.path.join(dest_folder, "multiple_pose_cmp.png")
    vis_multiple_frames(pose, dest_path, use_joint12=use_joint12)

def write_images_with_list(pose_list, image_directory, iterations, tag):
    # pose_list: each element is bs X 24 X 3
    dest_folder = os.path.join(image_directory, str(iterations), tag)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    bs = pose_list[0].size()[0]
    num_sampled_per_bs = len(pose_list)
    for bs_idx in range(bs):
        for s_idx in range(num_sampled_per_bs):
            single_pose = pose_list[s_idx][bs_idx, :, :].data.cpu().numpy() # 24 X 3
        
            dest_path = os.path.join(dest_folder, str(bs_idx)+"_"+str(s_idx)+".png")
            vis_single_frame(single_pose, dest_path)

def write_images_interpolation(interp_pose, image_directory, iterations, tag):
    # interp_pose: K X (bs/2) X 24 X 3
    num_pairs = interp_pose.size()[1]
    for i in range(num_pairs):
        dest_folder = os.path.join(image_directory, str(iterations), tag, str(i))
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        num_interp = interp_pose.size()[0]
        for j in range(num_interp):
            curr_pose = interp_pose[j, i, :, :].data.cpu().numpy() # 24 X 3
            dest_path = os.path.join(dest_folder, str(j)+".png")
            vis_single_frame(curr_pose, dest_path)

def show3Dpose_animation(channels, image_directory, iterations=0, tag="test", bs_idx=0, \
    use_joint12=False, use_amass=False, use_lafan=False, use_mujoco=False, dest_vis_path=None):
    # channels: K X T X n_joints X 3
    fig = plt.figure(figsize=(9, 7))
    ax = Axes3D(fig) 

    lcolor = '#E76F51'
    rcolor = '#F4A261'
    gt_color = '#27AE60'
    pred_color = '#E74C3C'
    vals = channels # K X T X 24 X 3, K represents how many skeleton showing in same figure(K=2: show gt and generation)
    num_cmp = vals.shape[0]
   
    if use_joint12:
        connections = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]
    elif use_lafan:
        connections = [[0, 1], [0, 5], [0, 9], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [11, 14], 
        [11, 18], [12, 13], [14, 15], [15, 16], [16, 17], [18, 19], [19, 20], [20, 21]]
    elif use_mujoco:
        # Mujoco model joints
        # 0: Pelvis, 1: L_Hip, 2: L_Knee, 3: L_Ankle, 4: L_Foot,
        # 5: R_Hip, 6: R_Knee, 7: R_Ankle, 8: R_Foot, 9: Spine1,
        # 10: Spine2, 11: Spine3, 12: Neck, 13: Head, 14: L_Collar, 15: L_Shoulder
        # 16: L_Elbow, 17: L_Wrist, 18: L_Index, 19: R_Collar, 20: R_Shoulder, 21: R_Elbow
        # 22: R_Wrist, 23: R_Index
        connections = [[0, 1], [0, 5], [0, 9], [1, 2], [5, 6], [9, 10], [2, 3], [6, 7], [10, 11], [3, 4], [7, 8], [11, 12], [11, 14], [11, 19],
                [12, 13], [14, 15], [19, 20], [15, 16], [20, 21], [16, 17], [21, 22], [17, 18], [22, 23]] 
    else:
        connections = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                    [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]

    LR = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)

    lines = []
    for cmp_idx in range(num_cmp):
        cur_line = []
        for ind, (i,j) in enumerate(connections):
            if num_cmp == 2 and cmp_idx == 0:
                cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=gt_color)[0])
            elif num_cmp == 2 and cmp_idx == 1:
                cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=pred_color)[0])
            else:
                cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=pred_color)[0])
                # cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=lcolor if LR[ind] else rcolor)[0])
        lines.append(cur_line)

    # ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], marker='o')
  
    def animate(i):
        changed = []
        for ai in range(len(vals)):
            for ind, (p_idx, j_idx) in enumerate(connections):
                lines[ai][ind].set_data([vals[ai][i, j_idx, 0], vals[ai][i, p_idx, 0]], \
                    [vals[ai][i, j_idx, 1], vals[ai][i, p_idx, 1]])
                lines[ai][ind].set_3d_properties(
                    [vals[ai][i, j_idx, 2], vals[ai][i, p_idx, 2]])
            changed += lines

        return changed

    RADIUS = 2  # space around the subject
    xroot, yroot, zroot = vals[0, 0, 0, 0], vals[0, 0, 0, 1], vals[0, 0, 0, 2]
    # xroot, yroot, zroot = 0, 0, 0 # For debug
    if use_joint12:
        # ax.view_init(0, 120)
        # ax.view_init(-90, -90)
        # ax.view_init(-90, 90) # For 3dpw original data
        ax.view_init(-90, -90)
    else:
        if use_amass:
            ax.view_init(0, 120) # Used in training AMASS dataset
        else:
            # ax.view_init(-90, -90) # Used in vibe?
            ax.view_init(0, 120)
        # ax.view_init(-90, -90)

    # ax.view_init(0, 0)
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    # ax.set_axis_off()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    ani = FuncAnimation(fig,                                            
                        animate,                                        
                        np.arange(len(vals[0])),                  
                        interval=33.33)  
    dest_folder = os.path.join(image_directory, str(iterations), tag)    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    if dest_vis_path is None:
        dest_gif_path = os.path.join(dest_folder, str(bs_idx)+".gif")      
    else:
        dest_gif_path = dest_vis_path 

    ani.save(dest_gif_path,                       
            writer="imagemagick",                                                
            fps=30) 

    # plt.draw()
    # plt.savefig(dest_img_path)
    plt.cla()
    plt.close()

def show3Dpose_animation_multiple(channels, image_directory, iterations, \
    tag, bs_idx, use_joint12=False, use_amass=False, use_lafan=False, \
    make_translation=False, eval_flag=False):
    # channels: K X T X n_joints X 3
    if make_translation:
        for tmp_idx in range(channels.shape[0]):
            channels[tmp_idx, :, :, 0] += tmp_idx

    fig = plt.figure(figsize=(9, 7))
    ax = Axes3D(fig) 

    lcolor = '#E76F51'
    rcolor = '#F4A261'
    gt_color = '#27AE60'
    pred_color = '#E74C3C'

    color_list = ['#8000ff', '#0000ff', '#87ff00', pred_color, gt_color]
    vals = channels # K X T X 24 X 3, K represents how many skeleton showing in same figure(K=2: show gt and generation)
    num_cmp = vals.shape[0]
    if num_cmp > len(color_list):
        color_list = []
        color_list.append(gt_color)
        color_list.extend([pred_color] * (num_cmp-1))
   
    if use_joint12:
        connections = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]
    elif use_lafan:
        connections = [[0, 1], [0, 5], [0, 9], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [11, 14], 
        [11, 18], [12, 13], [14, 15], [15, 16], [16, 17], [18, 19], [19, 20], [20, 21]]
    else:
        connections = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                    [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]

    LR = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)

    lines = []
    for cmp_idx in range(num_cmp):
        cur_line = []
        for ind, (i,j) in enumerate(connections):
            curr_color = color_list[cmp_idx]
            cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=curr_color)[0])
            # if num_cmp == 2 and cmp_idx == 0:
            #     cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=gt_color)[0])
            # elif num_cmp == 2 and cmp_idx == 1:
            #     cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=pred_color)[0])
            # else:
            #     cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=pred_color)[0])
                # cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=lcolor if LR[ind] else rcolor)[0])
        lines.append(cur_line)

    # ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], marker='o')
  
    def animate(i):
        changed = []
        for ai in range(len(vals)):
            for ind, (p_idx, j_idx) in enumerate(connections):
                lines[ai][ind].set_data([vals[ai][i, j_idx, 0], vals[ai][i, p_idx, 0]], \
                    [vals[ai][i, j_idx, 1], vals[ai][i, p_idx, 1]])
                lines[ai][ind].set_3d_properties(
                    [vals[ai][i, j_idx, 2], vals[ai][i, p_idx, 2]])
            changed += lines

        return changed

    RADIUS = 2  # space around the subject
    mid_idx = num_cmp // 2
    xroot, yroot, zroot = vals[mid_idx, 0, 0, 0], vals[mid_idx, 0, 0, 1], vals[mid_idx, 0, 0, 2]
    if use_joint12:
        # ax.view_init(0, 120)
        # ax.view_init(-90, -90)
        # ax.view_init(-90, 90) # For 3dpw original data
        ax.view_init(-90, -90)
    else:
        # ax.view_init(0, 120) # Used in training AMASS dataset
        if use_amass:
            ax.view_init(0, 120)
        elif use_lafan:
            ax.view_init(-90, 90)
        else:
            ax.view_init(-90, -90) # Used in vibe?
        # ax.view_init(-90, 90)
            # ax.view_init(-90, 90) # used in LAFAN1
        # ax.view_init(-90, -90)

    # ax.view_init(0, 0)
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    # ax.set_axis_off()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    ani = FuncAnimation(fig,                                            
                        animate,                                        
                        np.arange(len(vals[0])),                  
                        interval=33.33)  

    if eval_flag:
        dest_folder = image_directory 
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        dest_gif_path = os.path.join(dest_folder, "res.mp4")   
    else:
        dest_folder = os.path.join(image_directory, str(iterations), tag)    
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        dest_gif_path = os.path.join(dest_folder, str(bs_idx)+".mp4")                        
    ani.save(dest_gif_path,                       
            writer="imagemagick",                                                
            fps=30) 

    # plt.draw()
    # plt.savefig(dest_img_path)
    plt.cla()
    plt.close()

def show3Dpose_animation_with_mask(channels, mask, image_directory, iterations, tag, bs_idx):
    # channels: K X T X n_joints X 3
    # mask: T X n_joints
    fig = plt.figure(figsize=(9, 7))
    ax = Axes3D(fig) 

    lcolor = '#E76F51'
    rcolor = '#F4A261'
    gt_color = '#27AE60'
    pred_color = '#E74C3C'
    vals = channels # K X T X 24 X 3, K represents how many skeleton showing in same figure(K=2: show gt and generation)
    num_cmp = vals.shape[0]
   
    connections = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]

    LR = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)

    lines = []
    for cmp_idx in range(len(vals)): # Only visualize prediction as lines
        cur_line = []
        if cmp_idx == 0:
            for j_idx in range(24):
                cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], linestyle="", marker="o", c="red", markersize=4)[0])
        elif cmp_idx == 1:
            for ind, (i,j) in enumerate(connections):
                cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c="green")[0])
            
        lines.append(cur_line)

    # ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], marker='o')
    # points = []
    # for j_idx in range(24):
    #     points.append(ax.scatter(0, 0, 0, marker='o'))
        # ax.scatter(vals[mask==1, 0], vals[mask==1, 1], vals[mask==1, 2], marker='o')
        # ax.scatter(vals[mask==0, 0], vals[mask==0, 1], vals[mask==0, 2], c="#FF0000", marker='o')

    def animate(i):
        changed = []
        for ai in range(len(vals)): # Only visualize the prediction as lines
            if ai == 0:
                for j_idx in range(24):
                    if mask[i, j_idx] == 1:
                        lines[ai][j_idx].set_color("red")
                    else:
                        lines[ai][j_idx].set_color("white") # Makeing missing joints disappear
                    lines[ai][j_idx].set_data([vals[ai][i, j_idx, 0], vals[ai][i, j_idx, 1]])
                    lines[ai][j_idx].set_3d_properties(vals[ai][i, j_idx, 2])
            elif ai == 1:
                for ind, (p_idx, j_idx) in enumerate(connections):
                    lines[ai][ind].set_data([vals[ai][i, j_idx, 0], vals[ai][i, p_idx, 0]], \
                        [vals[ai][i, j_idx, 1], vals[ai][i, p_idx, 1]])
                    lines[ai][ind].set_3d_properties(
                        [vals[ai][i, j_idx, 2], vals[ai][i, p_idx, 2]])
            changed += lines

        # for j_idx in range(24): # Visualize the visible joints in gt
        #     if mask[i, j_idx] == 1:
        #         points[j_idx].set_data([vals[0][i, j_idx, 0], vals[0][i, j_idx, 1]])
        #         points[j_idx].set_3d_properties(vals[0][i, j_idx, 2])
        #         # points[j_idx]._offsets3d = (vals[0][i, j_idx, 0], vals[0][i, j_idx, 1], vals[0][i, j_idx, 2])
        # changed += points

        return changed

    RADIUS = 1  # space around the subject
    xroot, yroot, zroot = vals[0, 0, 0, 0], vals[0, 0, 0, 1], vals[0, 0, 0, 2]
    ax.view_init(0, 120)
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_axis_off()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    ani = FuncAnimation(fig,                                            
                        animate,                                        
                        np.arange(len(vals[0])),                  
                        interval=33.33)  
    dest_folder = os.path.join(image_directory, str(iterations), tag)    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    dest_gif_path = os.path.join(dest_folder, str(bs_idx)+".mp4")                        
    ani.save(dest_gif_path,                       
            writer="imagemagick",                                                
            fps=30) 

    # plt.draw()
    # plt.savefig(dest_img_path)
    plt.cla()
    plt.close()

class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

def test_animate_vis():
    # Test animation visualization
    fig = plt.figure(figsize=(9, 7))
    ax = Axes3D(fig)
    # channels: K X T X 24 X 3
    rest_pose_npy = "./utils/data/rest_pose_coord.npy"
    rest_pose_data = np.load(rest_pose_npy) # 24 X 3
    rest_pose_data = rest_pose_data[np.newaxis, np.newaxis, :, :] # 1 X 1 X 24 X 3
    channels = rest_pose_data
    # show3Dpose_animation(channels, ax, radius=40, lcolor='#ff0000', rcolor='#0000ff')
    # ax.view_init(-90, 90) # For rest pose visualization 

    # mask for missing joints, if joints are missing, visualize with another color
    lcolor = '#E76F51'
    rcolor = '#F4A261'
    vals = channels # K X T X 24 X 3, K represents how many skeleton showing in same figure(K=2: show gt and generation)
    num_cmp = vals.shape[0]
   
    connections = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]

    LR = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)

    lines = []
    for cmp_idx in range(num_cmp):
        cur_line = []
        for ind, (i,j) in enumerate(connections):
            cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=lcolor if LR[ind] else rcolor)[0])
        lines.append(cur_line)

    # ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], marker='o')
  
    def animate(i):
        changed = []
        for ai in range(len(vals)):
            for ind, (p_idx, j_idx) in enumerate(connections):
                lines[ai][ind].set_data([vals[ai][i, j_idx, 0], vals[ai][i, p_idx, 0]], \
                    [vals[ai][i, j_idx, 1], vals[ai][i, p_idx, 1]])
                lines[ai][ind].set_3d_properties(
                    [vals[ai][i, j_idx, 2], vals[ai][i, p_idx, 2]])
            changed += lines

        return changed

    RADIUS = 1  # space around the subject
    xroot, yroot, zroot = vals[0, 0, 0, 0], vals[0, 0, 0, 1], vals[0, 0, 0, 2]
    ax.view_init(0, 120)
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    # ax.set_axis_off()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    ani = FuncAnimation(fig,                                            
                        animate,                                        
                        np.arange(len(vals[0])),                  
                        interval=1000)                              
    ani.save("./tmp_test_vis.gif",                       
            writer="imagemagick",                                                
            fps=1) 

    # plt.draw()
    # plt.savefig(dest_img_path)
    plt.cla()
    plt.close()

def write_obj_file(vertices, faces, obj_path):
    w_f = open(obj_path, 'w')
    num_v = vertices.shape[0]
    num_f = faces.shape[0]
    for v_idx in range(num_v):
        w_f.write("v "+str(vertices[v_idx, 0])+" "+str(vertices[v_idx, 1])+" "+str(vertices[v_idx, 2])+"\n")

    for f_idx in range(num_f):
        w_f.write("f "+str(faces[f_idx, 0]+1)+" "+str(faces[f_idx, 1]+1)+" "+str(faces[f_idx, 2]+1)+"\n")

def save_mesh_obj(out_folder, rot_mat, root_trans, temporal_mask):
    # rot_mat: T X 24 X 3 X 3
    # root_trans: T X 3
    from lib.models.smpl import get_smpl_faces

    timesteps = rot_mat.shape[0]

    smpl_model = SMPL(
        SMPL_MODEL_DIR,
        batch_size=timesteps,
        create_transl=True
    )

    mean_params = np.load(SMPL_MEAN_PARAMS)
    init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
    init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
    # init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
    # init_cam = torch.from_numpy(np.asarray([0.3, 0.3, 0, 0])).unsqueeze(0)

    smpl_faces = get_smpl_faces()

    pred_v_list = []
    for frame_idx in range(timesteps):
        pred_rotmat = torch.from_numpy(rot_mat).float()[frame_idx:frame_idx+1] # 1 X 24 X 3 X 3
        pred_trans = torch.from_numpy(root_trans).float()[frame_idx:frame_idx+1] # 1 X 3
        pred_output = smpl_model(
            betas=init_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
            transl=pred_trans,
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        vertices_arr = pred_vertices[0].data.cpu().numpy()

        # Fix each frame's position 
        # dim_idx = 2
        # curr_floor_plane = vertices_arr[:, dim_idx].min()
        # curr_trans_offset = - curr_floor_plane
        # curr_converted_v = vertices_arr.copy()
        # curr_converted_v[:, dim_idx] = curr_converted_v[:, dim_idx] + curr_trans_offset

        pred_v_list.append(vertices_arr)

    # If process position for the whole sequence
    # pred_v_list = np.asarray(pred_v_list)
    # dim_idx = 2
    # floor_plane = pred_v_list.reshape(-1, 3)[:, dim_idx].min()
    # min_index = pred_v_list.reshape(-1, 3)[:, dim_idx].argmin()
    # print("floor plane:{0}".format(floor_plane))
    # print("index:{0}".format(min_index//6890))

    # max_floor_plane = pred_v_list.reshape(-1, 3)[:, dim_idx].max()
    # max_index = pred_v_list.reshape(-1, 3)[:, dim_idx].argmax()
    # print("max floor plane:{0}".format(max_floor_plane))
    # print("max index:{0}".format(max_index//6890))
    # trans_offset = - floor_plane
    # converted_v = pred_v_list.copy()
    # converted_v[:, :, dim_idx] = converted_v[:, :, dim_idx] + trans_offset


    # If process position for each single frame 
    converted_v = np.asarray(pred_v_list)

    # print("trans offset:{0}".format(trans_offset))

    dest_obj_folder = os.path.join(out_folder, "our_wo_root_objs")
    if not os.path.exists(dest_obj_folder):
        os.makedirs(dest_obj_folder)

    if temporal_mask is not None:
        dest_k_obj_folder = os.path.join(out_folder, "k_objs")
        if not os.path.exists(dest_k_obj_folder):
            os.makedirs(dest_k_obj_folder)
    
    for frame_idx in range(timesteps):
        dest_path = os.path.join(dest_obj_folder, ("%05d"%frame_idx)+".obj")
        vert_arr = converted_v[frame_idx]
        write_obj_file(vert_arr, smpl_faces, dest_path)

        if temporal_mask is not None:
            if temporal_mask[frame_idx] == 1:
                dest_k_path = os.path.join(dest_k_obj_folder, ("%05d"%frame_idx)+"_k.obj")
                k_vert_arr = converted_v[frame_idx]
                write_obj_file(k_vert_arr, smpl_faces, dest_k_path)

    if temporal_mask is not None:
        dest_mask_folder = os.path.join(out_folder, "mask")
        if not os.path.exists(dest_mask_folder):
            os.makedirs(dest_mask_folder)

        dest_mask_path = os.path.join(dest_mask_folder, "temporal_mask.npy")
        np.save(dest_mask_path, temporal_mask)

        dest_trans_y_path = os.path.join(dest_mask_folder, "trans_y_floor.npy")
        np.save(dest_trans_y_path, trans_offset)

def prep_mesh_for_sampled_seq(npy_folder):
    npy_files = os.listdir(npy_folder)
 
    for f_name in npy_files:
        if "_rot.npy" in f_name:
        # if "_rot_mat.npy" in f_name:
            rot_npy_path = os.path.join(npy_folder, f_name)
            root_trans_path = rot_npy_path.replace("_rot.npy", "_trans.npy")

            out_folder = os.path.join(npy_folder, f_name.replace(".npy", ""))
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            
            # k_idx = int(f_name.split("_")[1])
            # gen_npy_data = np.load(rot_npy_path)[k_idx] # T X 24 X 3 X 3
            # gen_root_trans = np.load(root_trans_path)[k_idx, :, 0, :] # T X 3
            # print("root trans:{0}".format(np.load(root_trans_path).shape))

            gen_npy_data = np.load(rot_npy_path) # T X 24 X 3 X 3
            gen_root_trans = np.load(root_trans_path) # T X 3
            
            save_mesh_obj(out_folder, gen_npy_data, gen_root_trans, temporal_mask=None)
            # break

def lerp_root_trajectory(root_trajectory, temporal_mask):
    # root_trajcetory: T X 3
    # temporal_mask: T
    root_trajectory = torch.from_numpy(root_trajectory).float()
    temporal_mask = torch.from_numpy(temporal_mask).float()

    timesteps = root_trajectory.size()[0]
    times = list(range(timesteps))

    temporal_idx = torch.nonzero(temporal_mask).squeeze(-1) # K 
    selected_root_data = torch.index_select(root_trajectory, 0, temporal_idx) # K X 3 

    num_dim = selected_root_data.size()[1]
    lerp_res = np.zeros((timesteps, 3)) # T X 3
    for dim_idx in range(num_dim):
        lerp_res[:, dim_idx] = np.interp(times, temporal_idx.data.cpu().numpy(), selected_root_data.cpu().numpy()[:, dim_idx])

    # lerp_res = torch.from_numpy(lerp_res).float()

    return lerp_res #  T X 3

def prep_mesh_for_motion_interpolation(data_folder):
    sub_folders = os.listdir(data_folder)
    for sub_name in sub_folders:
        # sub_folder_path = os.path.join(data_folder, sub_name, "0/0/temporal_interp_w_trajectory")
        sub_folder_path = os.path.join(data_folder, sub_name)
        if "Jog" in sub_name or "Walk" in sub_name:
            npy_files = os.listdir(sub_folder_path)

            for f_name in npy_files:
                if "_rot_" in f_name and ".npy" in f_name and "opt" in f_name:            
                    rot_npy_path = os.path.join(sub_folder_path, f_name)
                    root_trans_path = rot_npy_path.replace("_rot_", "_root_trans_")

                    out_folder = os.path.join(sub_folder_path, f_name.replace(".npy", ""))
                    if not os.path.exists(out_folder):
                        os.makedirs(out_folder)
                    
                    gen_npy_data = np.load(rot_npy_path)
                    gen_root_trans = np.load(root_trans_path)[:, 0, :]
                    
                    timesteps = gen_npy_data.shape[0]
                    temporal_mask = np.zeros(timesteps)
                    temporal_mask[::15] = 1
                    save_mesh_obj(out_folder, gen_npy_data, gen_root_trans, temporal_mask)

                # if "_rot_" in f_name and "slerped" in f_name and ".npy" in f_name:
                #     # Generate for lerp trajectory 
                #     rot_npy_path = os.path.join(sub_folder_path, f_name)
                #     root_trans_path = rot_npy_path.replace("_rot_", "_root_trans_")
                #     gt_trans_path = root_trans_path.replace("slerped", "gt")

                #     out_folder = os.path.join(sub_folder_path, "lerp_"+f_name.replace(".npy", ""))
                #     if not os.path.exists(out_folder):
                #         os.makedirs(out_folder)
                    
                #     timesteps = gen_npy_data.shape[0]
                #     temporal_mask = np.zeros(timesteps)
                #     temporal_mask[::5] = 1

                #     gen_npy_data = np.load(rot_npy_path)
                #     # gen_root_trans = np.load(root_trans_path)[:, 0, :]
                #     gt_root_trans = np.load(gt_trans_path)[:, 0, :]
                #     gen_root_trans = lerp_root_trajectory(gt_root_trans, temporal_mask)

                #     save_mesh_obj(out_folder, gen_npy_data, gen_root_trans, temporal_mask)


def prep_mesh_for_motion_completion(data_folder):
    seq_names = os.listdir(data_folder)

    for seq_name in seq_names:
        # if "Transitions" not in seq_name:
        if "tmp" in seq_name:
            sub_folder_path = os.path.join(data_folder, seq_name)


            # try_folders = os.listdir(sub_folder_path)

            # for try_idx in range(len(try_folders)):
            #     sub_try_name = str(try_idx)
            #     real_folder_path = os.path.join(sub_folder_path, sub_try_name, "0/temporal_interp_w_trajectory")
        
            real_folder_path = sub_folder_path

            npy_files = os.listdir(real_folder_path)
            for npy_name in npy_files:
                if ".npy" in npy_name and ".mp4" not in npy_name and "_rot_" in npy_name and "opt" in npy_name:
                # if "1_vibe" in npy_name:
                    rot_npy_path = os.path.join(real_folder_path, npy_name)
                    root_trans_path = rot_npy_path.replace("_rot_", "_root_trans_")

                    tag = None
                    if "gt" in npy_name:
                        tag = "gt"
                    elif "opt" in npy_name:
                        tag = "opt"

                    # out_folder = os.path.join(sub_folder_path, "processed_res", sub_try_name+"_"+tag)
                    out_folder = real_folder_path

                    if not os.path.exists(out_folder):
                        os.makedirs(out_folder)
                    
                    gen_npy_data = np.load(rot_npy_path)
                    # gen_root_trans = np.load(root_trans_path)[:, 0, :]
                    gen_root_trans = np.zeros((gen_npy_data.shape[0], 3))
                    save_mesh_obj(out_folder, gen_npy_data, gen_root_trans, temporal_mask=None)

# def prep_mesh_for_sampled_seq(data_folder):
#     npy_files = os.listdir(data_folder)

#     for npy_name in npy_files:
#         # if "_rot.npy" in npy_name:
#         if "_rot_mat.npy" in npy_name
#             rot_npy_path = os.path.join(data_folder, npy_name)
#             root_trans_path = rot_npy_path.replace("_rot_mat.npy", "_root_trans.npy")

#             out_folder = os.path.join(data_folder, "for_vis")
            
#             if not os.path.exists(out_folder):
#                 os.makedirs(out_folder)
            
#             gen_npy_data = np.load(rot_npy_path)
#             gen_root_trans = np.load(root_trans_path)
#             print("trans shape:{0}".format(gen_root_trans.shape))
            
#             save_mesh_obj(out_folder, gen_npy_data, gen_root_trans, temporal_mask=None)

def check_selected_name(npy_name):
    # if "rampAndStairs" in npy_name or "runForBus" in npy_name or "walkBridge" in npy_name or "walking" in npy_name or "walkUphill" in npy_name \
    #     or "windowShopping" in npy_name or "flat_guitar" in npy_name or "fencing" in npy_name:
    if "rampAndStairs" in npy_name or "runForBus" in npy_name or "walking" in npy_name  \
        or "windowShopping" in npy_name or "fencing" in npy_name:
        return True
    return False

def adjust_root_rot(ori_seq_data):
    # ori_seq_data: bs X T X 24 X 3 X 3
    bs, timesteps, _, _, _ = ori_seq_data.size()
    # target_root_rot = torch.eye(3).cuda() # 3 X 3
    target_root_rot = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    target_root_rot = torch.from_numpy(target_root_rot).float()
    target_root_rot = target_root_rot[None, :, :].repeat(bs, 1, 1) # bs X 3 X 3
    
    # ori_root_rot = ori_seq_data[:, 0, 0, :, :] # bs x 3 X 3
    ori_root_rot = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    ori_root_rot = torch.from_numpy(ori_root_rot).float()
    ori_root_rot = ori_root_rot[None, :, :].repeat(bs, 1, 1) # bs X 3 X 3
    relative_rot = torch.matmul(target_root_rot, ori_root_rot.transpose(1, 2)) # bs X 3 X 3

    relative_rot = relative_rot[:, None, :, :].repeat(1, timesteps, 1, 1) # bs X T X 3 X 3
    # print("relative_rot:{0}".format(relative_rot[0,0]))

    converted_seq_data = torch.matmul(relative_rot.view(-1, 3, 3), ori_seq_data[:, :, 0, :, :].view(-1, 3, 3)) # (bs*T) X 3 X 3
    converted_seq_data = converted_seq_data.view(bs, timesteps, 3, 3)

    dest_seq_data = ori_seq_data.clone()
    # print("dest seq:{0}".format(dest_seq_data.size()))
    # print("converted seq data:{0}".format(converted_seq_data.size()))
    dest_seq_data[:, :, 0, :, :] = converted_seq_data

    return dest_seq_data, relative_rot 
    # bs X T X 24 X 3 X 3, bs X T X 3 X 3

def prep_mesh_for_vibe_cmp(data_folder):
    npy_files = os.listdir(data_folder)

    for npy_name in npy_files:
        if ".npy" in npy_name and "objs" not in npy_name and "rot" in npy_name and "filter" in npy_name and \
            ("downtown_walkUphill" in npy_name or "weeklyMarket" in npy_name or "fencing" in npy_name):
        # if ".npy" in npy_name and "objs" not in npy_name and "rot" in npy_name:
            rot_npy_path = os.path.join(data_folder, npy_name)
            root_trans_path = rot_npy_path.replace("_rot_", "_root_trans_")

            out_folder = os.path.join(data_folder, npy_name.replace(".npy", "")+"_wo_trans_objs")
            
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            
                gen_npy_data = np.load(rot_npy_path) # T X 24 X 3 X 3
                timesteps = gen_npy_data.shape[0]

                rot_mat = torch.from_numpy(gen_npy_data).float()
                rot_mat, _ = adjust_root_rot(rot_mat[None]) # 1 X T X 24 X 3 X 3
                # ori_seq_data: bs X T X 24 X 3 X 3
                rot_mat = rot_mat.squeeze(0)
                
                gen_root_trans = np.zeros((timesteps, 3))
                # gen_root_trans = np.load(root_trans_path)[:, 0, :] # T X 3

                save_mesh_obj(out_folder, rot_mat.data.cpu().numpy(), gen_root_trans, temporal_mask=None)

def prep_mesh_for_interp_z(data_folder):
    sub_folders = os.listdir(data_folder)
    for sub_name in sub_folders:
        real_folder = os.path.join(data_folder, sub_name)
        npy_files = os.listdir(real_folder)

        for npy_name in npy_files:
            rot_npy_path = os.path.join(real_folder, npy_name)

            out_folder = os.path.join(real_folder, npy_name.replace(".npy", "")+"_objs")
            
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            
                gen_npy_data = np.load(rot_npy_path) # T X 24 X 3 X 3
                timesteps = gen_npy_data.shape[0]

                rot_mat = torch.from_numpy(gen_npy_data).float()
                
                gen_root_trans = np.zeros((timesteps, 3))
                # gen_root_trans = np.load(root_trans_path)[:, 0, :] # T X 3

                save_mesh_obj(out_folder, rot_mat.data.cpu().numpy(), gen_root_trans, temporal_mask=None)

def prep_mesh_for_check_z(data_folder):
    sub_folders = os.listdir(data_folder)
    for sub_name in sub_folders:
        real_folder = os.path.join(data_folder, sub_name)
        if "npys" in sub_name:
            npy_files = os.listdir(real_folder)

            for npy_name in npy_files:
                rot_npy_path = os.path.join(real_folder, npy_name)

                if ".npy" in npy_name:
                    out_folder = os.path.join(real_folder, npy_name.replace(".npy", "")+"_objs")
                    
                    if not os.path.exists(out_folder):
                        os.makedirs(out_folder)
                    
                        gen_npy_data = np.load(rot_npy_path) # T X 24 X 3 X 3
                        timesteps = gen_npy_data.shape[0]

                        rot_mat = torch.from_numpy(gen_npy_data).float()
                        
                        gen_root_trans = np.zeros((timesteps, 3))
                        # gen_root_trans = np.load(root_trans_path)[:, 0, :] # T X 3

                        save_mesh_obj(out_folder, rot_mat.data.cpu().numpy(), gen_root_trans, temporal_mask=None)

def prep_mesh_for_motion_rec(data_folder):
    
    npy_files = os.listdir(data_folder)

    npy_files.sort()
    selected_npy_files = []
    for idx in range(len(npy_files)):
        if (idx+1) % 20 == 0 and "Transitions" in npy_files[idx]:
            selected_npy_files.append(npy_files[idx])

    for npy_name in selected_npy_files:
        if "rot_gt" in npy_name:
            npy_name = npy_name.replace("rot_gt", "rot_our")
        elif "rot_our" in npy_name:
            npy_name = npy_name.replace("rot_our", "rot_gt")
        rot_npy_path = os.path.join(data_folder, npy_name)

        if ".npy" in npy_name:
            out_folder = os.path.join(data_folder, npy_name.replace(".npy", "")+"_objs")
            
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            
                gen_npy_data = np.load(rot_npy_path) # T X 24 X 3 X 3
                timesteps = gen_npy_data.shape[0]

                rot_mat = torch.from_numpy(gen_npy_data).float()
                
                gen_root_trans = np.zeros((timesteps, 3))
                # gen_root_trans = np.load(root_trans_path)[:, 0, :] # T X 3

                save_mesh_obj(out_folder, rot_mat.data.cpu().numpy(), gen_root_trans, temporal_mask=None)



if __name__ == "__main__":
    sampled_seq_folder = "./eval_final_single_window_w_trajectory/clean_no_aug_len_64_set_1/0/sampled_single_window"
    # prep_mesh_for_sampled_seq(sampled_seq_folder)

    # motion_completion_root_folder = "/glab2/data/Users/jiaman/adobe/github/ICCV_Paper_Materials/Motion_completion_res_missing_lower"
    # motion_completion_seq_folder = os.path.join(motion_completion_root_folder, "Hier_model")
    motion_completion_root_folder = "/glab2/data/Users/jiaman/adobe/github/ICCV_Paper_Materials/Final_Demo_Materials/Long_Seq_Motion_Completion_AMASS"
    # motion_completion_seq_folder = os.path.join(motion_completion_root_folder, "")
    # prep_mesh_for_motion_completion(motion_completion_seq_folder)

    # motion_interpolation_root_folder = "/glab2/data/Users/jiaman/adobe/github/ICCV_Paper_Materials/Motion_Interpolation_AMASS"
    # motion_interpolation_seq_folder = os.path.join(motion_interpolation_root_folder, "missing_5")
    motion_interpolation_root_folder = "/glab2/data/Users/jiaman/adobe/github/ICCV_Paper_Materials/Final_Demo_Materials/Long_Seq_Motion_Interpolation_AMASS"
    motion_interpolation_seq_folder = os.path.join(motion_interpolation_root_folder, "missing_15")
    # prep_mesh_for_motion_interpolation(motion_interpolation_seq_folder) 

    # sampled_seq_folder = "/glab2/data/Users/jiaman/adobe/github/ICCV_Paper_Materials/Figure_1_Materials/Sampled_seq"
    sampled_seq_folder = "/glab2/data/Users/jiaman/adobe/github/ICCV_Paper_Materials/For_Submit_Supplemental/Sampled_seq"
    if not os.path.exists(sampled_seq_folder):
        sampled_seq_folder = "/mount/Users/jiaman/adobe/github/ICCV_Paper_Materials/For_Submit_Supplemental/Sampled_seq"
    # prep_mesh_for_sampled_seq(sampled_seq_folder)

    vibe_seq_folder = "/glab2/data/Users/jiaman/adobe/github/ICCV_Paper_Materials/Figure_1_Materials/Refine_motion"
    # prep_mesh_for_vibe_cmp(vibe_seq_folder)

    seq_folder = "/glab2/data/Users/jiaman/adobe/github/motion_prior/eval_check_latent_space_w_trajectory/final_two_hier_model_no_aug_len_64_set_11/images/npy_out"
    # prep_mesh_for_sampled_seq(seq_folder)

    vibe_cmp_folder = "/glab2/data/Users/jiaman/adobe/github/VIBE/final_iccv_mesh_vis_our_decoded_3dpw_eval_results/final_two_hier_data_aug_len_8_set_7"
    # prep_mesh_for_vibe_cmp(vibe_cmp_folder)

    check_data_folder = "/glab2/data/Users/jiaman/adobe/github/motion_prior/eval_check_latent_space_w_motion_input/final_two_hier_model_no_aug_len_64_set_11/images"
    # prep_mesh_for_vibe_cmp(check_data_folder)

    dance_cmp_folder = "/glab2/data/Users/jiaman/adobe/github/motion_prior/eval_new_clean_cmp_human_dynamics_outputs/final_two_hier_model_data_aug_len_8_set_7/images"
    # prep_mesh_for_vibe_cmp(dance_cmp_folder)

    # For latent space interpolation
    interp_folder = "/glab2/data/Users/jiaman/adobe/github/ICCV_Paper_Materials/For_Submit_Supplemental/Interpolate_z_seq"
    if not os.path.exists(interp_folder):
        interp_folder = "/mount/Users/jiaman/adobe/github/ICCV_Paper_Materials/For_Submit_Supplemental/Interpolate_z_seq"
    # prep_mesh_for_interp_z(interp_folder)

    check_latent_space_folder = "/glab2/data/Users/jiaman/adobe/github/motion_prior/eval_final_submit_iccv_check_shallow_deep_latent_space/final_two_hier_model_no_aug_len_64_set_1/images/0/11_fix_deep_sample_shallow"
    # prep_mesh_for_check_z(check_latent_space_folder)

    motion_rec_folder = "/glab2/data/Users/jiaman/adobe/github/motion_prior/eval_motion_rec_amass/final_two_hier_model_no_aug_len_8_set_2/images"
    
    # prep_mesh_for_motion_rec(motion_rec_folder)

    vibe_cmp_folder = "/glab2/data/Users/jiaman/adobe/github/VIBE/for_supp_check_filter_final_iccv_mesh_vis_our_decoded_3dpw_eval_results/final_two_hier_data_aug_len_8_set_2_test_filter"
    # prep_mesh_for_vibe_cmp(vibe_cmp_folder)

    vibe_folder = "/glab2/data/Users/jiaman/adobe/github/ICCV_Paper_Materials/Final_Demo_Materials/VIBE_Partial"
    vibe_folder = "/glab2/data/Users/jiaman/adobe/github/ICCV_Paper_Materials/Final_Demo_Materials/Talk_Test"

    prep_mesh_for_motion_completion(vibe_folder)
     # Test mesh visualization
    # out_folder = "./tmp_test_smpl_vis"
    # npy_folder = "/data/jiaman/for_cvpr21/utils/data/processed_all_amass_data"
    # npy_path = os.path.join(npy_folder, "Transitions_mocap_mazen_c3d_walksideways_walk_poses.npy")
    # npy_data = np.load(npy_path)

    # timesteps = 300
    # rot_mat = npy_data[:timesteps, 24*6:24*6+24*3*3] # T X (24*3*3)
    # rot_mat = rot_mat.reshape(-1, 24, 3, 3) # T X 24 X 3 X 3
    
    # root_trans = npy_data[:timesteps, -3:] # T X 3
    
    # vis_rendered_mesh_only(out_folder, rot_mat, root_trans)
    # rot_mat: T X 24 X 3 X 3
    # root_trans: T X 3

    # For generated seq
    # gen_npy_path = "/glab2/data/Users/jiaman/adobe/github/motion_prior/eval_long_seq_generation/no_data_aug_len_64_wo_root_set_1/images/0/amass_sampled_long_seq_gen_overlap_1_vis/rot_19.npy"
    # gen_npy_data = np.load(gen_npy_path)[:timesteps] # T X 24 X 3 X 3
    # gen_root_trans = np.zeros((timesteps, 3)) # T X 3
    
    # npy_folder = "/glab2/data/Users/jiaman/adobe/github/motion_prior/tmp_debug_mesh_vis"
    # rot_npy_path = os.path.join(npy_folder, "rot_res.npy")
    # trans_npy_path = os.path.join(npy_folder, "trans_res.npy")
    # gen_npy_data = np.load(rot_npy_path)
    # gen_root_trans = np.load(trans_npy_path)[:, 0, :]
    # timesteps = gen_npy_data.shape[0]
    # temporal_mask = np.zeros(timesteps)
    # temporal_mask[::30] = 1
    # bg_img = get_interp_target_vis_rendered_mesh_only(out_folder, gen_npy_data, gen_root_trans, temporal_mask)
    # vis_rendered_mesh_only(out_folder, gen_npy_data, gen_root_trans, temporal_mask, bg_img)


    # npy_folder = "/glab2/data/Users/jiaman/adobe/github/Paper_Materials/mesh_interp_res_window"
    # npy_folder = "/glab2/data/Users/jiaman/adobe/github/Paper_Materials/window30_opt_traj_interp"
    # sub_folders = os.listdir(npy_folder)
    # for sub_name in sub_folders:
    #     sub_folder_path = os.path.join(npy_folder, sub_name)

    #     npy_files = os.listdir(sub_folder_path)

    #     for f_name in npy_files:
    #         if "_rot_" in f_name:
            
    #             rot_npy_path = os.path.join(sub_folder_path, f_name)
    #             root_trans_path = rot_npy_path.replace("_rot_", "_root_trans_")

    #             out_folder = os.path.join(sub_folder_path, f_name.replace(".npy", ""))
    #             if not os.path.exists(out_folder):
    #                 os.makedirs(out_folder)
                
    #             gen_npy_data = np.load(rot_npy_path)
    #             gen_root_trans = np.load(root_trans_path)[:, 0, :]
                
    #             temporal_mask = np.zeros(timesteps)
    #             temporal_mask[::30] = 1
    #             save_mesh_obj(out_folder, gen_npy_data, gen_root_trans, temporal_mask)
        
    # seq_name = "HumanEva_S2_Walking_3_poses.npy_block_0"
    # tag_list = ["opt", "gt", "slerped"]
    # tag_list = ["gt"]
    # for tag in tag_list:
    #     out_folder = os.path.join(npy_folder, seq_name)
    #     rot_npy_path = os.path.join(npy_folder, seq_name, seq_name+"_rot_"+tag+"_res.npy")
    #     root_trans_path = os.path.join(npy_folder, seq_name, seq_name+"_root_trans_"+tag+"_res.npy")
    #     gen_npy_data = np.load(rot_npy_path)
    #     gen_root_trans = np.load(root_trans_path)[:, 0, :]
    #     temporal_mask = np.zeros(timesteps)
    #     temporal_mask[::30] = 1
    #     # for_interpolation_vis_rendered_mesh_only(out_folder, gen_npy_data, gen_root_trans, temporal_mask, dest_name_tag=tag+"_only")
    #     # for_interpolation_vis_rendered_mesh_only(out_folder, gen_npy_data, gen_root_trans, temporal_mask, dest_name_tag=tag)
    #     save_mesh_obj(out_folder, gen_npy_data, gen_root_trans, temporal_mask)
    #     break;



    # vibe_pkl = "/glab2/data/Users/jiaman/adobe/github/VIBE/output/tmp_test/vibe_output.pkl"
    # pkl_data = joblib.load(vibe_pkl)
    # vertices = pkl_data[1]['verts'][:timesteps] # timesteps X 68890 X 3
    # cam = pkl_data[1]['orig_cam'][:timesteps] # timesteps X 4
    # debug_vis_rendered_mesh_only(out_folder, vertices, cam, orig_width=224, orig_height=224, orig_img=False, wireframe=False, sideview=False)
