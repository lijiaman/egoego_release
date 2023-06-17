import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from scipy.spatial.transform import Rotation as sRot

def normalize_vector(v):
    # return v.div(torch.norm(v, dim=-1)[..., None])
    return F.normalize(v, dim=-1, eps=1e-6)

def cross_product(u, v):
    i = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    j = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    k = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    out = torch.cat((i[..., None], j[..., None], k[..., None]), dim=-1)
    
    return out

def rot_mat_to_6d(rotMatrices):
    # rotMatrices:  B X J X 3 X 3  
    # return: B X J X 6 
    # Convert rotation matrix to 6D representation
    cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # bs X 24 X 2 X 3
    cont6DRep = cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # bs X 24 X 6

    return cont6DRep 
    
def rot_6d_to_mat(poses):
    # poses: bs X n_joints X 6
    # return: bs X n_joints X 3 X 3 
    x_raw = poses[..., 0:3] # bs X n_joints X 3
    # print("x_raw:{0}".format(x_raw))
    y_raw = poses[..., 3:6] # bs X n_joints X 3

    x = normalize_vector(x_raw) # bs X n_joints X 3
    # print("x normal:{0}".format(x))
    z = cross_product(x, y_raw) # bs X n_joints X 3
    z = normalize_vector(z)
    y = cross_product(z, x)

    matrix = torch.cat((x[..., None], y[..., None], z[..., None]), dim=-1) # bs X n_joints X 3 X 3
    
    return matrix

def rot6d2euler(rot_6d):
    # rot6d: B X J X 6 
    # return: B X J X 3
    curr_pose_mat = rot_6d_to_mat(rot_6d) # B X J X 3 X 3  
    bs, num_joints, _, _ = curr_pose_mat.shape 
    curr_spose = sRot.from_matrix(curr_pose_mat.reshape(-1, 3, 3).data.cpu().numpy())
    curr_spose_euler = curr_spose.as_euler("ZYX", degrees=False).reshape(bs, num_joints, -1) # B X J X 3 

    return curr_spose_euler 
    
