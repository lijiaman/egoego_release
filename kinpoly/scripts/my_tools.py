import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

def normalize_vector(v):
    # return v.div(torch.norm(v, dim=-1)[..., None])
    return F.normalize(v, dim=-1, eps=1e-6)

def cross_product(u, v):
    i = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    j = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    k = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    out = torch.cat((i[..., None], j[..., None], k[..., None]), dim=-1)
    
    return out

def rotation_matrix_from_ortho6d(poses):
    # poses: bs X n_joints X 6
    # Convert from vibe 6d to ours 6d
    # bs, n_joints, _ = poses.size()
    # poses = poses.view(bs, n_joints, 3, 2)
    # poses = poses.transpose(2, 3) # bs X n X 2 X 3
    # poses = poses.contiguous().view(bs, n_joints, -1) # bs X n X 6

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

def rot6d_to_rotmat_spin(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)

    # inp = a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1
    # denom = inp.pow(2).sum(dim=1).sqrt().unsqueeze(-1) + 1e-8
    # b2 = inp / denom

    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

# From vibe for debugging
def rot6d_to_rotmat(x):
    # x = x.view(-1, 2, 3)
    # x = x.transpose(1, 2)
    
    x = x.view(-1,3,2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)
    # print("x[:, :, 0]:{0}".format(x[:, :, 0]))
    # print("b1:{0}".format(b1))
    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats

if __name__ == "__main__":
    test_npy = "/Users/jiaman/adobe_intern/github/motion_prior/utils/data/processed_all_amass_data/EKUT_300_PushStandOpen_19_poses.npy"
    npy_data = torch.from_numpy(np.load(test_npy)) # N X 578
    rot_6d = npy_data[:1, :6] # 8 X 6 (bs X 6)

    rot_6d = rot_6d.view(1, 2, 3)
    rot_6d = rot_6d.transpose(1, 2) # 1 X 3 X 2
    rot_6d = rot_6d.contiguous().view(1, 6) # 1 X 6
    
    vibe_mat = rot6d_to_rotmat(rot_6d)
    # spin_mat = rot6d_to_rotmat_spin(rot_6d)
    our_mat = rotation_matrix_from_ortho6d(rot_6d[:, None, :])
    print("vibe res:{0}".format(vibe_mat))
    # print("spin mat:{0}".format(spin_mat))
    print("our res:{0}".format(our_mat))
