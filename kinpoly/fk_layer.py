"""Based on Daniel Holden code from:
   A Deep Learning Framework for Character Motion Synthesis and Editing
   (http://www.ipab.inf.ed.ac.uk/cgvu/motionsynthesis.pdf)"""

import numpy as np
import os 
import torch
import torch.nn as nn
import json 
import my_tools
class ForwardKinematicsLayer(torch.nn.Module):
    """ Forward Kinematics Layer Class """
    def __init__(self, device=torch.device("cuda"), parents=None, positions=None):
        super().__init__()
        self.b_idxs = None

        if parents is None and positions is None:
            parents_json = "/viscam/u/jiamanli/github/hm-vae/utils/data/joint24_parents.json"
            pos_npy = "/viscam/u/jiamanli/github/hm-vae/utils/data/skeleton_offsets.npy"
           
            ori_parents_list = json.load(open(parents_json, 'r'))
            self.parents = []
            for p in ori_parents_list:
                self.parents.append(torch.LongTensor([p])[0].to(device))
            self.parents = torch.stack(self.parents).to(device) # K   
            self.positions = torch.from_numpy(np.load(pos_npy)).float()[None, :, :].to(device) # 1 X 24 X 3
        else:
            self.parents = []
            for p in parents:
                self.parents.append(torch.LongTensor([p])[0].to(device))
            self.parents = torch.stack(self.parents).to(device) # K
            self.positions = torch.from_numpy(positions).float()[None, :, :].to(device)

        self.device = device

    def rotate(self, t0s, t1s):
        return torch.matmul(t0s, t1s)

    def identity_rotation(self, rotations):
        # rotations: bs X n_joints X rot_dim(3,4,6)
        diagonal = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0])).to(self.device) # 4 X 4
        diagonal = torch.reshape(
            diagonal, torch.Size([1] * len(rotations.shape[:-1]) + [4, 4])) # 1 X 1 X 4 X 4
        ts = diagonal.repeat(rotations.shape[:-1] + torch.Size([1, 1]))
        return ts # bs X n_joints X 4 X 4

    def make_fast_rotation_matrices(self, positions, rotations):
        # positions: bs X n_joints X 3
        # rotations: bs X n_joints X rot_dim(6, 3 X 3)
        if rotations.size()[-1] == 3 and rotations.size()[-2] == 3:
            rot_matrices = rotations # bs X n_joints X 3 X 3
        elif rotations.size()[-1] == 6:
            rot_matrices = my_tools.rotation_matrix_from_ortho6d(rotations) # bs X n_joints X 3 X 3
       
        rot_matrices = torch.cat([rot_matrices, positions[..., None]], dim=-1) # bs X n_joints X 3 X 4
        zeros = torch.zeros(rot_matrices.shape[:-2] + torch.Size([1, 3])).to(self.device) # bs X n_joints X 1 X 3   
        ones = torch.ones(rot_matrices.shape[:-2] + torch.Size([1, 1])).to(self.device) # bs X n_joints X 1 X 1
        zerosones = torch.cat([zeros, ones], dim=-1) # bs X n_joints X 1 X 4
        rot_matrices = torch.cat([rot_matrices, zerosones], dim=-2) # bs X n_joints X 4 X 4
        
        return rot_matrices # bs X n_joints X 4 X 4

    def rotate_global(self, parents, positions, rotations):
        locals = self.make_fast_rotation_matrices(positions, rotations) # bs X n_joints X 4 X 4
        if rotations.shape[-1] == 3 and rotations.shape[-2] == 3:
            rotations = rotations.view(rotations.shape[0], rotations.shape[1], -1) # bs X n_joints X 9
        globals = self.identity_rotation(rotations) # bs X n_joints X 4 X 4

        globals = torch.cat([locals[:, 0:1], globals[:, 1:]], dim=1) # bs X 1 X 4 X 4 cat with bs X (n_joints-1) X 4 X 4 --> bs X n_joints X 4 X 4
        b_size = positions.shape[0]
        if self.b_idxs is None:
            self.b_idxs = torch.LongTensor(np.arange(b_size)).to(self.device)
        elif self.b_idxs.shape[-1] != b_size:
            self.b_idxs = torch.LongTensor(np.arange(b_size)).to(self.device)

        for i in range(1, positions.shape[1]): # iterate n_joints
            globals[:, i] = self.rotate(
            globals[self.b_idxs, parents[i]], locals[:, i])

        return globals # bs X n_joints X 4 X 4

    def forward(self, rotations, positions=None, return_rot=False):
        # parents: a list, each element represent each joint's parent idx
        # positions: bs X n_joints X 3
        # rotations: bs X n_joints X rot_dim(6, 3 X 3)
        # Get the full transform with rotations for skinning
        bs = rotations.size()[0]
        if positions is None:
            positions = self.positions.repeat(bs, 1, 1)
        transforms = self.rotate_global(self.parents, positions, rotations) # bs X n_joints X 4 X 4
        coordinates = transforms[:, :, :3, 3] # bs X n_joints X 3
    
        if return_rot:
            return coordinates, transforms[:, :, :3, :3] # B X n_joint X 3 X 3  
        else:
            return coordinates 