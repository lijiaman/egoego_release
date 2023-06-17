import torch
import numpy as np
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from mujoco_py import load_model_from_path, MjSim
from relive.utils.torch_utils import *
from relive.utils.transform_utils import *
from scipy.spatial.transform import Rotation as sRot



class Humanoid:
    def __init__(self, model_file, dtype, device, off_obj_qpos, obj_names):
        print(model_file)
        self.obj_names = obj_names
        self.model = load_model_from_path(model_file) # load mujoco model 
        self.num_obj = len(obj_names)

        offsets = self.model.body_pos[(self.num_obj + 1):] # exclude world and chair

        parents = self.model.body_parentid[(self.num_obj + 1):] - (self.num_obj + 1)
        parents[0] = -1

    
        """global coordinate to local link coordinate"""
        self._offsets = torch.from_numpy(offsets).type(dtype).to(device)
        
        self._parents = np.array(parents)
        self.body_name = self.model.body_names[(self.num_obj + 1):]


        self.device = device
        self.dtype = dtype
        #chair offset
        self.off_obj_qpos = off_obj_qpos

        self.map_length = 0.6
        self.voxel_num = 32

        self._compute_metadata()
        # self._set_local_map()
        self._set_rotpad_indices()
        self._set_qpos_padding()
        
    def get_head_idx(self):
        return self.model._body_name2id["Head"] - self.num_obj - 1


    def _set_local_map(self):

        x = np.linspace(-self.map_length / 2.0, self.map_length / 2.0, self.voxel_num)
        y = np.linspace(-self.map_length / 2.0, self.map_length / 2.0, self.voxel_num)
        z = np.linspace(-self.map_length / 2.0, self.map_length / 2.0, self.voxel_num)
        X, Y, Z = np.meshgrid(x, y, z)
        self.base_grid = np.concatenate((X[:,:,:,np.newaxis],Y[:,:,:,np.newaxis], Z[:,:,:,np.newaxis]), axis=3)
        self.base_grid = torch.tensor(self.base_grid, dtype=self.dtype, device=self.device)

        obj_idx = self.model.body_names.index(self.obj_names[-1]) # ZL: not cool
        g_ids = [i for i, x in enumerate(self.model.geom_bodyid) if x == obj_idx]

        self.obj_geom_num = len(g_ids)
        self.obj_sizes    = torch.tensor(self.model.geom_size[g_ids], dtype=self.dtype, device=self.device)
        self.obj_loc_pos  = torch.tensor(self.model.geom_pos[g_ids] , dtype=self.dtype, device=self.device)
        self.obj_loc_quat = torch.tensor(self.model.geom_quat[g_ids], dtype=self.dtype, device=self.device)


    def _set_rotpad_indices(self):
        """to convert euler to quaternion: find the index 0 to 1 degrees of freedom joints"""
        self.pad_indices = []
        ind = 0
        for name in self.body_name:
            if name == "Hips":
                continue
            elif name == "LeftLeg" or name == "RightLeg":
                self.pad_indices.append(ind+1) # pad y, z
                ind += 1
            elif name == "LeftForeArm" or name == "RightForeArm":
                self.pad_indices.append(ind) # pad x, y
                ind += 1
            else:
                ind += 3
    
    def _set_qpos_padding(self):
        pad_indices = []
        ind = 0
        for idx in range(len(self.body_name)):
            name = self.body_name[idx]
            if name == "Hips":
                continue
            elif name == "LeftLeg" or name == "RightLeg":
        #         self.pad_indices.append(ind+1) # pad y, z
                pad_indices.append(ind)
                ind += 3
            elif name == "LeftForeArm" or name == "RightForeArm":
                pad_indices.append(ind + 2) # pad x, y
                ind += 3
            else:
                pad_indices += [ind, ind + 1, ind + 2]
                ind += 3
        self.qpos_pad_indices = pad_indices
    
    def _rotation_padding(self, rotations):
        B = rotations.size()[0]
        pad_rot = torch.tensor([0.0, 0.0]*B, dtype=self.dtype, device=self.device).view(-1, 2)

        padded_rot = torch.cat([rotations[:, :self.pad_indices[0]], pad_rot.clone()], dim=1)

        for pid in range(len(self.pad_indices[:-1])):
            _from = self.pad_indices[pid]
            _to = self.pad_indices[pid+1]
            padded_rot = torch.cat([padded_rot, rotations[:, _from:_to]], dim=1)
            padded_rot = torch.cat([padded_rot, pad_rot.clone()], dim=1)

        padded_rot = torch.cat([padded_rot, rotations[:, self.pad_indices[-1]:]], dim=1)
        return padded_rot.view(-1, 3)


    def to4x4(self, rotmat):
        obj_rot_tmp = torch.eye(4).to(self.device).type(self.dtype)
        obj_rot_tmp[:3, :3] = rotmat
        obj_rot = obj_rot_tmp.clone()
        return obj_rot

    def get_body_occup_map(self, qpos, body_name):
        """
        Input qpos: 1 x J (object pos, quat, root pos, root orientation, joint orientation)
        Output occupancy map: J x Vx x Vy x Vz x 1
        """
        tot_num = self.voxel_num * self.voxel_num * self.voxel_num  
        grid = self.base_grid.clone()

        occup_grid_size = (len(body_name), self.voxel_num, self.voxel_num, self.voxel_num)

        root_rot = qpos[self.off_obj_qpos+3:self.off_obj_qpos+7]
        root_pos = qpos[self.off_obj_qpos:self.off_obj_qpos+3]
        joint_rot = qpos[self.off_obj_qpos+7:]

        
        body_pos, body_quat = self.get_body_pos_quat(torch.unsqueeze(qpos[self.off_obj_qpos:], dim=0), select_joints = body_name)
        body_pos = body_pos.view(-1, 3)
        body_quat = body_quat.view(-1, 4)
        
        # obj_pos = qpos[:3]
        # obj_quat = qpos[3:7]
        # obj_rot = quaternion_matrix(obj_quat).t()
        # obj_rot = torch.cat([obj_rot, torch.zeros(3, 1).to(self.device)], dim = 1)
        # obj_rot[:3, 3] = -obj_pos

        ## World -> object root transformation
        obj_pos = qpos[:3]
        obj_quat = qpos[3:7]
        obj_rot = self.to4x4(quaternion_matrix(obj_quat).t())
        
        obj_rot[:3, 3] = -obj_pos

        ## Body -> World transformation
        body_rot = torch.stack([self.to4x4(quaternion_matrix(get_heading_q(b_quat))) for b_quat in body_quat], dim=0)
        body_rot[:, :3, 3] = body_pos

        ## Body -> object root transformation
        body_trans = torch.einsum("bij, bjk -> bik", torch.repeat_interleave(torch.unsqueeze(obj_rot, dim=0), len(body_name), dim=0), body_rot)
        

        ## Object root -> object part transformation
        obj_loc_trans = torch.stack([self.to4x4(quaternion_matrix(quat).T) for quat in self.obj_loc_quat], dim=0)
        obj_loc_trans[:, :3, 3] = -self.obj_loc_pos
        
        ## Body -> object part transformation
        total_trans = torch.einsum("blij, bljk -> blik", torch.repeat_interleave(torch.unsqueeze(obj_loc_trans, dim=1), len(body_name), dim=1), \
                                                         torch.repeat_interleave(torch.unsqueeze(body_trans, dim=0), self.obj_geom_num, dim=0))

        grid_h = torch.cat((grid.view(tot_num, 3).t(), torch.ones(1, tot_num, dtype=self.dtype, device=self.device)), dim=0)
        trans_grid = torch.einsum("bkij,jl->bkil", total_trans, grid_h)[:, :, :-1, :] ## object part, body num, xyz1, point num
        obj_sizes = torch.repeat_interleave(torch.repeat_interleave(torch.unsqueeze(torch.unsqueeze(self.obj_sizes, dim=1), dim=-1), len(body_name), dim=1), tot_num, dim=3)
        cond = torch.abs(trans_grid) < obj_sizes / 2.0
        occup_grid_batch = cond.all(dim=2).any(dim=0).view(occup_grid_size).type(self.dtype).to(self.device)

        return occup_grid_batch.unsqueeze(-1)
    
    
    def get_body_pos_quat(self, qpos, select_joints=None):
        """
        qpos: body representation (1->3: root position, 3->7: root orientation, 7->end: joint orientation)
        Rotations are represented in euler angles. 
        Note that some joints have 1 DoF.
        B = batch size, J = number of joints
        Input: rotations (B, L), root_rotations (B, 4), root_positions (B, 3),
        Output: (B, J, 3) J is number of joints or selected joints 
        """

        rotations = qpos[:, 7:]
        root_rotations = qpos[:, 3:7]
        root_positions = qpos[:, :3]

        assert rotations.size()[0] == root_positions.size()[0]
        B = rotations.size()[0]
        J = len(self.body_name) - 1 # Exclude root

        padded_rot = self._rotation_padding(rotations)
        quats = quaternion_from_euler(padded_rot[:, 0], padded_rot[:, 1], padded_rot[:, 2], axes='rxyz')
        quats = quats.reshape(B, J, 4)


        joint_positions, joint_quaterions = self.forward_kinematics_batch(quats, root_rotations, root_positions)
        if select_joints is None:
            return joint_positions, joint_quaterions
        else:
            ret_joint = []
            ret_quat = []
            for joint in select_joints:
                jidx = self.body_name.index(joint)
                ret_joint.append(joint_positions[:, jidx, :])
                ret_quat.append(joint_quaterions[:, jidx, :])
            ret_joint = torch.cat(ret_joint).view(B, len(select_joints), 3)
            ret_quat  = torch.cat(ret_quat).view(B, len(select_joints), 4)
            return ret_joint, ret_quat


    def forward_kinematics(self, rotations, root_rotations, root_positions):
        positions_world = []
        rotations_world = []
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                positions_world.append(quat_mul_vec(rotations_world[self._parents[i]], self._offsets[i]) + positions_world[self._parents[i]])            
                if self._has_children[i]:
                    rotations_world.append(quaternion_multiply(rotations_world[self._parents[i]], rotations[i]))
                else:
                    rotations_world.append(None)

        return torch.stack(positions_world), torch.stack(rotations_world)


    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """ 
        assert len(rotations.shape) == 3
        assert rotations.size()[-1] == 4
        B = rotations.size()[0]
        J = self._offsets.shape[0]
        positions_world = []
        rotations_world = []
        expanded_offsets = self._offsets.expand(B, J, self._offsets.shape[1])

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                positions_world.append(quat_mul_vec_batch(rotations_world[self._parents[i]], expanded_offsets[:, i, :]) \
                                       + positions_world[self._parents[i]]) 
                rotations_world.append(quaternion_multiply_batch(rotations_world[self._parents[i]], rotations[:, i - 1, :]))


        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.stack(rotations_world, dim=2)

        return positions_world.permute(0, 2, 1), rotations_world.permute(0, 2, 1)


    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)
                
    def qpos_2_6d(self, qpos):
        rotations = qpos[:, 7:]
        root_rotations = qpos[:, 3:7]
        root_positions = qpos[:, :3]
        
        assert rotations.size()[0] == root_positions.size()[0]
        B = rotations.size()[0]
        J = len(self.body_name) - 1 # Exclude root

        
        padded_rot = self._rotation_padding(rotations)
        quats = quaternion_from_euler(padded_rot[:, 0], padded_rot[:, 1], padded_rot[:, 2], axes='rxyz')
        
        quats = quats.reshape(B, J, 4)
        
        rot_6d = convert_quat_to_6d(quats)
        
        return qpos[:, :7], rot_6d
    
    
    def qpos_from_6d(self, orth6d):
        B, J, _ = orth6d.shape
        assert J == 20
        quats = convert_6d_to_quat(orth6d)
        
        ## This will no longer be differentiable 
        quat_numpy = quats.cpu().numpy()
        quat_numpy_flat = quat_numpy.reshape(-1, 4)[:, [1,2,3,0]]
        
        euler_numpy = sRot.from_quat(quat_numpy_flat).as_euler("XYZ").reshape(B, J, -1)
        qpos_numpy = euler_numpy.reshape(B, -1)[:, self.qpos_pad_indices]
        return qpos_numpy
        
    def qpos_from_euler(self, euler_angles):
        B, J, _= euler_angles.shape
        qpos = euler_angles.reshape(B, -1)[:, self.qpos_pad_indices]
        return qpos

        

def get_expert(expert_qpos):
    joint_pos_mujoco = []
    joint_quat_mujoco = []
    occup_map = []
    for i in range(expert_qpos.shape[0]):
        qpos = expert_qpos[i]
        env.data.qpos[:] = qpos.copy()
        env.sim.forward()
        joint_pos = env.get_body_pos().reshape(-1, 3)
        bwquat = env.get_world_body_quat().reshape(-1, 4)
        # _map, _ = env.get_body_occup_map(occup_joints)
        # _map_torch = env_humanoid.get_body_occup_map(torch.tensor(qpos, dtype=dtype, device=device), occup_joints).squeeze()
        # assert np.array_equal(_map, _map_torch.cpu().numpy()), "Map creation error"
        joint_pos_mujoco.append(joint_pos)
        joint_quat_mujoco.append(bwquat)
        
        
    joint_pos_mujoco = np.array(joint_pos_mujoco)
    qpos  = torch.tensor(expert_qpos[:, env.off_obj_qpos:], dtype=dtype, device=device)

    joint_pos_torch, joint_quat_torch = env_humanoid.get_body_pos_quat(qpos)

    assert np.array_equal(np.round(joint_pos_mujoco, 6), np.round(joint_pos_torch.cpu().numpy(), 6)), "Joint position error"
    assert np.array_equal(np.round(joint_quat_mujoco, 6), np.round(joint_quat_torch.cpu().numpy(), 6)), "Joint orientation error"
    print(np.array_equal(np.round(joint_pos_mujoco, 6), np.round(joint_pos_torch.cpu().numpy(), 6)), np.array_equal(np.round(joint_quat_mujoco, 6), np.round(joint_quat_torch.cpu().numpy(), 6)))

if __name__ == "__main__":
    import argparse
    from relive.data_loaders.statereg_dataset import Dataset
    from relive.envs.humanoid_v2 import HumanoidEnv
    from relive.utils.egomimic_config import Config as EgoConfig

    model_id = '1205'

    action = "sit"
    cfg = EgoConfig(action, create_dirs=False, cfg_id = "all_traj_3")

    env = HumanoidEnv(cfg)
    dataset = Dataset(cfg, cfg.meta_id, 'all', 0, 'iter', False, 0)

    dtype=torch.float64
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    obj_names = ["chair"]
    env_humanoid = Humanoid(model_file=cfg.mujoco_model_file, dtype=dtype, device=device, off_obj_qpos=env.off_obj_qpos, obj_names=obj_names)
    occup_joints = ['LeftFoot', 'RightFoot', 'LeftHand', 'RightHand', 'Hips']

    take = dataset.takes[0]
    for expert_qpos in dataset.orig_trajs:
        get_expert(expert_qpos)
