import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from torch import nn
from collections import defaultdict
import joblib

import time 

from relive.utils.torch_ext import *
from relive.models.rnn import RNN
from relive.models.resnet import ResNet
from relive.models.mobile_net import MobileNet
from relive.models.mlp import MLP
from relive.utils.torch_utils import (get_heading_q, quaternion_multiply, quaternion_inverse, get_heading_q_batch,transform_vec_batch, quat_from_expmap_batch,
                quat_mul_vec_batch, get_qvel_fd_batch, transform_vec, rotation_from_quaternion, de_heading_batch, quat_mul_vec, quat_from_expmap, quaternion_multiply_batch, quaternion_inverse_batch)
    
from relive.utils.torch_smpl_humanoid import Humanoid
from relive.utils.compute_loss import pose_rot_loss, root_pos_loss, root_orientation_loss, end_effector_pos_loss, linear_velocity_loss, angular_velocity_loss, action_loss, position_loss, orientation_loss

class TrajARNet(nn.Module):

    def __init__(self, cfg, data_sample, device, dtype, mode = "train"):
        super(TrajARNet, self).__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.specs = cfg.model_specs
        self.mode = mode
        self.base_rot =  torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])
        self.model_v = self.specs.get("model_v", 1)
        self.pose_delta = self.specs.get("pose_delta", False)

        self.gt_rate = 0
        self.fk_model = Humanoid(model_file=cfg.mujoco_model_file)
        
        self.htype = htype = self.specs.get("mlp_htype", "relu")
        self.mlp_hsize = mlp_hsize = self.specs.get("mlp_hsize", [1024, 512])
        self.rnn_hdim = rnn_hdim = self.specs.get("rnn_hdim", 512)
        self.rnn_type = rnn_type = self.specs.get("rnn_type", "gru")
        self.cnn_fdim = cnn_fdim = self.specs.get("cnn_fdim", 128)

        self.sim = dict()
        self.get_dim(data_sample)

        if self.model_v == 1 or self.model_v == 0:
            self.action_rnn = RNN(self.state_dim, rnn_hdim, rnn_type)
            self.action_rnn.set_mode('step')
            self.action_mlp = MLP(rnn_hdim + self.state_dim, mlp_hsize, htype)
            self.action_fc = nn.Linear(mlp_hsize[-1], self.action_dim)
        elif self.model_v == 2:
            self.action_mlp = MLP(self.state_dim, mlp_hsize, htype)
            self.action_fc = nn.Linear(mlp_hsize[-1], self.action_dim)

        # if self.cfg.use_of:
            # self.cnn = ResNet(cnn_fdim, running_stats=False, pretrained=True)

        print("********************************")
        print("Traj AR Net!")
        print(f"use_of: {self.cfg.use_of}")
        print(f"use_action: {self.cfg.use_action}")
        print(f"use_vel: {self.cfg.use_vel}")
        print(f"use_context: {self.cfg.use_context}")
        print(f"add_noise: {self.cfg.add_noise}")
        print(f"pose_delta: {self.pose_delta}")
        print("********************************")
        #     self.cnn = ResNet(cnn_fdim, running_stats=False, pretrained=True)
        #     for param in self.cnn.parameters():
        #         param.requires_grad = False

        self.context_rnn = RNN(self.context_dim, rnn_hdim, rnn_type)
        self.context_rnn.set_mode('batch')
        self.context_mlp = MLP(rnn_hdim, mlp_hsize, htype)
        self.context_fc = nn.Linear(mlp_hsize[-1], self.init_dim)
        self.qpos_lm = 74
        self.qvel_lm = 75
        self.pose_start = 7

        # Netural data
        
        self.netural_data = joblib.load(osp.join(cfg.data_dir, "standing_neutral.pkl"))
        fk_res = self.fk_model.qpos_fk(torch.from_numpy(self.netural_data['qpos'][None, ]).to(device).type(dtype))
        fk_res['qvel'] = (torch.from_numpy(self.netural_data['qvel']).to(device).type(dtype))
        self.netural_target = fk_res
        
    def to(self, device):
        self.device = device
        super().to(device)
        return self


    def set_schedule_sampling(self, gt_rate):
        self.gt_rate = gt_rate
        
    
    def get_dim(self, data):
        qpos_curr = data[f'qpos'][:, 0, :]
        zero_qpos = torch.zeros(qpos_curr.shape).to(self.device).type(self.dtype)
        zero_qpos[:,3] = 1
        zero_qvel = torch.zeros(data[f'qvel'][:,0,:].shape).to(self.device).type(self.dtype)
        self.set_sim(zero_qpos, zero_qvel)
        
        state, _ = self.get_obs(data, 0)
        self.state_dim = state.shape[-1]
        self.action_dim = data[f'target'].shape[-1]
        self.init_dim = self.action_dim + zero_qvel.shape[-1]

        self.context_dim = self.get_context_dim(data)
        print(f"Context dim: {self.context_dim}, State dim: {self.state_dim}, Init dim: {self.init_dim}, Action dim: {self.action_dim}")
        
    def set_sim(self, qpos, qvel = None):
        self.sim["qpos"] = qpos if torch.is_tensor(qpos) else torch.from_numpy(qpos).to(self.device).type(self.dtype)
        if not qvel is None:
            self.sim["qvel"] = qvel if torch.is_tensor(qvel) else torch.from_numpy(qvel).to(self.device).type(self.dtype)
        else:
            self.sim["qvel"] = torch.zeros(self.qvel_lm).to(self.device).type(self.dtype)

    def get_context_dim(self, data):
        context_d = 0
        # if self.cfg.use_of: context_d += self.cnn_fdim
        if self.cfg.use_of: context_d += data['of'].shape[-1]
        if self.cfg.use_head: context_d += data['obj_head_relative_poses'].shape[-1] + data['head_vels'].shape[-1]
        if self.cfg.use_action: context_d += data['action_one_hot'].shape[-1]
        return context_d


    def remove_base_rot_batch(self, quat):
        base_rot_batch = self.base_rot.repeat(quat.shape[0], 1)
        return quaternion_multiply_batch(quat, quaternion_inverse_batch(base_rot_batch))

    def add_base_rot_batch(self, quat):
        base_rot_batch = self.base_rot.repeat(quat.shape[0], 1)
        return quaternion_multiply_batch(quat, base_rot_batch)

    def get_context_feat(self, data):
        data_acc = []
        if self.cfg.use_of:
            # of_data = data['of']
            # of_data = torch.cat((of_data, torch.zeros(of_data.shape[:-1] + (1,), device=data['of'].device)), dim=-1)
            # batch, seq_len, h, w, c = of_data.shape
            # of_data = of_data.reshape(-1, h, w, c).permute(0, 3, 1, 2)
            # cnn_feat = self.cnn(of_data).reshape(batch, seq_len, self.cnn_fdim)
            # data_acc.append(cnn_feat)
            data_acc.append(data['of'])

        
        if self.cfg.use_head:
            data_acc.append(data['obj_head_relative_poses'][:, :data['head_vels'].shape[1]])
            data_acc.append(data['head_vels'])
        
        if self.cfg.use_action:
            data_acc.append(data['action_one_hot'])

        # import pdb 
        # pdb.set_trace()
        # if data_acc[0].shape[1] != data_acc[1].shape[1]:
        #     import pdb 
        #     pdb.set_trace()  
        context_feat = torch.cat(data_acc, dim = 2).to(self.device).type(self.dtype)

        batch_size, seq_len, _ = context_feat.shape
        self.context_rnn.initialize(batch_size)
        context_feat_rnn = self.context_rnn(context_feat.permute(1, 0, 2).contiguous())
        context_feat_rnn = context_feat_rnn.permute(1, 0, 2) # B X T X D 
        return context_feat_rnn

    def init_pred_qpos(self, init_pred_state, data):
        init_pos, init_rot = data['qpos'][:,0,:3], data['qpos'][:,0,3:7]

        # init_heading = get_heading_q_batch(self.remove_base_rot_batch(init_rot))
        init_heading = get_heading_q_batch(init_rot)
        pred_qpos = torch.cat([init_pos[:,:2], init_pred_state[:, :self.qpos_lm]], dim=1)
        pred_qpos_root = quaternion_multiply_batch(init_heading, pred_qpos[:, 3:7])

        pred_qpos_root_norm = pred_qpos_root/torch.norm(pred_qpos_root, dim = 1).view(-1, 1)
        pred_qpos[:, 3:7] = pred_qpos_root_norm
        # pred_qpos[:, 3:7] = self.add_base_rot_batch(pred_qpos[:, 3:7])

        return pred_qpos

    def init_states(self, data):

        batch_size, seq_len, _ = data['qpos'].shape # 

        context_feat_rnn = self.get_context_feat(data) # each timestep contains the information of the head diff between current step and next step. 
        data['context_feat_rnn'] = context_feat_rnn
        context_feat_rnn_mean = context_feat_rnn.mean(dim = 1)

        if self.model_v == 1 or self.model_v == 0:
            self.action_rnn.initialize(batch_size)
        elif self.model_v == 2:
            pass
            
        init_state = self.context_fc(self.context_mlp(context_feat_rnn_mean)) # Need a loss on this directly, full init states, qvel and qpos
        init_pred_state, init_pred_vel = init_state[:,:self.action_dim], init_state[:,self.action_dim:]
        qpos_cur = self.init_pred_qpos(init_pred_state, data)

        self.set_sim(qpos_cur, init_pred_vel)
        data['init_qpos'] = qpos_cur
        data['init_qvel'] = init_pred_vel

        return data
    
    def get_obs(self, data, t):
        # Everything in obs need to be available in test time
        obs = []
        batch_size, seq_len, _ = data['qpos'].shape


        curr_qpos = self.sim['qpos'].clone() # Simulation contains the full qpos
        curr_qvel = self.sim['qvel'].clone() # if use GT data, for qpos x0, current qvel is x1-x0. contains all the information to get qpos x1 in next step. 

        
        fk_res = self.fk_model.qpos_fk(curr_qpos)
        pred_wbpos, pred_wbquat, pred_bquat = fk_res['wbpos'], fk_res['wbquat'], fk_res['bquat']
        curr_qpos_local = curr_qpos.clone()
        # curr_qpos_local[:, 3:7] = self.remove_base_rot_batch(curr_qpos_local[:,3:7])   
        curr_qpos_local[:,3:7] = de_heading_batch(curr_qpos_local[:,3:7])
        
        # get pred_head
        head_idx = self.fk_model.get_head_idx()
        pred_hrot = pred_wbquat[:,head_idx]
        pred_hpos = pred_wbpos[:,head_idx]

        # pred_hrot_heading = self.remove_base_rot_batch(pred_hrot)
        pred_hrot_heading = pred_hrot # assign world space head rotation to head rotation heading. 
       
        if self.cfg.use_context or self.cfg.use_of: # how can this happen???
            if 'context_feat_rnn' in data:
                obs.append(data['context_feat_rnn'][:, t, :]) # in timestep t, the context has information of next step's head pose. 
            else: # this is for getting orbs before doing the thing
                obs.append(torch.zeros((batch_size, self.rnn_hdim)).type(self.dtype).to(self.device))

        if self.cfg.use_head:
            # get target head
            t_hrot = data['head_pose'][:, t, 3:].clone() # t represents current step's GT head pose 
            t_hpos = data['head_pose'][:, t, :3].clone()
            # get target head vel
            t_havel = data['head_vels'][:, t, 3:].clone() # for head pose h0, head vel in this step represents h1-h0. 
            t_hlvel = data['head_vels'][:, t, :3].clone()
            t_obj_relative_head = data["obj_head_relative_poses"][:,t,:].clone()
            
            if self.cfg.add_noise and self.mode == "train":
                t_hrot += torch.empty(t_hrot.shape, dtype=self.dtype, device=self.device).normal_(mean=0.0,std=self.cfg.noise_std)
                t_hpos += torch.empty(t_hpos.shape, dtype=self.dtype, device=self.device).normal_(mean=0.0,std=self.cfg.noise_std)
                t_havel += torch.empty(t_havel.shape, dtype=self.dtype, device=self.device).normal_(mean=0.0,std=self.cfg.noise_std)
                t_hlvel += torch.empty(t_hlvel.shape, dtype=self.dtype, device=self.device).normal_(mean=0.0,std=self.cfg.noise_std)
                t_obj_relative_head += torch.empty(t_obj_relative_head.shape, dtype=self.dtype, device=self.device).normal_(mean=0.0,std=self.cfg.noise_std)

            # difference in head, in head's heading coordinates
            diff_hpos = t_hpos - pred_hpos # current step's prediction difference with GT. 
            diff_hpos = transform_vec_batch(diff_hpos, pred_hrot_heading, "heading")
            diff_hrot = quaternion_multiply_batch(quaternion_inverse_batch(t_hrot), pred_hrot)
            
        # get_obj_relative_to_current_head
        q_heading = get_heading_q_batch(pred_hrot_heading).clone()
        obj_rot = data['obj_pose'][:, t, 3:7].clone() # ZL: only doing the first object yeah !!!!!
        obj_pos = data['obj_pose'][:, t, :3].clone()
        diff_obj = obj_pos - pred_hpos
        diff_obj_loc = transform_vec_batch(diff_obj, pred_hrot_heading, "heading")
        obj_rot_local = quaternion_multiply_batch(quaternion_inverse_batch(q_heading), obj_rot) # Object local coordinate
        pred_obj_relative_head = torch.cat([diff_obj_loc, obj_rot_local], dim = 1)
        
        # state 
        # order of these matters !!! 74 + 75 + 3 + 4 + 7 + 3 + 3 + 7
        obs.append(curr_qpos_local[:, 2:]) # current height + local body orientation + body pose 74
        if self.cfg.use_vel:
            obs.append(curr_qvel) # current velocities 75, shouldn't be used????!!!! since the GT qvel contains all the information for next step. 

        if self.cfg.use_head:
            obs.append(diff_hpos) # diff head position 3, current step's head position diff? why not next step diff? 
            obs.append(diff_hrot) # diff head rotation 4, current step's head rotation diff?

        obs.append(pred_obj_relative_head) # predicted object relative to head 7

        if self.cfg.use_head:
            obs.append(t_havel)   # target head angular velocity 3, here, represents the difference between next step and current step GT. 
            obs.append(t_hlvel)   # target head linear  velocity 3
            obs.append(t_obj_relative_head)  # target object relative to head 7 

        ################################################################################
        # if self.cfg.use_action and self.model_v > 0:
        #     obs.append(data['action_one_hot'][:, t, :])
        #     # print(data['action_one_hot'][:, t, :].shape)

        # if self.cfg.use_of:
        #     # Not sure what to do yet......
        #     obs.append(data['of'][:, t, :])
        
        
        obs = torch.cat(obs, dim = 1)

        return obs, {"pred_wbpos": pred_wbpos, "pred_wbquat": pred_wbquat,  "pred_rot": pred_bquat,
                     "qvel": curr_qvel, "qpos": curr_qpos, "obj_2_head": pred_obj_relative_head}
    

    
    def step(self, action, dt = 1/30):
        curr_qpos = self.sim['qpos'].clone()
        curr_qvel = self.sim['qvel'].clone()
        
        curr_pos, curr_rot = curr_qpos[:, :3], curr_qpos[:, 3:7]
        
        # curr_rot = self.remove_base_rot_batch(curr_rot)
        curr_heading = get_heading_q_batch(curr_rot)
            
        body_pose = action[:, (self.pose_start-2):self.qpos_lm].clone() # 69 dim 
        if self.pose_delta:
            body_pose = body_pose + curr_qpos[:, self.pose_start:]
            body_pose[body_pose>np.pi] -= 2 * np.pi
            body_pose[body_pose<-np.pi] += 2 * np.pi
            
            
        # action is: root_z(1), root_quat(4)(seems not used?), euler angle(69), root linear v(3), root angular v(3) = 80 dim.   
        if self.cfg.has_z:
            next_qpos = torch.cat([curr_pos[:,:2], action[:, :(self.pose_start-2)], body_pose], dim=1) # 2 + 5 + 69

            root_qvel = action[:, self.qpos_lm:]
            linv = quat_mul_vec_batch(curr_heading, root_qvel[:, :3]) 
            next_qpos[:, :2] += linv[:, :2] * dt
        else:
            next_qpos = torch.cat([curr_pos[:,:3], action[:, :4], body_pose], dim=1)
            root_qvel = action[:, self.qvel_lm:]
            linv = quat_mul_vec_batch(curr_heading, root_qvel[:, :3]) 
            next_qpos[:, :3] += linv[:, :3] * dt
        
        
        angv = quat_mul_vec_batch(curr_rot, root_qvel[:, 3:6])
        new_rot = quaternion_multiply_batch(quat_from_expmap_batch(angv * dt), curr_rot)
        new_rot_norm = new_rot/torch.norm(new_rot, dim = 1).view(-1, 1)

        # new_rot = self.add_base_rot_batch(new_rot)
        next_qpos[:, 3:7] = new_rot_norm
        self.sim['qpos'] = next_qpos
        self.sim['qvel'] = get_qvel_fd_batch(curr_qpos, next_qpos, dt, transform=None)
        return self.sim['qpos'], self.sim['qvel']
        
    
    def get_action(self, state): 
        # the state contains currrent qpos in local frame, difference between current step's head prediction and gt, next step's head vel. 
        if self.model_v == 1 or self.model_v == 0:
            rnn_out = self.action_rnn(state)
            x = torch.cat((state, rnn_out), dim=1) # 2 self.qvel_lm + 142 = 398
            x = self.action_mlp(x)
            action = self.action_fc(x)
        elif self.model_v == 2:
            x = self.action_mlp(state)
            action = self.action_fc(x)

        return action           
    

    def forward(self, data):
        # pose: 69 dim body pose
        batch_size, seq_len, _ = data['qpos'].shape # 

        seq_len = data['head_pose'].shape[1] # Tmp usefor input slam res

        data = self.init_states(data)
        
        # Previous version, make sometimes the pinitial prediction is actually not getting trained? 
        if self.gt_rate > 0.0 and np.random.binomial(1, self.gt_rate):  # Scheduled Sampling for intialization state
            self.set_sim(data['qpos'][:,0,:], data['qvel'][:,0,:])

        # use_gt_first_pose = False # Only for testing, check the effects of first pose. 
        # if use_gt_first_pose:
        #     self.set_sim(data['qpos'][:,0,:], data['qvel'][:,0,:]) # Since original training, there's a bug, that the first frame is never trained! 
        #     # init_qpos = data['qpos'][:,0,:]
        #     # init_qvel = data['qvel'][:,0,:]

        feature_pred = defaultdict(list)
        
        state, feature = self.get_obs(data, 0) # state is always in local body space, not world space. 
        for key in feature.keys():
            feature_pred[key].append(feature[key]) # notice: if use gt qvel, then current qvel here contains all the information for next step. 
                
        # if self.gt_rate > 0.0 and np.random.binomial(1, self.gt_rate):  # Scheduled Sampling for intialization state
        #     self.set_sim(data['qpos'][:,0,:], data['qvel'][:,0,:])
        #     state, feature = self.get_obs(data, 0) # state is always in local body space, not world space. 

        # time_list = []
        # the state contains currrent qpos in local frame, difference between current step's head prediction and gt, next step's head vel. 
        for t in range(1, seq_len):
            start_time = time.time()

            action = self.get_action(state)
            # action = data['target'][:,t-1,:] # Debugging GT
            # action is: root_z(1), root_quat(4)(seems not used?), euler angle(69), root linear v(3), root angular v(3) = 80 dim. 
            self.step(action) # To get next step's qpos, qvel, assign to self.sim

            # scheduled sampling previous version 
            if self.gt_rate > 0.0 and np.random.binomial(1, self.gt_rate):
                self.set_sim(data['qpos'][:,t,:], data['qvel'][:,t,:])
            state, feature = self.get_obs(data, t)

            end_time = time.time()
            interval = end_time - start_time 
            # print("Current t takes time:{0} seconds".format(interval))
            # time_list.append(interval)

            # for key in feature.keys():
            #     feature_pred[key].append(feature[key])
            
            # feature_pred['action'].append(action)

            # Modified version, because that previous version looks will lose some training chances when the current step using gt for next step prediction.
            # state, feature = self.get_obs(data, t)
            
            for key in feature.keys():
                feature_pred[key].append(feature[key])
            
            feature_pred['action'].append(action)

            # Then scheduled sampling for deciding whether using gt as input to next step 
            # if self.gt_rate > 0.0 and np.random.binomial(1, self.gt_rate):
            #     self.set_sim(data['qpos'][:,t,:], data['qvel'][:,t,:])
            #     state, feature = self.get_obs(data, t)

        # avg_time = np.asarray(time_list).mean()
        # print("The avg time is :{0} seconds".format(avg_time)) 

        action = self.get_action(state)
        # action = data['target'][:,t,:] # Debugging GT
        feature_pred['action'].append(action)
        
        for key in feature_pred.keys():
            feature_pred[key] = torch.stack(feature_pred[key], dim=1)
            
        self.fix_qvel(feature_pred)

        return feature_pred
    
    def fix_qvel(self, feature_pred):
        pred_qvel = feature_pred['qvel']
        feature_pred['qvel'] = torch.cat((pred_qvel[:,1:,:], pred_qvel[:,-2:-1,:]), dim = 1) # ? 
        
    def compute_loss(self, feature_pred, data):
        w_rp, w_rr, w_p, w_v, w_ee, w_a, w_op, w_or = self.specs.get("w_rp", 50), self.specs.get("w_rr", 50), self.specs.get("w_p", 1), self.specs.get("w_v", 1), self.specs.get("w_ee", 1), self.specs.get("w_a", 1), self.specs.get("w_op", 1), self.specs.get("w_or", 1)

        b_size, seq_len, nq = feature_pred['qpos'].shape
        pred_qpos = feature_pred['qpos'].reshape(b_size * seq_len, -1)
        gt_qpos = data['qpos'].reshape(b_size * seq_len, -1)
        gt_obj_2_head = data['obj_head_relative_poses'].reshape(b_size * seq_len, -1)
        pred_obj_2_head = feature_pred['obj_2_head'].reshape(b_size * seq_len, -1)
        
        pred_qvel = feature_pred['qvel'][:,:-1,:].reshape(b_size * (seq_len - 1), -1)
        gt_qvel = data['qvel'][:,1:,:].reshape(b_size * (seq_len - 1), -1) # ZL: GT qvel is one step ahead 

        action = feature_pred['action'].reshape(b_size * seq_len, -1)

        pred_wbpos = feature_pred['pred_wbpos'].reshape(b_size * seq_len, -1)
        gt_w_pos = data['wbpos'].reshape(b_size *  seq_len, -1)
        # pred_wbquat = feature_pred['pred_wbquat'].view(b_size, seq_len, -1)
        # wbquat = data['wbquat'].view(b_size, seq_len, -1)
        target_action = data['target'].reshape(b_size * seq_len, -1)
        
        r_pos_loss = root_pos_loss(gt_qpos, pred_qpos).mean()
        r_rot_loss = root_orientation_loss(gt_qpos, pred_qpos).mean() 
        p_rot_loss = pose_rot_loss(gt_qpos, pred_qpos).mean()    # pose loss
        vl_loss = linear_velocity_loss(gt_qvel, pred_qvel).mean() # Root angular velocity loss
        va_loss = angular_velocity_loss(gt_qvel, pred_qvel).mean() # Root angular velocity loss
        ee_loss = end_effector_pos_loss(gt_w_pos, pred_wbpos).mean() # End effector loss
        # a_loss = action_loss(target_action, action).mean()

        o_pos_loss = position_loss(gt_obj_2_head[:, :3], pred_obj_2_head[:, :3]).mean() ## Object to head loss
        o_rot_loss = orientation_loss(gt_obj_2_head[:, 3:], pred_obj_2_head[:, 3:]).mean() ## Object to head loss

        # loss = w_rp * r_pos_loss + w_rr * r_rot_loss + w_p * p_rot_loss + w_v * vl_loss + w_v * va_loss + w_ee * ee_loss + w_a * a_loss
        loss = w_rp * r_pos_loss + w_rr * r_rot_loss + w_p * p_rot_loss + w_v * vl_loss + w_v * va_loss + w_ee * ee_loss + w_op * o_pos_loss + w_or * o_rot_loss
 

        return loss, [i.item() for i in [r_pos_loss, r_rot_loss,  p_rot_loss, vl_loss, va_loss, ee_loss, o_pos_loss, o_rot_loss]]
        
    def compute_loss_lite(self, pred_qpos, gt_qpos):
        w_rp, w_rr, w_p, w_v, w_ee, w_a, w_op, w_or = self.specs.get("w_rp", 50), self.specs.get("w_rr", 50), self.specs.get("w_p", 1), self.specs.get("w_v", 1), self.specs.get("w_ee", 1), self.specs.get("w_a", 1), self.specs.get("w_op", 1), self.specs.get("w_or", 1)
        fk_res_pred = self.fk_model.qpos_fk(pred_qpos)
        fk_res_gt = self.fk_model.qpos_fk(gt_qpos)

        pred_wbpos = fk_res_pred['wbpos'].reshape(pred_qpos.shape[0], -1)
        gt_wbpos = fk_res_gt['wbpos'].reshape(pred_qpos.shape[0], -1)
        
        
        r_pos_loss = root_pos_loss(gt_qpos, pred_qpos).mean()
        r_rot_loss = root_orientation_loss(gt_qpos, pred_qpos).mean() 
        p_rot_loss = pose_rot_loss(gt_qpos, pred_qpos).mean()    # pose loss
        ee_loss = end_effector_pos_loss(gt_wbpos, pred_wbpos).mean() # End effector loss

        loss = w_rp * r_pos_loss + w_rr * r_rot_loss + w_p * p_rot_loss +  w_ee * ee_loss
 

        return loss, [i.item() for i in [r_pos_loss, r_rot_loss,  p_rot_loss, ee_loss]]

    def compute_loss_init(self, pred_qpos, gt_qpos, pred_qvel, gt_qvel):
        w_rp, w_rr, w_p, w_v, w_ee, w_a, w_op, w_or = self.specs.get("w_rp", 50), self.specs.get("w_rr", 50), self.specs.get("w_p", 1), self.specs.get("w_v", 1), self.specs.get("w_ee", 1), self.specs.get("w_a", 1), self.specs.get("w_op", 1), self.specs.get("w_or", 1)
        fk_res_pred = self.fk_model.qpos_fk(pred_qpos)
        fk_res_gt = self.fk_model.qpos_fk(gt_qpos)

        pred_wbpos = fk_res_pred['wbpos'].reshape(pred_qpos.shape[0], -1)
        gt_wbpos = fk_res_gt['wbpos'].reshape(pred_qpos.shape[0], -1)
        
        
        r_pos_loss = root_pos_loss(gt_qpos, pred_qpos).mean()
        r_rot_loss = root_orientation_loss(gt_qpos, pred_qpos).mean() 
        p_rot_loss = pose_rot_loss(gt_qpos, pred_qpos).mean()    # pose loss
        ee_loss = end_effector_pos_loss(gt_wbpos, pred_wbpos).mean() # End effector loss

        loss = w_rp * r_pos_loss + w_rr * r_rot_loss + w_p * p_rot_loss +  w_ee * ee_loss
 

        return loss, [i.item() for i in [r_pos_loss, r_rot_loss,  p_rot_loss, ee_loss]]
        
'''
Each step, input current height z, removed heading root orientation quaternion, local joint rotation, 
removed heading head vels, head linear difference in removed heading frame, head rotation difference.

Output: next step's height z, local joint rotation, root linear velocity in local frame, root angular velocity in local frame.  
'''
