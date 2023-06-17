import os
import sys
sys.path.append(os.getcwd())

from copycat.khrylib.rl.envs.common import mujoco_env
from copycat.khrylib.utils import *
from copycat.khrylib.utils.transformation import quaternion_from_euler
from copycat.khrylib.rl.core.policy_gaussian import PolicyGaussian
from copycat.khrylib.rl.core.critic import Value
from copycat.khrylib.models.mlp import MLP
from copycat.core.policy_mcp import PolicyMCP
from relive.utils.numpy_smpl_humanoid import Humanoid

from gym import spaces
from mujoco_py import functions as mjf
import pickle
import time
from scipy.linalg import cho_solve, cho_factor
import joblib
from relive.utils.flags import flags


class HumanoidAREnv(mujoco_env.MujocoEnv):
    # Wrapper class that wraps around Copycat agent

    def __init__(self, cfg, cc_cfg, init_context, cc_iter = -1, mode = "train", wild = False, ar_mode = False):
        mujoco_env.MujocoEnv.__init__(self, cc_cfg.mujoco_model_file, 15)
        self.cc_cfg = cc_cfg
        self.cfg = cfg
        self.wild = wild

        self.set_cam_first = set()
        # env specific
        self.base_rot = cc_cfg.data_specs.get("base_rot", [0.7071, 0.7071, 0.0, 0.0])
        self.qpos_lim = 76
        self.qvel_lim = 75
        self.body_lim = 25
        self.num_obj = self.get_obj_qpos().shape[0]//7
        self.end_reward = 0.0
        self.start_ind = 0
        self.action_index_map = [0, 7, 21, 28]
        self.action_len = [7, 14, 7, 7]
        self.action_names = ["sit", "push", "avoid", "step"]
        self.smpl_humanoid = Humanoid(model_file=cfg.mujoco_model_file)
        
        self.body_qposaddr = get_body_qposaddr(self.model)
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.prev_hpos = None
        self.set_model_params()
        self.load_context(init_context)
        self.policy_v = cfg.policy_specs['policy_v']
        self.scheduled_smpling = cfg.policy_specs.get("scheduled_smpling", 0)
        self.pose_delta = self.cfg.model_specs.get("pose_delta", False)
        self.ar_model_v = self.cfg.model_specs.get("model_v", 1)
        self.ar_mode = ar_mode

        print(f"scheduled_sampling: {self.scheduled_smpling}")

        self.mode = mode
        
        self.set_spaces()
        self.jpos_diffw = np.array(cfg.reward_weights.get("jpos_diffw", [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))[:, None]


        ''' Load CC Controller '''
        state_dim = self.get_cc_obs().shape[0]
        action_dim = self.cc_action_dim
        if cc_cfg.actor_type == "gauss":
            self.cc_policy = PolicyGaussian(cc_cfg, action_dim = action_dim, state_dim = state_dim)
        elif cc_cfg.actor_type == "mcp":
            self.cc_policy = PolicyMCP(cc_cfg, action_dim = action_dim, state_dim = state_dim)

        self.cc_value_net = Value(MLP(state_dim, cc_cfg.value_hsize, cc_cfg.value_htype))
        print(cc_cfg.model_dir)
        if cc_iter != -1:
            cp_path = '%s/iter_%04d.p' % (cc_cfg.model_dir, cc_iter)
        else:
            cc_iter = np.max([int(i.split("_")[-1].split(".")[0]) for i in os.listdir(cc_cfg.model_dir)])
            cp_path = '%s/iter_%04d.p' % (cc_cfg.model_dir, cc_iter)
        print(('loading model from checkpoint: %s' % cp_path))
        model_cp = pickle.load(open(cp_path, "rb"))
        self.cc_running_state = model_cp['running_state']
        self.cc_policy.load_state_dict(model_cp['policy_dict'])
        self.cc_value_net.load_state_dict(model_cp['value_dict'])


    def load_context(self, data_dict):
        self.ar_context = {k:v[0].detach().cpu().numpy() if v.requires_grad else v[0].cpu().numpy() for k, v in data_dict.items()}

        self.ar_context['len'] = self.ar_context['qpos'].shape[0] - 1
        self.gt_targets = self.smpl_humanoid.qpos_fk_batch(self.ar_context['qpos'])
        self.target = self.smpl_humanoid.qpos_fk(self.ar_context['ar_qpos'][0])
        
            
    def set_model_params(self):
        if self.cc_cfg.action_type == 'torque' and hasattr(self.cc_cfg, 'j_stiff'):
            self.model.jnt_stiffness[1:] = self.cc_cfg.j_stiff
            self.model.dof_damping[6:] = self.cc_cfg.j_damp

    def set_spaces(self):
        cfg = self.cc_cfg
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        self.vf_dim = 0
        if cfg.residual_force:
            if cfg.residual_force_mode == 'implicit':
                self.vf_dim = 6
            else:
                if cfg.residual_force_bodies == 'all':
                    self.vf_bodies = SMPL_BONE_NAMES
                else:
                    self.vf_bodies = cfg.residual_force_bodies
                self.body_vf_dim = 6 + cfg.residual_force_torque * 3
                self.vf_dim = self.body_vf_dim * len(self.vf_bodies)
        self.cc_action_dim = self.ndof + self.vf_dim

        self.action_dim = 75
        self.action_space = spaces.Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim), dtype=np.float32)
        obs = self.get_obs()

        self.obs_dim = obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def get_obs(self):
        ar_obs = self.get_ar_obs_v1()
        return ar_obs

    def get_cc_obs(self):
        if self.cc_cfg.obs_v == 0:
            cc_obs = self.get_full_obs()
        elif self.cc_cfg.obs_v == 1:
            cc_obs = self.get_full_obs_v1()
        return cc_obs


    def get_full_obs(self):
        data = self.data
        qpos = data.qpos[:self.qpos_lim].copy()
        qvel = data.qvel[:self.qvel_lim].copy()
        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cc_cfg.obs_coord).ravel()
        obs = []
        # pos
        if self.cc_cfg.obs_heading:
            obs.append(np.array([get_heading(qpos[3:7])]))
        if self.cc_cfg.root_deheading:
            qpos[3:7] = de_heading(qpos[3:7])
        obs.append(qpos[2:])
        # vel
        if self.cc_cfg.obs_vel == 'root':
            obs.append(qvel[:6])
        elif self.cc_cfg.obs_vel == 'full':
            obs.append(qvel)

        obs.append(self.get_target_kin_pose())

        obs = np.concatenate(obs)
        return obs

    def remove_base_rot(self, quat):
        return quaternion_multiply(quat, quaternion_inverse(self.base_rot))

    def get_full_obs_v1(self):
        data = self.data
        qpos = data.qpos[:self.qpos_lim].copy()
        qvel = data.qvel[:self.qvel_lim].copy()
        
        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cc_cfg.obs_coord).ravel() # body angular velocity 
        obs = []

        curr_root_quat = self.remove_base_rot(qpos[3:7])
        hq = get_heading_q(curr_root_quat)
        obs.append(hq) # obs: heading (4,)

        ################ Body pose and z ################
        target_body_qpos = self.get_target_qpos() # target body pose (1, 76)
        target_root_quat = self.remove_base_rot(target_body_qpos[3:7])

        qpos[3:7] = de_heading(curr_root_quat) # deheading the root 
        diff_qpos  = target_body_qpos.copy()
        diff_qpos[2] -= qpos[2]
        diff_qpos[7:] -= qpos[7:]
        diff_qpos[3:7] = quaternion_multiply(target_root_quat, quaternion_inverse(curr_root_quat))

        obs.append(target_body_qpos[2:]) # obs: target z + body pose (1, 70)
        obs.append(qpos[2:]) # obs: target z +  body pose (1, 70)
        obs.append(diff_qpos[2:]) # obs:  difference z + body pose (1, 70)
        
        ################ vels ################
        # vel
        qvel[:3] = transform_vec(qvel[:3], curr_root_quat, self.cc_cfg.obs_coord).ravel()
        if self.cc_cfg.obs_vel == 'root':
            obs.append(qvel[:6])
        elif self.cc_cfg.obs_vel == 'full':
            obs.append(qvel)

        ################ relative heading and root position ################
        rel_h = get_heading(target_root_quat) - get_heading(curr_root_quat)
        if rel_h > np.pi:
            rel_h -= 2 * np.pi
        if rel_h < -np.pi:
            rel_h += 2 * np.pi
        obs.append(np.array([rel_h])) # obs: heading difference in angles (1, 1)

        rel_pos = target_root_quat[:3] - qpos[:3]
        rel_pos = transform_vec(rel_pos, curr_root_quat, self.cc_cfg.obs_coord).ravel()
        obs.append(rel_pos[:2]) # obs: relative x, y difference (1, 2)
        
        ################ target/difference joint/com positions ################
        target_jpos = self.get_target_joint_pos()
        curr_jpos = self.data.body_xpos[1:self.body_lim].copy()
        r_jpos = curr_jpos - qpos[None, :3] 
        r_jpos = transform_vec_batch(r_jpos, curr_root_quat, self.cc_cfg.obs_coord) # body frame position 
        obs.append(r_jpos.ravel()) # obs: target body frame joint position (1, 72)

        diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
        diff_jpos = transform_vec_batch(diff_jpos, curr_root_quat, self.cc_cfg.obs_coord)
        obs.append(diff_jpos.ravel()) # obs: current diff body frame joint position  (1, 72)

        target_com = self.get_target_com_pos() # body frame position
        curr_com = self.data.xipos[1:self.body_lim].copy()
        
        r_com = curr_com - qpos[None, :3]
        r_com = transform_vec_batch(r_com, curr_root_quat, self.cc_cfg.obs_coord)
        obs.append(r_com.ravel()) # obs: current target body frame com position  (1, 72)
        diff_com = target_com.reshape(-1, 3) - curr_com
        diff_com = transform_vec_batch(diff_com, curr_root_quat, self.cc_cfg.obs_coord)
        obs.append(diff_com.ravel()) # obs: current body frame com position (1, 72)
        

        ################ target/relative global joint quaternions ################
        # target_quat = self.get_expert_bquat().reshape(-1, 4)
        target_quat = self.get_target_wbquat().reshape(-1, 4)
        
        cur_quat = self.data.body_xquat.copy()[1:self.body_lim]

        if cur_quat[0, 0] == 0:
            cur_quat = target_quat.copy()

        r_quat = cur_quat.copy()
        for i in range(r_quat.shape[0]):
            r_quat[i] = quaternion_multiply(quaternion_inverse(hq), r_quat[i])
        obs.append(r_quat.ravel()) # obs: current target body quaternion (1, 92)

        rel_quat = np.zeros_like(cur_quat)
        for i in range(rel_quat.shape[0]):
            rel_quat[i] = quaternion_multiply(quaternion_inverse(cur_quat[i]), target_quat[i])
        obs.append(rel_quat.ravel()) # obs: current target body quaternion (1, 92)

        obs = np.concatenate(obs)
        return obs

    def get_head_idx(self):
        return self.model._body_name2id["Head"]  - 1

    def get_ar_obs_v1(self):
        t = self.cur_t 
        curr_action = self.ar_context["action_one_hot"][0]
        obs = []
        curr_qpos = self.data.qpos[:self.qpos_lim].copy()
        curr_qvel = self.data.qvel[:self.qvel_lim].copy()

        curr_qpos_local = curr_qpos.copy()
        curr_qpos_local[3:7] = de_heading(curr_qpos_local[3:7]) 

        pred_wbpos, pred_wbquat = self.get_wbody_pos().reshape(-1, 3), self.get_wbody_quat().reshape(-1, 4)

        head_idx = self.get_head_idx()
        pred_hrot = pred_wbquat[head_idx]
        pred_hpos = pred_wbpos[head_idx]

        pred_hrot_heading = pred_hrot


        if self.cfg.use_context or self.cfg.use_of: 
            if "context_feat_rnn" in self.ar_context:
                obs.append(self.ar_context['context_feat_rnn'][t, :])
            else:
                obs.append(np.zeros((256)))

        if self.cfg.use_head:
            # get target head
            t_hrot = self.ar_context['head_pose'][t, 3:].copy()
            t_hpos = self.ar_context['head_pose'][t, :3].copy()
            # get target head vel
            t_havel = self.ar_context['head_vels'][t, 3:].copy()
            t_hlvel = self.ar_context['head_vels'][t, :3].copy()
            t_obj_relative_head = self.ar_context["obj_head_relative_poses"][t,:].copy()

            # difference in head, in head's heading coordinates
            diff_hpos = t_hpos - pred_hpos 
            diff_hpos = transform_vec(diff_hpos, pred_hrot_heading, "heading")
            diff_hrot = quaternion_multiply(quaternion_inverse(t_hrot), pred_hrot)
        
        
        q_heading = get_heading_q(pred_hrot_heading).copy()
        obj_pos = self.get_obj_qpos(action_one_hot=curr_action)[:3] 
        obj_rot = self.get_obj_qpos(action_one_hot=curr_action)[3:7]

        diff_obj = obj_pos - pred_hpos
        diff_obj_loc = transform_vec(diff_obj, pred_hrot_heading, "heading")
        obj_rot_local = quaternion_multiply(quaternion_inverse(q_heading), obj_rot) # Object local coordinate
        pred_obj_relative_head = np.concatenate([diff_obj_loc, obj_rot_local], axis = 0)

        # state 
        # order of these matters !!!
        obs.append(curr_qpos_local[2:]) # current height + local body orientation + body pose self.qpose_lm  74
        if self.cfg.use_vel:
            obs.append(curr_qvel) # current velocities 75

        if self.cfg.use_head:
            obs.append(diff_hpos) # diff head position 3
            obs.append(diff_hrot) # diff head rotation 4

        obs.append(pred_obj_relative_head) # predicted object relative to head 7

        if self.cfg.use_head:
            obs.append(t_havel)   # target head angular velocity 3
            obs.append(t_hlvel)    # target head linear  velocity 3
            obs.append(t_obj_relative_head)  # target object relative to head 7 

        if self.cfg.use_action and self.ar_model_v > 0:
            obs.append(curr_action)

        if self.cfg.use_of:
            # Not sure what to do yet......
            obs.append(self.ar_context['of'][t, :])
        
        
        # obs.append(curr_qpos)
        # obs.append(self.get_obj_qpos())
        if self.policy_v == 2:
            obs.append(self.ar_context['ar_qpos'][self.cur_t])

        obs = np.concatenate(obs)
        
        return obs

    def get_full_obs_v1_notrans(self):
        # no root translation model
        pass

    def get_ee_pos(self, transform):
        data = self.data
        ee_name = ['L_Toe', 'R_Toe', 'L_Wrist', 'R_Wrist', 'Head']
        ee_pos = []
        root_pos = data.qpos[:3]
        root_q = data.qpos[3:7].copy()
        for name in ee_name:
            bone_id = self.model._body_name2id[name]
            bone_vec = self.data.body_xpos[bone_id]
            if transform is not None:
                bone_vec = bone_vec - root_pos
                bone_vec = transform_vec(bone_vec, root_q, transform)
            ee_pos.append(bone_vec)
        return np.concatenate(ee_pos)

    def get_body_quat(self):
        qpos = self.get_humanoid_qpos()
        body_quat = [qpos[3:7]]
        for body in self.model.body_names[1:self.body_lim]:
            if body == 'Pelvis' or not body in self.body_qposaddr:
                continue
            start, end = self.body_qposaddr[body]
            euler = np.zeros(3)
            euler[:end - start] = qpos[start:end]
            quat = quaternion_from_euler(euler[0], euler[1], euler[2])
            body_quat.append(quat)
        body_quat = np.concatenate(body_quat)
        return body_quat

    def get_wbody_quat(self, selectList=None):
        body_pos = []
        if selectList is None:
            # body_names = self.model.body_names[1:] # ignore plane
            return self.data.body_xquat[1:self.body_lim].copy().ravel()
        else:
            body_names = selectList
        for body in body_names: 
            bone_idx = self.model._body_name2id[body]
            bone_vec = self.data.body_xquat[bone_idx]
            body_pos.append(bone_vec)
        return np.concatenate(body_pos)

    def get_com(self):
        return self.data.subtree_com[0, :].copy()

    def get_head(self):
        bone_id = self.model._body_name2id['Head']
        head_pos = self.data.body_xpos[bone_id]
        head_quat = self.data.body_xquat[bone_id]
        return np.concatenate((head_pos, head_quat))

    def get_wbody_pos(self, selectList=None):
        body_pos = []
        if selectList is None:
            # body_names = self.model.body_names[1:] # ignore plane
            return self.data.body_xpos[1:self.body_lim].copy().ravel()
        else:
            body_names = selectList
        for body in body_names: 
            bone_idx = self.model._body_name2id[body]
            bone_vec = self.data.body_xpos[bone_idx]
            body_pos.append(bone_vec)
        return np.concatenate(body_pos)


    def get_full_body_com(self, selectList=None):
        body_pos = []
        if selectList is None:
            body_names = self.model.body_names[1:self.body_lim] # ignore plane
        else:
            body_names = selectList
        
        for body in body_names: 
            bone_vec = self.data.get_body_xipos(body)
            body_pos.append(bone_vec)
        
        return np.concatenate(body_pos)

    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):
        dt = self.model.opt.timestep
        nv = self.model.nv
        
        M = np.zeros(nv * nv)
        mjf.mj_fullM(self.model, M, self.data.qM)
        M.resize(nv, nv)
        M = M[:self.qvel_lim, :self.qvel_lim]
        C = self.data.qfrc_bias.copy()[:self.qvel_lim]
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(cho_factor(M + K_d*dt, overwrite_a=True, check_finite=False),
                            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]), overwrite_b=True, check_finite=False)
        return q_accel.squeeze()

    def compute_torque(self, ctrl):
        cfg = self.cc_cfg
        dt = self.model.opt.timestep

        ctrl_joint = ctrl[:self.ndof] * cfg.a_scale
        qpos = self.get_humanoid_qpos()
        qvel = self.get_humanoid_qvel()
        if self.cc_cfg.action_v == 1 or self.cc_cfg.action_v == 2 or self.cc_cfg.action_v == 3:
            base_pos = self.get_target_kin_pose()
            while np.any(base_pos - qpos[7:] > np.pi):
                base_pos[base_pos - qpos[7:] > np.pi] -= 2 * np.pi
            while np.any(base_pos - qpos[7:] < -np.pi):
                base_pos[base_pos - qpos[7:] < -np.pi] += 2 * np.pi

        elif self.cc_cfg.action_v == 0:
            base_pos = cfg.a_ref
        target_pos = base_pos + ctrl_joint

        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])

        k_p[6:] = cfg.jkp
        k_d[6:] = cfg.jkd
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:]*dt - target_pos))
        qvel_err = qvel
        q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d)
        qvel_err += q_accel * dt
        torque = -cfg.jkp * qpos_err[6:] - cfg.jkd * qvel_err[6:]

        return torque

    """ RFC-Explicit """
    def rfc_explicit(self, vf):
        qfrc = np.zeros_like(self.data.qfrc_applied)
        for i, body in enumerate(self.vf_bodies):
            body_id = self.model._body_name2id[body]
            contact_point = vf[i*self.body_vf_dim: i*self.body_vf_dim + 3]
            force = vf[i*self.body_vf_dim + 3: i*self.body_vf_dim + 6] * self.cc_cfg.residual_force_scale
            torque = vf[i*self.body_vf_dim + 6: i*self.body_vf_dim + 9] * self.cc_cfg.residual_force_scale if self.cc_cfg.residual_force_torque else np.zeros(3)
            contact_point = self.pos_body2world(body, contact_point)
            force = self.vec_body2world(body, force)
            torque = self.vec_body2world(body, torque)
            mjf.mj_applyFT(self.model, self.data, force, torque, contact_point, body_id, qfrc)
        self.data.qfrc_applied[:] = qfrc

    """ RFC-Implicit """
    def rfc_implicit(self, vf):
        vf *= self.cc_cfg.residual_force_scale
        curr_root_quat = self.remove_base_rot(self.get_humanoid_qpos()[3:7])
        hq = get_heading_q(curr_root_quat)
        # hq = get_heading_q(self.get_humanoid_qpos()[3:7])
        vf[:3] = quat_mul_vec(hq, vf[:3])
        vf = np.clip(vf, -self.cc_cfg.residual_force_lim, self.cc_cfg.residual_force_lim)
        self.data.qfrc_applied[:vf.shape[0]] = vf

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cc_cfg
        for i in range(n_frames):
            ctrl = action
            if cfg.action_type == 'position':
                torque = self.compute_torque(ctrl)
            elif cfg.action_type == 'torque':
                torque = ctrl * cfg.a_scale
            
            torque = np.clip(torque, -cfg.torque_lim, cfg.torque_lim)
            self.data.ctrl[:] = torque

            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[-self.vf_dim:].copy()
                if cfg.residual_force_mode == 'implicit':
                    self.rfc_implicit(vf)
                else:
                    self.rfc_explicit(vf)

            try:
                self.sim.step()
            except Exception as e:
                print(e, action)
            

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0

    def step_ar(self, a, dt = 1/30):
        qpos_lm = 74
        pose_start = 7
        curr_qpos = self.data.qpos[:self.qpos_lim].copy()
        curr_qvel = self.data.qvel[:self.qvel_lim].copy()

        curr_pos, curr_rot = curr_qpos[:3], curr_qpos[3:7]
        curr_heading = get_heading_q(curr_rot)

        body_pose = a[(pose_start - 2):qpos_lm]
        

        if self.pose_delta:
            body_pose += curr_qpos[pose_start:]
            body_pose[body_pose>np.pi] -= 2 * np.pi
            body_pose[body_pose<-np.pi] += 2 * np.pi

        next_qpos = np.concatenate([curr_pos[:2], a[:(pose_start - 2)], body_pose], axis=0)
        root_qvel = a[qpos_lm:]
        linv = quat_mul_vec(curr_heading, root_qvel[:3]) 
        next_qpos[ :2] += linv[:2] * dt

        angv = quat_mul_vec(curr_rot, root_qvel[3:6])
        new_rot = quaternion_multiply(quat_from_expmap(angv * dt), curr_rot)

        next_qpos[3:7] = new_rot
        return next_qpos



    def step(self, a):
        cfg = self.cc_cfg
        # record prev state
        self.prev_qpos = self.get_humanoid_qpos()
        self.prev_qvel = self.get_humanoid_qvel()
        self.prev_bquat = self.bquat.copy()
        self.prev_hpos = self.get_head().copy()
        
        if self.policy_v == 1:
            next_qpos = self.step_ar(a.copy())
        elif self.policy_v == 2:
            next_qpos = a.copy()

        self.target = self.smpl_humanoid.qpos_fk(next_qpos) # forming target from arnet
        
        # if flags.debug:
            # next_qpos = self.step_ar(self.ar_context['target'][self.cur_t]) # 
            # self.target = self.smpl_humanoid.qpos_fk(self.ar_context['qpos'][self.cur_t + 1]) # GT
            # self.target = self.smpl_humanoid.qpos_fk(self.ar_context['ar_qpos'][self.cur_t + 1]) # Debug
        if self.ar_mode:
            self.target = self.smpl_humanoid.qpos_fk(self.ar_context['ar_qpos'][self.cur_t + 1]) # 
        
        cc_obs = self.get_cc_obs()
        cc_obs = self.cc_running_state(cc_obs, update=False)
        cc_a = self.cc_policy.select_action(torch.from_numpy(cc_obs)[None, ], mean_action=True)[0].numpy() # CC step
        
        if flags.debug:
            # self.do_simulation(cc_a, self.frame_skip)
            # self.data.qpos[:self.qpos_lim] = self.get_target_qpos() # debug
            # self.data.qpos[:self.qpos_lim] = self.ar_context['qpos'][self.cur_t + 1] # debug
            # self.data.qpos[:self.qpos_lim] = self.gt_targets['qpos'][self.cur_t + 1] # debug
            self.data.qpos[:self.qpos_lim] = self.ar_context['ar_qpos'][self.cur_t + 1] # ARNet Qpos
            self.sim.forward() # debug

        else:
            # if self.mode == "train" and self.scheduled_smpling != 0 and np.random.binomial(1, self.scheduled_smpling):
            #     self.data.qpos[:self.qpos_lim] = self.ar_context['qpos'][self.cur_t + 1]
            #     self.data.qvel[:self.qvel_lim] = self.ar_context['qvel'][self.cur_t + 1]

            #     self.sim.forward() # debug
            # else:
                # self.do_simulation(cc_a, self.frame_skip)
            self.do_simulation(cc_a, self.frame_skip)


        self.cur_t += 1

        self.bquat = self.get_body_quat()
        # get obs
        head_pos = self.get_body_com('Head')
        reward = 1.0
        
        if cfg.env_term_body == 'body':
            # body_diff = self.calc_body_diff()
            # fail = body_diff > 8

            if self.wild:
                body_diff = self.calc_body_diff()
                fail = body_diff > 8 
            else:
                body_diff = self.calc_body_diff()
                if self.mode == "train":
                    body_gt_diff = self.calc_body_gt_diff()
                    fail = body_diff > 10 or body_gt_diff > 12
                else:
                    fail = body_diff > 10 

            
            # if flags.debug:
                # fail =  False
        else:
            raise NotImplemented()
            
        
        end =  (self.cur_t >= cfg.env_episode_len) or (self.cur_t + self.start_ind >= self.ar_context['len'])
        done = fail or end

        # if done: # ZL: Debug
        #     exit()
            # print("done!!!", self.cur_t, self.ar_context['len'] )

        percent = self.cur_t/self.ar_context['len']
        obs = self.get_obs()
        return obs, reward, done, {'fail': fail, 'end': end, "percent": percent}

    def set_mode(self, mode):
        self.mode = mode

    def ar_fail_safe(self):
        self.data.qpos[:self.qpos_lim] = self.ar_context['ar_qpos'][self.cur_t + 1]
        # self.data.qpos[:self.qpos_lim] = self.get_target_qpos()
        self.data.qvel[:self.qvel_lim] = self.ar_context['ar_qvel'][self.cur_t + 1]
        self.sim.forward() 

    def reset_model(self):
        cfg = self.cc_cfg
        ind = 0
        self.start_ind = 0

        if self.ar_mode:
            init_pose_exp = self.ar_context['ar_qpos'][0].copy()
            init_vel_exp = self.ar_context['ar_qvel'][0].copy() 
        else:
            init_pose_exp = self.ar_context['init_qpos'].copy()
            init_vel_exp = self.ar_context['init_qvel'].copy() 

        
        # init_vel_exp = self.ar_context['qvel'][ind].copy() 

        # init_vel_exp = np.zeros(self.ar_context['init_qvel'].shape)

        # if flags.debug:
        #     init_pose_exp = self.ar_context['qpos'][ind].copy()
        #     init_vel_exp = self.ar_context['qvel'][ind].copy() 
        # init_pose_exp[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.qpos_lim - 7)
        
        # if cfg.reactive_v == 0:
        #     # self.set_state(init_pose, init_vel)
        #     pass
        # elif cfg.reactive_v == 1:
        #     if self.mode == "train" and np.random.binomial(1, 1- cfg.reactive_rate):
        #         # self.set_state(init_pose, init_vel)
        #         pass
        #     elif self.mode == "test":
        #         # self.set_state(init_pose, init_vel)
        #         # netural_qpos = self.netural_data['qpos']
        #         # init_pose_exp = self.match_heading_and_pos(init_pose_exp, netural_qpos)
        #         # init_vel_exp = self.netural_data['qvel']
        #         pass
        #     else:
        #         netural_qpos = self.netural_data['init_qpos']
        #         init_pose_exp = self.match_heading_and_pos(init_pose_exp, netural_qpos)
        #         init_vel_exp = self.netural_data['init_qvel']

        #     # self.set_state(init_pose, init_vel)
        #     self.bquat = self.get_body_quat()
        # else:
        #     init_pose = self.get_humanoid_qpos()
        #     init_pose[2] += 1.0
        #     self.set_state(init_pose, self.data.qvel)
        
        obj_pose = self.convert_obj_qpos(self.ar_context["action_one_hot"][0], self.ar_context['obj_pose'][0])
        init_pose = np.concatenate([init_pose_exp, obj_pose])
        init_vel = np.concatenate([init_vel_exp, np.zeros(self.num_obj * 6)])
        
        self.set_state(init_pose, init_vel)
        self.target = self.smpl_humanoid.qpos_fk(init_pose_exp)

        return self.get_obs()

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.get_humanoid_qpos()[:2]
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 33
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.2
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

    def match_heading_and_pos(self, qpos_1, qpos_2):
        posxy_1 = qpos_1[:2]
        qpos_1_quat = self.remove_base_rot(qpos_1[3:7])
        qpos_2_quat = self.remove_base_rot(qpos_2[3:7])
        heading_1 = get_heading_q(qpos_1_quat)
        qpos_2[3:7] = de_heading(qpos_2[3:7])
        qpos_2[3:7]  = quaternion_multiply(heading_1, qpos_2[3:7])
        qpos_2[:2]  = posxy_1
        return qpos_2



    def get_target_qpos(self):
        expert_qpos = self.target['qpos'].copy()
        return expert_qpos 

    
    def get_target_kin_pose(self):
        return self.get_target_qpos()[7:]

    def get_target_joint_pos(self):
        # world joint position 
        wbpos = self.target['wbpos']
        return wbpos

    def get_target_com_pos(self):
        # body joint position 
        body_com = self.target['body_com']
        return body_com

    def get_target_bquat(self):
        bquat = self.target['bquat']
        return bquat

    def get_target_wbquat(self):
        wbquat = self.target['wbquat']
        return wbquat

    def calc_body_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        e_wbpos = self.get_target_joint_pos().reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def calc_body_ar_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        # e_wbpos = self.get_target_joint_pos().reshape(-1, 3)
        e_wbpos = self.ar_context['ar_wbpos'][self.cur_t + 1].reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def calc_body_gt_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        e_wbpos = self.gt_targets['wbpos'][self.cur_t]
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def get_humanoid_qpos(self):
        return self.data.qpos.copy()[:self.qpos_lim]

    def get_humanoid_qvel(self):
        return self.data.qvel.copy()[:self.qvel_lim]

    def get_obj_qpos(self, action_one_hot = None):
        obj_pose_full = self.data.qpos.copy()[self.qpos_lim:]
        if action_one_hot is None:
            return obj_pose_full
        elif np.sum(action_one_hot) == 0:
            return np.array([0,0,0,1,0,0,0])

        action_idx = np.nonzero(action_one_hot)[0][0]
        obj_start = self.action_index_map[action_idx]
        obj_end = obj_start + self.action_len[action_idx]
        
        return obj_pose_full[obj_start:obj_end][:7] # ZL: only support handling one obj right now...
        
    def convert_obj_qpos(self, action_one_hot, obj_pose): # T X 4, T X 7 
        if np.sum(action_one_hot) == 0:
            obj_qos = np.zeros(self.get_obj_qpos().shape[0])
            for i in range(self.num_obj):
                obj_qos[(i*7):(i*7+3)] = [(i + 1) * 100, 100, 0]
            return obj_qos
        else:
            action_idx = np.nonzero(action_one_hot)[0][0]
            obj_qos = np.zeros(self.get_obj_qpos().shape[0])
            # setting defult location for objects
            for i in range(self.num_obj):
                obj_qos[(i*7):(i*7+3)] = [(i + 1) * 100, 100, 0]
                
            obj_start = self.action_index_map[action_idx]
            obj_end = obj_start + self.action_len[action_idx]
            obj_qos[obj_start:obj_end] = obj_pose

            return obj_qos

    def get_obj_qvel(self):
        return self.data.qvel.copy()[self.qvel_lim:]

    

if __name__ == "__main__":
    pass
