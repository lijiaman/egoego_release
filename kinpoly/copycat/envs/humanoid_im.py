import os
import sys

sys.path.append(os.getcwd())

from copycat.khrylib.rl.envs.common import mujoco_env
from copycat.khrylib.utils import *
from copycat.khrylib.utils.transformation import quaternion_from_euler
from gym import spaces
from copycat.utils.tools import get_expert
from mujoco_py import functions as mjf
import pickle
import time
from scipy.linalg import cho_solve, cho_factor
import joblib


class HumanoidEnv(mujoco_env.MujocoEnv):
    def __init__(self, cfg, init_expert, data_specs, mode="train", no_root=False):
        mujoco_env.MujocoEnv.__init__(self, cfg.mujoco_model_file, 15)
        self.cfg = cfg
        self.set_cam_first = set()
        # env specific
        self.qpos_lim = 76
        self.qvel_lim = 75
        self.body_lim = 25
        self.end_reward = 0.0
        self.start_ind = 0
        self.body_qposaddr = get_body_qposaddr(self.model)
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.set_model_params()
        self.expert = None
        self.base_rot = data_specs.get("base_rot", [0.7071, 0.7071, 0.0, 0.0])
        self.netural_path = data_specs.get(
            "neutral_path", "/insert_directory_here/standing_neutral.pkl"
        )
        self.netural_data = joblib.load(self.netural_path)
        self.no_root = no_root
        print("********************* Humanoid Env *********************")
        print(f"No root: {no_root}")
        print(f"Obs root: {self.cfg.obs_v}")
        print("********************* Humanoid Env *********************")

        self.mode = mode
        self.load_expert(init_expert)
        self.set_spaces()
        self.jpos_diffw = np.array(
            cfg.reward_weights.get(
                "jpos_diffw",
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
            )
        )[:, None]

    def load_expert(self, expert_data):
        expert_qpos = expert_data["qpos"]
        expert_meta = {"cyclic": False, "seq_name": expert_data["seq_name"]}

        self.expert = get_expert(expert_qpos, expert_meta, self)
        self.expert.update(expert_data)

    def set_model_params(self):
        if self.cfg.action_type == "torque" and hasattr(self.cfg, "j_stiff"):
            self.model.jnt_stiffness[1:] = self.cfg.j_stiff
            self.model.dof_damping[6:] = self.cfg.j_damp

    def set_spaces(self):
        cfg = self.cfg
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        self.vf_dim = 0
        if cfg.residual_force:
            if cfg.residual_force_mode == "implicit":
                self.vf_dim = 6
            else:
                if cfg.residual_force_bodies == "all":
                    self.vf_bodies = SMPL_BONE_NAMES
                else:
                    self.vf_bodies = cfg.residual_force_bodies
                self.body_vf_dim = 6 + cfg.residual_force_torque * 3
                self.vf_dim = self.body_vf_dim * len(self.vf_bodies)

        self.action_dim = self.ndof + self.vf_dim
        self.action_space = spaces.Box(
            low=-np.ones(self.action_dim),
            high=np.ones(self.action_dim),
            dtype=np.float32,
        )
        self.obs_dim = self.get_obs().size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def remove_base_rot(self, quat):
        return quaternion_multiply(quat, quaternion_inverse(self.base_rot))

    def get_obs(self):
        if self.cfg.obs_type == "full":
            if self.cfg.obs_v == 0:
                obs = self.get_full_obs()
            elif self.cfg.obs_v == 1:
                obs = self.get_full_obs_v1()
            elif self.cfg.obs_v == 2:
                obs = self.get_full_obs_v2()
        return obs

    def get_full_obs(self):
        data = self.data
        qpos = data.qpos[: self.qpos_lim].copy()
        qvel = data.qvel[: self.qvel_lim].copy()
        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cfg.obs_coord).ravel()
        obs = []
        # pos
        if self.cfg.obs_heading:
            obs.append(np.array([get_heading(qpos[3:7])]))
        if self.cfg.root_deheading:
            qpos[3:7] = de_heading(qpos[3:7])
        obs.append(qpos[2:])
        # vel
        if self.cfg.obs_vel == "root":
            obs.append(qvel[:6])
        elif self.cfg.obs_vel == "full":
            obs.append(qvel)

        obs.append(self.get_expert_kin_pose())

        # phase
        if self.cfg.obs_phase:
            phase = self.get_phase()
            obs.append(np.array([phase]))

        obs = np.concatenate(obs)
        return obs

    def get_phase(self):
        return self.cur_t / self.expert["len"]

    def get_full_obs_v1(self):
        data = self.data
        qpos = data.qpos[: self.qpos_lim].copy()
        qvel = data.qvel[: self.qvel_lim].copy()

        # transform velocity
        qvel[:3] = transform_vec(
            qvel[:3], qpos[3:7], self.cfg.obs_coord
        ).ravel()  # body angular velocity
        obs = []

        curr_root_quat = self.remove_base_rot(qpos[3:7])
        hq = get_heading_q(curr_root_quat)
        obs.append(hq)  # obs: heading (4,)

        ################ Body pose and z ################
        target_body_qpos = self.get_expert_qpos(delta_t=1)  # target body pose (1, 76)
        target_root_quat = self.remove_base_rot(target_body_qpos[3:7])

        qpos[3:7] = de_heading(curr_root_quat)  # deheading the root
        diff_qpos = target_body_qpos.copy()
        diff_qpos[2] -= qpos[2]
        diff_qpos[7:] -= qpos[7:]
        diff_qpos[3:7] = quaternion_multiply(
            target_root_quat, quaternion_inverse(curr_root_quat)
        )

        obs.append(
            target_body_qpos[2:]
        )  # obs: target z + body pose (1, 74) # ZL: shounldn't you remove base root here???
        obs.append(qpos[2:])  # obs: current z +  body pose (1, 74)
        obs.append(diff_qpos[2:])  # obs:  difference z + body pose (1, 74)

        ################ vels ################
        # vel
        qvel[:3] = transform_vec(qvel[:3], curr_root_quat, self.cfg.obs_coord).ravel()
        if self.cfg.obs_vel == "root":
            obs.append(qvel[:6])
        elif self.cfg.obs_vel == "full":
            obs.append(qvel)

        ################ relative heading and root position ################
        rel_h = get_heading(target_root_quat) - get_heading(curr_root_quat)
        if rel_h > np.pi:
            rel_h -= 2 * np.pi
        if rel_h < -np.pi:
            rel_h += 2 * np.pi
        obs.append(np.array([rel_h]))  # obs: heading difference in angles (1, 1)

        rel_pos = target_root_quat[:3] - qpos[:3]
        rel_pos = transform_vec(rel_pos, curr_root_quat, self.cfg.obs_coord).ravel()
        obs.append(rel_pos[:2])  # obs: relative x, y difference (1, 2)

        ################ target/difference joint/com positions ################
        target_jpos = self.get_expert_joint_pos(delta_t=1)
        curr_jpos = self.data.body_xpos[1 : self.body_lim].copy()
        r_jpos = curr_jpos - qpos[None, :3]
        r_jpos = transform_vec_batch(
            r_jpos, curr_root_quat, self.cfg.obs_coord
        )  # body frame position
        obs.append(r_jpos.ravel())  # obs: target body frame joint position (1, 72)

        diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
        diff_jpos = transform_vec_batch(diff_jpos, curr_root_quat, self.cfg.obs_coord)
        obs.append(
            diff_jpos.ravel()
        )  # obs: current diff body frame joint position  (1, 72)

        target_com = self.get_expert_com_pos(delta_t=1)  # body frame position
        curr_com = self.data.xipos[1 : self.body_lim].copy()

        r_com = curr_com - qpos[None, :3]
        r_com = transform_vec_batch(r_com, curr_root_quat, self.cfg.obs_coord)
        obs.append(
            r_com.ravel()
        )  # obs: current target body frame com position  (1, 72)
        diff_com = target_com.reshape(-1, 3) - curr_com
        diff_com = transform_vec_batch(diff_com, curr_root_quat, self.cfg.obs_coord)
        obs.append(diff_com.ravel())  # obs: current body frame com position (1, 72)

        ################ target/relative global joint quaternions ################
        # target_quat = self.get_expert_bquat(delta_t=1).reshape(-1, 4)
        target_quat = self.get_expert_wbquat(delta_t=1).reshape(-1, 4)

        cur_quat = self.data.body_xquat.copy()[1 : self.body_lim]

        if cur_quat[0, 0] == 0:
            cur_quat = target_quat.copy()

        r_quat = cur_quat.copy()
        for i in range(r_quat.shape[0]):
            r_quat[i] = quaternion_multiply(quaternion_inverse(hq), r_quat[i])
        obs.append(r_quat.ravel())  # obs: current target body quaternion (1, 92)

        rel_quat = np.zeros_like(cur_quat)
        for i in range(rel_quat.shape[0]):
            rel_quat[i] = quaternion_multiply(
                quaternion_inverse(cur_quat[i]), target_quat[i]
            )
        obs.append(rel_quat.ravel())  # obs: current target body quaternion (1, 92)

        obs = np.concatenate(obs)
        return obs

    def fail_safe(self):
        self.data.qpos[: self.qpos_lim] = self.get_expert_qpos()
        self.data.qvel[: self.qvel_lim] = self.get_expert_qvel()
        self.sim.forward()

    def get_full_obs_v2(self):
        data = self.data
        qpos = data.qpos[: self.qpos_lim].copy()
        qvel = data.qvel[: self.qvel_lim].copy()

        # transform velocity
        qvel[:3] = transform_vec(
            qvel[:3], qpos[3:7], self.cfg.obs_coord
        ).ravel()  # body angular velocity
        obs = []

        curr_root_quat = self.remove_base_rot(qpos[3:7])
        hq = get_heading_q(curr_root_quat)
        obs.append(hq)  # obs: heading (4,)

        ################ Body pose and z ################
        target_body_qpos = self.get_expert_qpos(delta_t=1)  # target body pose (1, 76)
        target_root_quat = self.remove_base_rot(target_body_qpos[3:7])

        qpos[3:7] = de_heading(curr_root_quat)  # deheading the root
        diff_qpos = target_body_qpos.copy()
        diff_qpos[2] -= qpos[2]
        diff_qpos[7:] -= qpos[7:]
        diff_qpos[3:7] = quaternion_multiply(
            target_root_quat, quaternion_inverse(curr_root_quat)
        )

        obs.append(target_body_qpos[2:])  # obs: target z + body pose (1, 74)
        obs.append(qpos[2:])  # obs: target z +  body pose (1, 74)
        obs.append(diff_qpos[2:])  # obs:  difference z + body pose (1, 74)

        ################ vels ################
        # vel
        qvel[:3] = transform_vec(qvel[:3], curr_root_quat, self.cfg.obs_coord).ravel()
        if self.cfg.obs_vel == "root":
            obs.append(qvel[:6])
        elif self.cfg.obs_vel == "full":
            obs.append(qvel)  # full qvel, 75

        ################ relative heading and root position ################
        rel_h = get_heading(target_root_quat) - get_heading(curr_root_quat)
        if rel_h > np.pi:
            rel_h -= 2 * np.pi
        if rel_h < -np.pi:
            rel_h += 2 * np.pi
        obs.append(np.array([rel_h]))  # obs: heading difference in angles (1, 1)

        rel_pos = target_root_quat[:3] - qpos[:3]
        rel_pos = transform_vec(rel_pos, curr_root_quat, self.cfg.obs_coord).ravel()
        obs.append(rel_pos[:2])  # obs: relative x, y difference (1, 2)

        ################ target/difference joint positions ################
        target_jpos = self.get_expert_joint_pos(delta_t=1)
        curr_jpos = self.data.body_xpos[1 : self.body_lim].copy()
        r_jpos = curr_jpos - qpos[None, :3]  # translate to body frame (zero-out root)
        r_jpos = transform_vec_batch(
            r_jpos, curr_root_quat, self.cfg.obs_coord
        )  # body frame position
        obs.append(r_jpos.ravel())  # obs: target body frame joint position (1, 72)

        diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
        diff_jpos = transform_vec_batch(diff_jpos, curr_root_quat, self.cfg.obs_coord)
        obs.append(
            diff_jpos.ravel()
        )  # obs: current diff body frame joint position  (1, 72)

        ################ target/relative global joint quaternions ################
        # target_quat = self.get_expert_bquat(delta_t=1).reshape(-1, 4)
        target_quat = self.get_expert_wbquat(delta_t=1).reshape(-1, 4)
        cur_quat = self.data.body_xquat.copy()[1 : self.body_lim]

        if cur_quat[0, 0] == 0:
            cur_quat = target_quat.copy()

        r_quat = cur_quat.copy()
        for i in range(r_quat.shape[0]):
            r_quat[i] = quaternion_multiply(
                quaternion_inverse(hq), r_quat[i]
            )  # ZL: you have gotta batch this.....
        obs.append(
            r_quat.ravel()
        )  # obs: current target body quaternion (1, 96) # this contains redundent information

        rel_quat = np.zeros_like(cur_quat)
        for i in range(rel_quat.shape[0]):
            rel_quat[i] = quaternion_multiply(
                quaternion_inverse(cur_quat[i]), target_quat[i]
            )  # ZL: you have gotta batch this.....
        obs.append(rel_quat.ravel())  # obs: current target body quaternion (1, 96)

        obs = np.concatenate(obs)
        return obs

    def get_full_obs_v1_notrans(self):
        # no root translation model
        pass

    def get_ee_pos(self, transform):
        data = self.data
        ee_name = ["L_Toe", "R_Toe", "L_Wrist", "R_Wrist", "Head"]
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
        for body in self.model.body_names[1 : self.body_lim]:
            if body == "Pelvis" or not body in self.body_qposaddr:
                continue
            start, end = self.body_qposaddr[body]
            euler = np.zeros(3)
            euler[: end - start] = qpos[start:end]
            quat = quaternion_from_euler(euler[0], euler[1], euler[2])
            body_quat.append(quat)
        body_quat = np.concatenate(body_quat)
        return body_quat

    def get_wbody_quat(self, selectList=None):
        body_pos = []
        if selectList is None:
            # body_names = self.model.body_names[1:] # ignore plane
            return self.data.body_xquat[1 : self.body_lim].copy().ravel()
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
        bone_id = self.model._body_name2id["Head"]
        head_pos = self.data.body_xpos[bone_id]
        head_quat = self.data.body_xquat[bone_id]
        return np.concatenate((head_pos, head_quat))

    def get_wbody_pos(self, selectList=None):
        body_pos = []
        if selectList is None:
            # body_names = self.model.body_names[1:] # ignore plane
            return self.data.body_xpos[1 : self.body_lim].copy().ravel()
        else:
            body_names = selectList
        for body in body_names:
            bone_idx = self.model._body_name2id[body]
            bone_vec = self.data.body_xpos[bone_idx]
            body_pos.append(bone_vec)
        return np.concatenate(body_pos)

    def get_body_com(self, selectList=None):
        body_pos = []
        if selectList is None:
            body_names = self.model.body_names[1 : self.body_lim]  # ignore plane
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
        M = M[: self.qvel_lim, : self.qvel_lim]
        C = self.data.qfrc_bias.copy()[: self.qvel_lim]
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(
            cho_factor(M + K_d * dt, overwrite_a=True, check_finite=False),
            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]),
            overwrite_b=True,
            check_finite=False,
        )
        return q_accel.squeeze()

    def compute_torque(self, ctrl):
        cfg = self.cfg
        dt = self.model.opt.timestep

        ctrl_joint = ctrl[: self.ndof] * cfg.a_scale
        qpos = self.get_humanoid_qpos()
        qvel = self.get_humanoid_qvel()
        if self.cfg.action_v == 1 or self.cfg.action_v == 2 or self.cfg.action_v == 3:
            base_pos = self.get_expert_kin_pose()
            while np.any(base_pos - qpos[7:] > np.pi):
                base_pos[base_pos - qpos[7:] > np.pi] -= 2 * np.pi
            while np.any(base_pos - qpos[7:] < -np.pi):
                base_pos[base_pos - qpos[7:] < -np.pi] += 2 * np.pi

        elif self.cfg.action_v == 0:
            base_pos = cfg.a_ref
        target_pos = base_pos + ctrl_joint

        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])

        k_p[6:] = cfg.jkp
        k_d[6:] = cfg.jkd
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:] * dt - target_pos))
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
            contact_point = vf[i * self.body_vf_dim : i * self.body_vf_dim + 3]
            force = (
                vf[i * self.body_vf_dim + 3 : i * self.body_vf_dim + 6]
                * self.cfg.residual_force_scale
            )
            torque = (
                vf[i * self.body_vf_dim + 6 : i * self.body_vf_dim + 9]
                * self.cfg.residual_force_scale
                if self.cfg.residual_force_torque
                else np.zeros(3)
            )
            contact_point = self.pos_body2world(body, contact_point)
            force = self.vec_body2world(body, force)
            torque = self.vec_body2world(body, torque)
            mjf.mj_applyFT(
                self.model, self.data, force, torque, contact_point, body_id, qfrc
            )
        self.data.qfrc_applied[:] = qfrc

    """ RFC-Implicit """

    def rfc_implicit(self, vf):
        vf *= self.cfg.residual_force_scale
        curr_root_quat = self.remove_base_rot(self.get_humanoid_qpos()[3:7])
        hq = get_heading_q(curr_root_quat)
        # hq = get_heading_q(self.get_humanoid_qpos()[3:7])
        vf[:3] = quat_mul_vec(hq, vf[:3])
        vf = np.clip(vf, -self.cfg.residual_force_lim, self.cfg.residual_force_lim)
        self.data.qfrc_applied[: vf.shape[0]] = vf

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cfg
        for i in range(n_frames):
            ctrl = action
            if cfg.action_type == "position":
                torque = self.compute_torque(ctrl)
            elif cfg.action_type == "torque":
                torque = ctrl * cfg.a_scale

            torque = np.clip(torque, -cfg.torque_lim, cfg.torque_lim)
            self.data.ctrl[:] = torque

            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[-self.vf_dim :].copy()
                if cfg.residual_force_mode == "implicit":
                    self.rfc_implicit(vf)
                else:
                    self.rfc_explicit(vf)

            try:
                self.sim.step()
            except Exception as e:
                print(e, action)

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0

    def step(self, a):
        cfg = self.cfg
        # record prev state
        self.prev_qpos = self.get_humanoid_qpos()
        self.prev_qvel = self.get_humanoid_qvel()
        self.prev_bquat = self.bquat.copy()
        # do simulation
        # if np.isnan(a).any():
        #     print(self.data_loader.curr_key)
        #     print(a)

        self.do_simulation(a, self.frame_skip)

        self.cur_t += 1

        self.bquat = self.get_body_quat()
        # get obs
        head_pos = self.get_body_com(["Head"])
        reward = 1.0
        if cfg.env_term_body == "Head":
            fail = (
                self.expert is not None
                and head_pos[2] < self.expert["head_height_lb"] - 0.1
            )
        elif cfg.env_term_body == "root":
            fail = (
                self.expert is not None
                and self.get_humanoid_qpos()[2] < self.expert["height_lb"] - 0.1
            )
        elif cfg.env_term_body == "body":
            body_diff = self.calc_body_diff()
            fail = body_diff > 0.5

        # fail = False

        end = (self.cur_t >= cfg.env_episode_len) or (
            self.cur_t + self.start_ind
            >= self.expert["len"] + cfg.env_expert_trail_steps
        )
        done = fail or end

        # if done:
        # print("done!!!", self.cur_t, self.expert['len'] )

        percent = self.cur_t / self.expert["len"]
        obs = self.get_obs()
        return obs, reward, done, {"fail": fail, "end": end, "percent": percent}

    def reset_model(self):
        cfg = self.cfg
        ind = 0
        self.start_ind = 0

        init_pose_exp = self.expert["qpos"][ind, :].copy()
        init_vel_exp = self.expert["qvel"][ind, :].copy()  ## Using GT joint velocity
        init_pose_exp[7:] += self.np_random.normal(
            loc=0.0, scale=cfg.env_init_noise, size=self.qpos_lim - 7
        )

        if cfg.reactive_v == 0:
            # self.set_state(init_pose, init_vel)
            pass
        elif cfg.reactive_v == 1:
            if self.mode == "train" and np.random.binomial(1, 1 - cfg.reactive_rate):
                # self.set_state(init_pose, init_vel)
                pass
            elif self.mode == "test":
                # self.set_state(init_pose, init_vel)
                # netural_qpos = self.netural_data['qpos']
                # init_pose_exp = self.match_heading_and_pos(init_pose_exp, netural_qpos)
                # init_vel_exp = self.netural_data['qvel']
                pass
            else:
                netural_qpos = self.netural_data["qpos"]
                init_pose_exp = self.match_heading_and_pos(init_pose_exp, netural_qpos)
                init_vel_exp = self.netural_data["qvel"]

            # self.set_state(init_pose, init_vel)
            self.bquat = self.get_body_quat()
        else:
            init_pose = self.get_humanoid_qpos()
            init_pose[2] += 1.0
            self.set_state(init_pose, self.data.qvel)

        if self.expert["has_obj"]:
            obj_pose = self.expert["obj_pose"][ind, :].copy()
            init_pose = np.concatenate([init_pose_exp, obj_pose])
            init_vel = np.concatenate(
                [init_vel_exp, np.zeros(self.expert["num_obj"] * 6)]
            )
        else:
            init_pose = init_pose_exp
            init_vel = init_vel_exp

        self.set_state(init_pose, init_vel)

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
        qpos_2[3:7] = quaternion_multiply(heading_1, qpos_2[3:7])
        qpos_2[:2] = posxy_1
        return qpos_2

    def get_expert_index(self, t):
        return (
            (self.start_ind + t) % self.expert["len"]
            if self.expert["meta"]["cyclic"]
            else min(self.start_ind + t, self.expert["len"] - 1)
        )

    def get_expert_offset(self, t):
        if self.expert["meta"]["cyclic"]:
            n = (self.start_ind + t) // self.expert["len"]
            offset = self.expert["meta"]["cycle_offset"] * n
        else:
            offset = np.zeros(2)
        return offset

    def get_expert_attr(self, attr, ind):
        return self.expert[attr][ind]

    def get_expert_qpos(self, delta_t=0):
        ind = self.get_expert_index(self.cur_t + delta_t)
        expert_qpos = self.get_expert_attr("qpos", ind)

        if self.no_root:
            expert_qpos[:3] = self.data.qpos[:3]
        return expert_qpos

    def get_expert_qvel(self, delta_t=0):
        ind = self.get_expert_index(self.cur_t + delta_t)
        expert_vel = self.get_expert_attr("qvel", ind)

        return expert_vel

    def get_expert_kin_pose(self, delta_t=0):
        return self.get_expert_qpos(delta_t=delta_t)[7:]

    def get_expert_joint_pos(self, delta_t=0):
        # world joint position
        ind = self.get_expert_index(self.cur_t + delta_t)
        wbpos = self.get_expert_attr("wbpos", ind)
        if self.no_root:
            all_wbpos = wbpos.reshape(-1, 3).copy()
            curr_root_pos = all_wbpos[0]
            curr_sim_root_pos = self.data.body_xpos[1 : self.body_lim][0]
            all_wbpos[:, :] += (curr_sim_root_pos - curr_root_pos)[:3]
            wbpos = all_wbpos.flatten()

        return wbpos

    def get_expert_com_pos(self, delta_t=0):
        # body joint position
        ind = self.get_expert_index(self.cur_t + delta_t)
        body_com = self.get_expert_attr("body_com", ind)

        if self.no_root:
            all_body_com = body_com.reshape(-1, 3).copy()
            curr_root_pos = all_body_com[0]
            curr_sim_root_pos = self.get_body_com()[:3]
            all_body_com[:, :] += (curr_sim_root_pos - curr_root_pos)[:3]
            body_com = all_body_com.flatten()

        return body_com

    def get_expert_bquat(self, delta_t=0):
        ind = self.get_expert_index(self.cur_t + delta_t)
        bquat = self.get_expert_attr("bquat", ind)

        return bquat

    def get_expert_wbquat(self, delta_t=0):
        ind = self.get_expert_index(self.cur_t + delta_t)
        wbquat = self.get_expert_attr("wbquat", ind)
        return wbquat

    def calc_body_diff(self):

        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        e_wbpos = self.get_expert_joint_pos().reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).mean()
        return jpos_dist

    def get_humanoid_qpos(self):
        return self.data.qpos.copy()[: self.qpos_lim]

    def get_humanoid_qvel(self):
        return self.data.qvel.copy()[: self.qvel_lim]

    def get_obj_qpos(self):
        return self.data.qpos.copy()[self.qpos_lim :]

    def get_obj_qvel(self):
        return self.data.qvel.copy()[self.qvel_lim :]


if __name__ == "__main__":
    pass
