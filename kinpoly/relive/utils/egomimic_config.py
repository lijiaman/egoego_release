import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import yaml
import numpy as np
from relive.utils import recreate_dirs


class Config:

    def __init__(self, action, cfg_id, create_dirs=False):
        self.action = action
        self.id = cfg_id
        cfg_name = osp.join("config", "egomimic", f'{cfg_id}.yml')
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.load(open(cfg_name, 'r'), Loader=yaml.FullLoader)
        # create dirs
        self.base_dir = 'results'
        self.cfg_dir =  osp.join(self.base_dir, action, "egomimic", cfg_id) 
        self.model_dir = osp.join(self.cfg_dir, "models")
        self.result_dir = osp.join(self.cfg_dir, "results") 
        self.log_dir = osp.join(self.cfg_dir, "log") 
        self.tb_dir = osp.join(self.cfg_dir, "tb")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        if create_dirs: 
            recreate_dirs(self.log_dir, self.tb_dir)

        # data
        self.meta_id = cfg['meta_id']
        # self.data_dir = './datasets'
        self.data_dir = cfg.get('dataset_path', 'datasets')
        #
        self.meta = yaml.load(open(osp.join(self.data_dir, "meta", f"{self.meta_id}.yml"), 'r'), Loader=yaml.FullLoader)
        self.object = self.meta['object'][action]

        self.take_actions = self.meta['action_type']
        self.offset_z = self.meta['offset_z'][action]
        
        self.all_takes = {x: self.meta[x] for x in ['train', 'test']}
        self.takes = {'train': [], 'test': []}

        for x in ['train', 'test']:
            _takes = self.all_takes[x]
            for take in _takes:
                if self.take_actions[take] == action:
                    self.takes[x].append(take)
        
        
        self.expert_feat_file = osp.join(self.data_dir, "features", f"expert_{cfg['expert_feat']}.p" ) if 'expert_feat' in cfg else None
        self.cnn_feat_file = osp.join(self.data_dir, "features",  f"cnn_feat_{cfg['cnn_feat']}_{action}.p") if 'cnn_feat' in cfg else None
        self.occup_feat_file = osp.join(self.data_dir, "features", f"occ_feat_{cfg['occ_feat']}.p") if 'occ_feat' in cfg else None
        if cfg.get("statenet", "reg") == "reg":
            self.kinematic_file = osp.join("results", action, "statereg", cfg.get('state_net_cfg', None), "results", f"iter_{cfg.get('state_net_iter', 0):04d}_all.p")
        elif cfg.get("statenet", "reg") == "ar":
            self.kinematic_file = osp.join("results", action, "statear", cfg.get('state_net_cfg', None), "results", f"iter_{cfg.get('state_net_iter', 0):04d}_all.p")


        self.fr_margin = cfg.get('fr_margin', 10)

        # state net
        self.state_net_cfg = cfg.get('state_net_cfg', None)
        self.state_net_iter = cfg.get('state_net_iter', None)
        if self.state_net_cfg is not None:
            self.state_net_model = osp.join(self.base_dir, action, "statereg", self.state_net_cfg, "models", f"iter_{self.state_net_iter:04d}.p")
        # training config
        self.gamma = cfg.get('gamma', 0.95)
        self.tau = cfg.get('tau', 0.95)
        self.causal = cfg.get('causal', False)
        self.policy_htype = cfg.get('policy_htype', 'relu')
        self.policy_hsize = cfg.get('policy_hsize', [300, 200])
        self.policy_v_hdim = cfg.get('policy_v_hdim', 128)
        self.policy_v_net = cfg.get('policy_v_net', 'lstm')
        self.policy_v_net_param = cfg.get('policy_v_net_param', None)
        self.policy_optimizer = cfg.get('policy_optimizer', 'Adam')
        self.policy_lr = cfg.get('policy_lr', 5e-5)
        self.policy_momentum = cfg.get('policy_momentum', 0.0)
        self.policy_weightdecay = cfg.get('policy_weightdecay', 0.0)
        self.value_htype = cfg.get('value_htype', 'relu')
        self.value_hsize = cfg.get('value_hsize', [300, 200])
        self.value_v_hdim = cfg.get('value_v_hdim', 128)
        self.value_v_net = cfg.get('value_v_net', 'lstm')
        self.value_v_net_param = cfg.get('value_v_net_param', None)
        self.value_optimizer = cfg.get('value_optimizer', 'Adam')
        self.value_lr = cfg.get('value_lr', 3e-4)
        self.value_momentum = cfg.get('value_momentum', 0.0)
        self.value_weightdecay = cfg.get('value_weightdecay', 0.0)
        self.adv_clip = cfg.get('adv_clip', np.inf)
        self.clip_epsilon = cfg.get('clip_epsilon', 0.2)
        self.log_std = cfg.get('log_std', -2.3)
        self.fix_std = cfg.get('fix_std', False)
        self.num_optim_epoch = cfg.get('num_optim_epoch', 10)
        self.min_batch_size = cfg.get('min_batch_size', 50000)
        self.max_iter_num = cfg.get('max_iter_num', 1000)
        self.seed = cfg.get('seed', 1)
        self.save_model_interval = cfg.get('save_model_interval', 100)
        self.reward_id = cfg.get('reward_id', 'quat')
        self.reward_weights = cfg.get('reward_weights', None)
        self.random_cur_t = cfg.get('random_cur_t', False)

        # adaptive parameters
        self.adp_iter_cp = np.array(cfg.get('adp_iter_cp', [0]))
        self.adp_noise_rate_cp = np.array(cfg.get('adp_noise_rate_cp', [1.0]))
        self.adp_noise_rate_cp = np.pad(self.adp_noise_rate_cp, (0, self.adp_iter_cp.size - self.adp_noise_rate_cp.size), 'edge')
        self.adp_log_std_cp = np.array(cfg.get('adp_log_std_cp', [self.log_std]))
        self.adp_log_std_cp = np.pad(self.adp_log_std_cp, (0, self.adp_iter_cp.size - self.adp_log_std_cp.size), 'edge')
        self.adp_policy_lr_cp = np.array(cfg.get('adp_policy_lr_cp', [self.policy_lr]))
        self.adp_policy_lr_cp = np.pad(self.adp_policy_lr_cp, (0, self.adp_iter_cp.size - self.adp_policy_lr_cp.size), 'edge')
        self.adp_noise_rate = None
        self.adp_log_std = None
        self.adp_policy_lr = None

        # env config
        
        self.mujoco_model_file = osp.join(os.getcwd(), "assets", "mujoco_models", cfg['model_id'], action, "humanoid.xml")
        self.vis_model_file = osp.join(os.getcwd(), "assets/mujoco_models", cfg['model_id'], action, "humanoid_vis.xml")
        self.env_start_first = cfg.get('env_start_first', False)
        self.env_init_noise = cfg.get('env_init_noise', 0.0)
        self.env_episode_len = cfg.get('env_episode_len', 200)
        self.random_start_ind = cfg.get('random_start_ind', False)
        self.obs_type = cfg.get('obs_type', 'full')
        self.obs_coord = cfg.get('obs_coord', 'heading')
        self.obs_heading = cfg.get('obs_heading', False)
        self.obs_vel = cfg.get('obs_vel', 'full')
        self.root_deheading = cfg.get('root_deheading', False)
        self.obs_global = cfg.get('obs_global', False)
        self.obs_angle = cfg.get('obs_angle', False)
        self.obs_max_out = cfg.get('obs_max_out', False)
        self.obs_quat = cfg.get('obs_quat', False)
        self.obs_occup = cfg.get('obs_occup', False)
        self.obs_polar = cfg.get('obs_polar', False)
        self.obs_expert = cfg.get('obs_expert', False)
        self.obs_max_out_dist = cfg.get('obs_max_out_dist', 0.0)
        self.sync_exp_interval = cfg.get('sync_exp_interval', 100)
        self.action_type = cfg.get('action_type', 'position')
        self.action_v = cfg.get('action_v', False)
        self.pred_init = cfg.get('pred_init', False)
        self.obs_v2 = cfg.get('obs_v2', False)
        self.pose_only = cfg.get('pose_only', False)
        self.noise_object = cfg.get('noise_object', False)
        self.residual_force_scale = cfg.get("residual_force_scale", 100)

        # joint param
        if 'joint_params' in cfg:
            jparam = zip(*cfg['joint_params'])
            jparam = [np.array(p) for p in jparam]
            self.jkp, self.jkd, self.a_ref, self.a_scale, self.torque_lim = jparam[1:6]
            self.a_ref = np.deg2rad(self.a_ref)
            jkp_multiplier = cfg.get('jkp_multiplier', 1.0)
            jkd_multiplier = cfg.get('jkd_multiplier', jkp_multiplier)
            self.jkp *= jkp_multiplier
            self.jkd *= jkd_multiplier

        # body param
        if 'body_params' in cfg:
            bparam = zip(*cfg['body_params'])
            bparam = [np.array(p) for p in bparam]
            self.b_diffw = bparam[1]

    def update_adaptive_params(self, i_iter):
        cp = self.adp_iter_cp
        ind = np.where(i_iter >= cp)[0][-1]
        nind = ind + int(ind < len(cp) - 1)
        t = (i_iter - self.adp_iter_cp[ind]) / (cp[nind] - cp[ind]) if nind > ind else 0.0
        self.adp_noise_rate = self.adp_noise_rate_cp[ind] * (1-t) + self.adp_noise_rate_cp[nind] * t
        self.adp_log_std = self.adp_log_std_cp[ind] * (1-t) + self.adp_log_std_cp[nind] * t
        self.adp_policy_lr = self.adp_policy_lr_cp[ind] * (1-t) + self.adp_policy_lr_cp[nind] * t
