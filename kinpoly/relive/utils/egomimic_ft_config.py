import yaml
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
import numpy as np
from relive.utils import recreate_dirs


class Config:

    def __init__(self, cfg_id=None, create_dirs=False, cfg_dict=None, take=None, _iter=None, action=None):
        self.id = cfg_id
        cfg_name = osp.join("config", "egomimic", f'{cfg_id}.yml')
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.load(open(cfg_name, 'r'), Loader=yaml.FullLoader)
        self.meta_id = cfg['meta_id']
        # self.data_dir = './datasets'
        self.data_dir = cfg.get('dataset_path', 'datasets')
        
        #
        self.meta = yaml.load(open(osp.join(self.data_dir, "meta", f"{self.meta_id}.yml"), 'r'), Loader=yaml.FullLoader)
        
        if action is None:  
            self.action = action  = self.meta['action_type'][take]
        else:
            self.action = action
        self.object = self.meta['object'][action]
        # create dirs
        self.base_dir = 'results'
        self.cfg_dir =  osp.join(self.base_dir, action, "egomimic", cfg_id) 
        self.ft_dir = '%s/ft/%s' % (self.cfg_dir, take)
        os.makedirs(self.ft_dir, exist_ok=True)
        self.model_dir = osp.join(self.ft_dir, "models")
        self.result_dir = osp.join(self.ft_dir, "results") 
        self.log_dir = osp.join(self.ft_dir, "log") 
        self.tb_dir = osp.join(self.ft_dir, "tb")
        self.pretrain_dir = '%s/%s/egomimic/%s/models' % (self.base_dir, self.action, cfg['pretrain_model']) 
        self.value_feat_file = '%s/%s/egomimic/%s/results/iter_%04d_test.p' % (self.base_dir, self.action, cfg['pretrain_model'], _iter)  if _iter is not None else None
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.pretrain_dir, exist_ok=True)
        if create_dirs:
            recreate_dirs(self.log_dir, self.tb_dir)
 

        # create dirs
        

        

        self.expert_feat_file = osp.join(self.data_dir, "features", f"expert_{cfg['expert_feat']}.p" ) if 'expert_feat' in cfg else None
        self.cnn_feat_file = osp.join(self.data_dir, "features",  f"cnn_feat_{cfg['cnn_feat']}_{action}.p") if 'cnn_feat' in cfg else None
        self.occup_feat_file = osp.join(self.data_dir, "features", f"occ_feat_{cfg['occ_feat']}.p") if 'occ_feat' in cfg else None
        self.kinematic_file = osp.join("results", action, "statereg", cfg.get('state_net_cfg', None), "results", f"iter_{cfg.get('state_net_iter', 0):04d}_all.p")
        self.fr_margin = cfg.get('fr_margin', 10)

        # state net
        self.state_net_cfg = cfg.get('state_net_cfg', None)
        self.state_net_iter = cfg.get('state_net_iter', None)
        if self.state_net_cfg is not None:
            self.state_net_model = osp.join(self.base_dir, action, "statereg", self.state_net_cfg, "models", f"iter_{self.state_net_iter:04d}.p")
        # training config
        #self.gamma = cfg.get('gamma', 0.95)
        self.gamma = cfg.get('ft_gamma', 0.95)
        self.tau = cfg.get('tau', 0.95)
        self.causal = cfg.get('causal', False)
        self.policy_htype = cfg.get('policy_htype', 'relu')
        self.policy_hsize = cfg.get('policy_hsize', [300, 200])
        self.policy_v_hdim = cfg.get('policy_v_hdim', 128)
        self.policy_v_net = cfg.get('policy_v_net', 'lstm')
        self.policy_v_net_param = cfg.get('policy_v_net_param', None)
        self.policy_optimizer = cfg.get('policy_optimizer', 'Adam')
        self.policy_lr = cfg.get('ft_policy_lr', 5e-5)
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
        self.adap_lr = cfg.get('ft_adap_lr', False)
        self.adv_clip = cfg.get('adv_clip', np.inf)
        self.clip_epsilon = cfg.get('clip_epsilon', 0.2)
        self.log_std = cfg.get('log_std', -2.3)
        self.fix_std = cfg.get('fix_std', False)
        self.num_optim_epoch = cfg.get('num_optim_epoch', 10)
        self.min_batch_size = cfg.get('ft_min_batch_size', 10000)
        self.max_iter_num = cfg.get('max_iter_num', 1000)
        self.seed = cfg.get('seed', 1)
        self.save_model_interval = cfg.get('ft_save_model_interval', 20)

        # Fine-tuning parameters
        self.reward_id = cfg.get('ft_reward', 'traj_dist_reward')
        self.reward_weights = cfg.get('ft_reward_weights', None)
        self.ft_max_iter = cfg.get('ft_max_iter', 1000)
        self.random_cur_t = cfg.get('random_cur_t', False)
        self.adap_weight = cfg.get('ft_adap_weight', False)

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
        self.mujoco_model_file = '%s/assets/mujoco_models/%s/%s/humanoid.xml' % (os.getcwd(), cfg['model_id'], self.action)
        self.vis_model_file = '%s/assets/mujoco_models/%s/%s/humanoid_vis.xml' % (os.getcwd(), cfg['model_id'], self.action)
        self.env_start_first = cfg.get('env_start_first', False)
        self.env_init_noise = cfg.get('env_init_noise', 0.0)
        #self.env_episode_len = cfg.get('env_episode_len', 200)
        self.env_episode_len = cfg.get('ft_env_episode_len', 200)
        self.ft_all_seq = cfg.get('ft_all_seq', False)
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
        self.obs_max_out_dist = cfg.get('obs_max_out_dist', 0.0)
        self.sync_exp_interval = cfg.get('sync_exp_interval', 100)
        self.action_type = cfg.get('action_type', 'position')
        self.action_v = cfg.get('action_v', False)
        self.pred_init = cfg.get('pred_init', False)
        self.obs_v2 = cfg.get('obs_v2', False)
        self.value_sampling = cfg.get('value_sampling', False)

        self.noise_object = False

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
