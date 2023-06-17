import yaml
import glob
import os
import sys
import os.path as osp
sys.path.append(os.getcwd())

from relive.utils import recreate_dirs

class Config:

    def __init__(self, action, cfg_id, wild=False, create_dirs=False, mujoco_path='%s.xml'):
        self.id = cfg_id
        self.action = action
        self.wild = wild
        self.all_actions = ["sit", "push", "avoid", "step"]
        cfg_name = 'config/statear/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            cfg_name = cfg_id 
            # print("Config file doesn't exist: %s" % cfg_name)
            # exit(0)
        self.yaml_data = cfg = yaml.load(open(cfg_name, 'r'), Loader=yaml.FullLoader)

        self.base_dir = '/viscam/projects/egoego/results/kinpoly'
        
        self.data_dir = cfg.get('dataset_path', '/Users/jiamanli/datasets/kin-poly/MoCapData')
        self.batch_size = cfg.get("batch_size", 128)
        
        self.cfg_dir = osp.join(self.base_dir, "all", "statear", cfg_id.split("/")[-1].replace(".yml", ""))
        self.model_dir = osp.join(self.cfg_dir, "models")
        self.policy_model_dir = osp.join(self.cfg_dir, "models_policy")
        self.result_dir = osp.join(self.cfg_dir, "results")
        self.log_dir = osp.join(self.cfg_dir, "log")
        self.tb_dir = osp.join(self.cfg_dir, "tb")
        self.tb_test_dir = osp.join(self.cfg_dir, "tb_test")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.policy_model_dir, exist_ok=True)
        
        os.makedirs(self.result_dir, exist_ok=True)

        if wild:
            self.data_file = cfg["data_wild_file"]
            self.meta_id = cfg['meta_wild_id']
            self.of_file = cfg.get("of_file_wild", "of_feat_wild_all")
        else:
            self.data_file = cfg['data_file']
            self.meta_id = cfg['meta_id']
            self.of_file = cfg.get('of_file', "of_feat_smpl_all")

        # training config
        
        self.meta = yaml.load(open(osp.join(self.data_dir, "meta", self.meta_id + ".yml"), 'r'), Loader=yaml.FullLoader)
        self.object = self.meta['object']
        self.mujoco_model_file = mujoco_path % cfg['mujoco_model']

        self.take_actions = self.meta['action_type']        
        self.all_takes = {x: self.meta[x] for x in ['train', 'test']}
        self.takes = {'train': [], 'test': []}

        for x in ['train', 'test']:
            _takes = self.all_takes[x]
            for take in _takes:
                curr_action = self.take_actions[take]
                self.takes[x].append(take)
        # if create_dirs:
            # recreate_dirs(self.log_dir, self.tb_dir)

        # training config
        self.meta_id = cfg['meta_id']
        self.seed = cfg['seed']
        self.fr_num = cfg['fr_num']
        self.v_net_param = cfg.get('v_net_param', None)
        self.lr = cfg['lr']
        self.weightdecay = cfg.get('weightdecay', 0.0)
        self.num_epoch = cfg['num_epoch']
        self.num_epoch_fix = cfg.get("num_epoch_fix", 100)
        self.iter_method = cfg['iter_method']
        self.shuffle = cfg.get('shuffle', False)
        self.num_sample = cfg.get('num_sample', 20000)
        self.save_model_interval = cfg['save_model_interval']
        self.fr_margin = cfg['fr_margin']
        self.pose_only = cfg.get('pose_only', False)
        self.causal = cfg.get('causal', False)
        self.no_cnn = cfg.get('no_cnn', False)
        self.dropout = cfg.get('dropout', False)
        self.model_specs = cfg.get("model_specs", dict())

        self.traj_only = cfg.get('traj_only', False)
        self.traj_loss = cfg.get('traj_loss', False)

        self.state_loss = cfg.get('state_loss', False)
        self.DM_loss = cfg.get('DM_loss', False)
        self.scheduled = cfg.get('scheduled', False)
        self.scheduled_k = cfg.get('scheduled_k', 0.0)
        self.noise_schedule = cfg.get('noise_schedule', False)
        self.scheduled_noise = cfg.get('scheduled_noise', 0.0)

        self.norm_pose = cfg.get('norm_pose', False)
        self.norm_obs = cfg.get('norm_obs', False)
        self.norm_state = cfg.get('norm_state', False)
        self.noise_object = cfg.get('noise_object', False)

        self.add_noise = cfg.get('add_noise', False)
        self.noise_std = cfg.get('noise_std', 0.0)
        
        self.obs_coord = cfg.get('obs_coord', 'heading')
        self.obs_heading = cfg.get('obs_heading', False)
        self.obs_vel = cfg.get('obs_vel', True)
        self.root_deheading = cfg.get('root_deheading', False)
        self.obs_global = cfg.get('obs_global', False)
        self.obs_angle = cfg.get('obs_angle', False)
        self.obs_max_out = cfg.get('obs_max_out', False)
        self.obs_quat = cfg.get('obs_quat', False)
        self.obs_max_out_dist = cfg.get('obs_max_out_dist', 0.0)
        self.obs_occ = cfg.get('obs_occ', False)
        self.obs_hum_glob = cfg.get('obs_hum_glob', False)
        self.obs_3dpoint = cfg.get('obs_3dpoint', False)
        self.augment = cfg.get('augment', False)
        self.policy_specs = cfg.get("policy_specs", {})

        self.rotrep = cfg.get("rotrep", "euler")
        self.has_z = cfg.get("has_z", True)
        self.reward_weights = cfg.get("reward_weights", {})

        self.use_of = cfg.get("use_of", True)
        self.use_head = cfg.get("use_head", True)
        self.use_action = cfg.get("use_action", True)
        self.use_vel = cfg.get("use_vel", False)
        self.use_context = cfg.get("use_context", True)

        self.use_action_transformer = cfg.get("use_action_transformer", False)
        self.use_context_transformer = cfg.get("use_context_transformer", False)

        self.use_scheduled_sampling = cfg.get("use_scheduled_sampling", True) 

        self.smooth = cfg.get("smooth", False)

        self.step_size = cfg.get("step_size", 1000) 

    def get(self, query, default_value = None):
        if hasattr(self, query):
            return getattr(self, query)
        else:
            return self.yaml_data.get(query, default_value)
    