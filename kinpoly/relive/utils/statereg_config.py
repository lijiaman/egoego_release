import yaml
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from relive.utils import recreate_dirs



class Config:

    def __init__(self, action, cfg_id, create_dirs=False, meta_id=None):
        self.id = cfg_id
        self.action = action
        cfg_name = f'config/statereg/{cfg_id}.yml' 
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.load(open(cfg_name, 'r'), Loader=yaml.FullLoader)

        # create dirs
        self.base_dir = 'results'
        self.data_dir = cfg.get('dataset_path', 'datasets')
        
        self.cfg_dir = osp.join(self.base_dir, action, "statereg", cfg_id)
        self.model_dir = osp.join(self.cfg_dir, "models")
        self.result_dir = osp.join(self.cfg_dir, "results")
        self.log_dir = osp.join(self.cfg_dir, "log")
        self.tb_dir = osp.join(self.cfg_dir, "tb")
        self.data_file = cfg.get("data_file", "traj_all")
        self.mujoco_model_file = osp.join(os.getcwd(), "assets", "mujoco_models", cfg.get('model_id', "1125"), action, "humanoid.xml")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        if create_dirs:
            recreate_dirs(self.log_dir, self.tb_dir)

        # training config
        if meta_id is None:
            self.meta_id = cfg['meta_id']
        else:
            self.meta_id = meta_id
        
        self.meta = yaml.load(open(osp.join(self.data_dir, "meta", f"{self.meta_id}.yml"), 'r'), Loader=yaml.FullLoader)
        self.object = self.meta['object'][action]

        self.take_actions = self.meta['action_type']        
        self.all_takes = {x: self.meta[x] for x in ['train', 'test']}
        self.takes = {'train': [], 'test': []}
        for x in ['train', 'test']:
            _takes = self.all_takes[x]

            for take in _takes:
                if self.take_actions[take] == action:
                    self.takes[x].append(take)

        self.seed = cfg['seed']
        self.fr_num = cfg['fr_num']
        self.v_net = cfg.get('v_net', 'lstm')
        self.v_net_param = cfg.get('v_net_param', None)
        self.v_hdim = cfg['v_hdim']
        self.mlp_dim = cfg['mlp_dim']
        self.cnn_fdim = cfg['cnn_fdim']
        self.optimizer = cfg.get('optimizer', 'Adam')
        self.lr = cfg['lr']
        self.num_epoch = cfg['num_epoch']
        self.iter_method = cfg['iter_method']
        self.shuffle = cfg.get('shuffle', False)
        self.num_sample = cfg.get('num_sample', 20000)
        self.save_model_interval = cfg['save_model_interval']
        self.fr_margin = cfg['fr_margin']
        self.pose_only = cfg.get('pose_only', False)
        self.causal = cfg.get('causal', False)
        self.cnn_type = cfg.get('cnn_type', 'resnet')
        self.is_dropout = cfg.get('dropout', False)
        self.weightdecay = cfg.get('weight_decay', 0.0)
        self.augment = cfg.get('augment', False)

        self.humanoid_model = cfg['humanoid_model']
        self.vis_model = cfg['vis_model']
        self.rotrep = cfg.get("rotrep", "euler")
        self.batch_size = cfg.get("batch_size", 512)
        self.add_noise = cfg.get("add_noise", False)
