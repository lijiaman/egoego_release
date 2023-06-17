import sys
sys.path.append("./")
import torch
import os
import argparse
import shutil
import numpy as np 
import tqdm 
import json 

from torch.utils.tensorboard import SummaryWriter

from utils_common import get_config, make_result_folders
from utils_common import write_loss, show3Dpose_animation

from relive.data_loaders.hm_vae_dataset import HMVAEDataset 

from trainer_motion_vae import Trainer

from relive.utils.statear_smpl_config import Config

import torch.backends.cudnn as cudnn
# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # From kinpoly code base 
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--data', default=None)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--action', type=str, default='all')
    parser.add_argument('--perspective', type=str, default='first')
    parser.add_argument('--wild', action='store_true', default=False)

    # From original hm-vae code base 
    parser.add_argument('--config',
                        type=str,
                        default='',
                        help='configuration file for training and testing')
    parser.add_argument('--output_path',
                        type=str,
                        default='/viscam/u/jiamanli/hm_vae_res',
                        help="outputs path")
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=10)
    parser.add_argument('--multigpus',
                        action="store_true")
    parser.add_argument("--resume",
                        action="store_true")
    parser.add_argument('--test_model',
                        type=str,
                        default='',
                        help="trained model for evaluation")
    opts = parser.parse_args()

    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']

    trainer = Trainer(config)
    trainer.cuda()
    trainer.model = trainer.model.cuda()
    if opts.multigpus:
        ngpus = torch.cuda.device_count()
        config['gpus'] = ngpus
        print("Number of GPUs: %d" % ngpus)
        trainer.model = torch.nn.DataParallel(trainer.model, device_ids=range(ngpus))
    else:
        config['gpus'] = 1

    cfg = Config(opts.action, opts.cfg, wild = opts.wild, \
        create_dirs=(opts.iter == 0), mujoco_path = "assets/mujoco_models/%s.xml")
    
    # Set up 
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda', index=opts.gpu_index) if (torch.cuda.is_available()) else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(opts.gpu_index)

    # Datasets 
    train_dataset = HMVAEDataset(cfg, "train")

    train_dataloader = train_dataset.iter_data().items() 

    fk_bpos_seq = None 
    fk_root_v_seq = None 
    
    cur_jobs = list(train_dataset.iter_data().items())
        
    for seq_key, data_dict in cur_jobs:
        data_dict = {k:torch.from_numpy(v).to(device).type(dtype) for k, v in data_dict.items()}
          
        fk_bpos, fk_root_v = trainer.model.gen_mean_std(data_dict)
                
        if fk_bpos_seq is None:
            fk_bpos_seq = fk_bpos.squeeze(0)
        else:
            fk_bpos_seq = torch.cat((fk_bpos_seq, fk_bpos.squeeze(0)), dim=0) 

        if fk_root_v_seq is None:
            fk_root_v_seq = fk_root_v.squeeze(0)
        else:
            fk_root_v_seq = torch.cat((fk_root_v_seq, fk_root_v.squeeze(0)), dim=0)

    fk_bpos_arr = fk_bpos_seq.data.cpu().numpy() # T X (24*3)
    fk_root_v_arr = fk_root_v_seq .data.cpu().numpy() # (T-1) X 3

    fk_bpos_mean = np.mean(fk_bpos_arr, axis=0) # 72 
    fk_bpos_std = np.std(fk_bpos_arr, axis=0)

    fk_root_v_mean = np.mean(fk_root_v_arr, axis=0) # 3 
    fk_root_v_std = np.std(fk_root_v_arr, axis=0) 

    print("fk_bpos mean:{0}".format(fk_bpos_mean.shape))
    print("fk_root_v mean:{0}".format(fk_root_v_mean.shape))

    dest_json_path = "/viscam/u/jiamanli/github/egomotion/kin-poly/for_hm_vae_traj_model_mean_std.json"
    dest_json_dict = {}
    dest_json_dict['fk_bpos_mean'] = fk_bpos_mean.tolist() 
    dest_json_dict['fk_bpos_std'] = fk_bpos_std.tolist() 
    dest_json_dict['fk_root_v_mean'] = fk_root_v_mean.tolist()
    dest_json_dict['fk_root_v_std'] = fk_root_v_std.tolist()

    json.dump(dest_json_dict, open(dest_json_path, 'w'))
