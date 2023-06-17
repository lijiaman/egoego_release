import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import sys
import pickle
import time
import math
import torch
import numpy as np
sys.path.append(os.getcwd())

import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from relive.utils import *
from relive.models.mlp import MLP
from relive.posereg_models.video_reg_net import VideoRegNet
# from relive.models.traj_ar_kin_net import TrajARNet
from relive.data_loaders.posereg_dataset import PoseRegDataset
from relive.utils.torch_humanoid import Humanoid
from relive.data_process.process_trajs import get_expert
from relive.utils.torch_ext import get_scheduler
from relive.utils.statear_smpl_config import Config

from relive.utils.math_utils import (get_heading_q, quaternion_multiply, quaternion_inverse,
        quat_mul_vec, quat_from_expmap)

def get_traj_from_state_pred(state_pred, init_pos, init_heading, dt):
    pos = init_pos.copy()
    heading = init_heading.copy()
    traj_pred = []
    for i in range(state_pred.shape[0]):
        qpos = np.concatenate((pos, state_pred[i, :74]))
        qvel = state_pred[i, 74:]
        qpos[3:7] = quaternion_multiply(heading, qpos[3:7])
        linv = quat_mul_vec(heading, qvel[:3])
        angv = quat_mul_vec(qpos[3:7], qvel[3:6])
        pos += linv[:2] * dt
        new_q = quaternion_multiply(quat_from_expmap(angv * dt), qpos[3:7])
        heading = get_heading_q(new_q)
        traj_pred.append(qpos)
    traj_qpos = np.vstack(traj_pred)
    return traj_qpos

def eval_sequences(cur_jobs):
    with torch.no_grad():
        traj_ar_net.load_state_dict(model_cp['stateAR_net_dict'], strict=True)
        results = defaultdict(dict)
        pbar = tqdm(cur_jobs)
        counter = 0
        for seq_key, data_dict in pbar:
            data_dict = {k:torch.from_numpy(v).to(device).type(dtype) for k, v in data_dict.items()}
            # data_dict = {k:v.clone().to(device).type(dtype) for k, v in data_dict.items()}
            data_acc = defaultdict(list)
            # if args.cfg.startswith("wild"):
            #     from scripts.wild_meta import take_names
            #     start = take_names[seq_key]['start_idx']
            #     print(f"Wild start {start}")
            # action = seq_key.split("-")[0]
            # if action != "push":
            #     continue
            feature_pred = traj_ar_net.forward(data_dict)

            state_pred = feature_pred['pred_traj'].squeeze(0).data.cpu().numpy() # T X D(149=76+75-2)
            # state_pred = data_dict['posereg_traj_norm'].squeeze(0).data.cpu().numpy() 
            # traj_orig = data_dict['posereg_traj_norm'].squeeze(0).data.cpu().numpy() # T X 149 
            traj_orig = data_dict['qpos'].squeeze(0).data.cpu().numpy()

            init_pos = traj_orig[0, :2]
            init_heading = get_heading_q(traj_orig[0, 3:7])
            traj_pred = get_traj_from_state_pred(state_pred, \
                init_pos, init_heading, 1./30)

            # import pdb 
            # pdb.set_trace() 

            data_acc['qpos'] = traj_pred
            data_acc['qpos_gt'] = traj_orig[:, :76]

            # data_acc['qpos'] = feature_pred['qpos'][0].cpu().numpy()
            # data_acc['qpos_gt'] = data_dict['qpos'][0].cpu().numpy()

            data_acc['obj_pose'] = data_dict['obj_pose'][0].cpu().numpy()
            
            
            # print(orig_qpos.shape)
            # if args.smooth:
                # from scipy.ndimage import gaussian_filter1d
                # pred_qpos[:, 7:] = gaussian_filter1d(pred_qpos[:, 7:], 1, axis = 0).copy()
            results[seq_key] = data_acc
            # results["pesudo_expert"][seq_key] = get_expert(results["traj_pred"][seq_key], 0, results["traj_pred"][seq_key].shape[0], cfg=cfg, env=env)
            counter += 1
            # if counter > 1:
                # break
    return results
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--data', default=None)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--action', type=str, default='all')
    parser.add_argument('--perspective', type=str, default='first')
    parser.add_argument('--wild', action='store_true', default=False)
    args = parser.parse_args()

    if args.data is None:
        args.data = args.mode if args.mode in {'train', 'test'} else 'train'
    
    # args.data = 'train' # Temporary use, for testing training data results 

    cfg = Config(args.action, args.cfg, wild = args.wild, create_dirs=(args.iter == 0), mujoco_path = "assets/mujoco_models/%s.xml")
    
    
    """setup"""
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda', index=args.gpu_index) if (torch.cuda.is_available() and args.mode == "train") else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    tb_logger = Logger(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    print("mode:{0}".format(args.mode))

    """Datasets"""
    if args.mode == 'train':
        dataset = PoseRegDataset(cfg, args.data)
    else:
        dataset = PoseRegDataset(cfg, args.data, sim = True)
    data_sample = dataset.sample_seq()
    data_sample =  {k:v.clone().to(device).type(dtype) for k, v in data_sample.items()}

    """networks"""
    state_dim = dataset.traj_dim
    pr_state_dim = 74 + 75 
    pr_v_hdim = 128
    pr_cnn_fdim = 512  
    pr_cnn_type = "resnet"
    pr_mlp_dim = [300, 200]
    pr_v_net = "lstm"
    traj_ar_net = VideoRegNet(pr_state_dim, pr_v_hdim, pr_cnn_fdim, no_cnn=True, cnn_type=pr_cnn_type, \
                        mlp_dim=pr_mlp_dim, v_net_type=pr_v_net, \
                        v_net_param=None, causal=False)
    if args.iter > 0:
        cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
        logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp, meta = pickle.load(open(cp_path, "rb"))
        # traj_ar_net.load_state_dict(model_cp['stateAR_net_dict'], strict=True)

    traj_ar_net.to(device)
    curr_lr = cfg.lr
    optimizer = torch.optim.Adam(traj_ar_net.parameters(), lr=curr_lr, weight_decay=cfg.weightdecay)

    scheduler = get_scheduler(
        optimizer,
        policy="lambda",
        nepoch_fix=cfg.num_epoch_fix,
        nepoch=cfg.num_epoch,
    )

    fr_num_start = 80
    fr_num_end = 150

    # For debug 
    # fr_num_start = 500
    # fr_num_end = 6000
    if args.mode == 'train':
        print("Mode is training!")

        traj_ar_net.train()
        for i_epoch in range(args.iter, cfg.num_epoch):
            # sampling_rate = max((1-i_epoch/cfg.num_epoch) * 0.3 , 0)
            # fr_num = fr_num_start + i_epoch/cfg.num_epoch * (fr_num_end - fr_num_start) // 5 * 5

            # fr_num = cfg.fr_num 

            # print(f"sampling_rate: {sampling_rate:.3f}, fr_num: {fr_num}, cfg: {args.cfg}")
            
            t0 = time.time()
            losses = np.array(0)
            epoch_loss = 0
            generator = dataset.sampling_generator(num_samples= cfg.num_sample, \
                batch_size=cfg.batch_size, num_workers=2, fr_num=cfg.fr_num)
            
            pbar = tqdm(generator)
            optimizer = torch.optim.Adam(traj_ar_net.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)
            for data_dict in pbar:
                data_dict
                data_dict = {k:v.clone().to(device).type(dtype) for k, v in data_dict.items()}

                feature_pred = traj_ar_net.forward(data_dict)
                loss, loss_idv = traj_ar_net.compute_loss(feature_pred, data_dict)
                
                try:
                    # zero the gradients
                    optimizer.zero_grad()
                   
                    if torch.isnan(loss).item():
                        Logger.log('WARNING: NaN loss. Skipping to next data...')
                        torch.cuda.empty_cache()
                        continue
                    # backprop and step
                    loss.backward()
                    # check gradients
                    parameters = [p for p in traj_ar_net.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
                    if torch.isnan(total_norm):
                        Logger.log('WARNING: NaN gradients. Skipping to next data...')
                        torch.cuda.empty_cache()
                        continue
                    torch.nn.utils.clip_grad_norm_(traj_ar_net.parameters(), 0.25)
                    optimizer.step()
                except (RuntimeError, AssertionError) as e:
                    if i_epoch > 0:
                        # to catch bad dynamics, but keep training
                        Logger.log('WARNING: caught an exception during forward or backward pass. Skipping to next data...')
                        Logger.log(e)
                        traceback.print_exc()
                        continue
                    else:
                        raise e

                pbar.set_description(f"loss: {loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_idv])}]")
                epoch_loss += loss.cpu().item()
                losses = loss_idv + losses
                
            epoch_loss /= cfg.num_sample
            losses /= cfg.num_sample
            curr_lr = optimizer.param_groups[0]["lr"]
            logger.info(f'epoch {i_epoch:4d}    time {time.time() - t0:.2f}   loss {epoch_loss:.4f} {np.round(losses * 100, 4).tolist()} lr: {curr_lr} ')
            tb_logger.scalar_summary('loss', epoch_loss, i_epoch)
            [tb_logger.scalar_summary('loss' + str(i), losses[i], i_epoch) for i in range(losses.shape[0])] 
            scheduler.step()
            
            if cfg.save_model_interval > 0 and (i_epoch + 1) % cfg.save_model_interval == 0:
                with to_cpu(traj_ar_net):
                    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, i_epoch + 1)
                    model_cp = {'stateAR_net_dict': traj_ar_net.state_dict()}
                    meta = {}
                    pickle.dump((model_cp, meta), open(cp_path, 'wb'))

    elif args.mode == 'test':
        print("Mode is testing!")

       

        start = 0
        counter = 0
        traj_ar_net.eval()
        

        jobs = list(dataset.iter_data().items())
        data_res_full = eval_sequences(jobs)
        num_jobs = 5
        
        
        res_path = '%s/iter_%04d_%s_%s.p' % (cfg.result_dir, args.iter, args.data, cfg.data_file)
        print(f"results dir: {res_path}")
        pickle.dump(data_res_full, open(res_path, 'wb'))
