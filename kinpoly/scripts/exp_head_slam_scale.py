import sys 
sys.path.append("../")

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
from relive.models.head_mapping_slam_scale_transformer import HeadSlamScaleTransformer 
from relive.data_loaders.head_mapping_dataset import HeadMappingDataset
from relive.utils.torch_ext import get_scheduler
from relive.utils.statear_smpl_config import Config
from relive.utils.torch_utils import rotation_from_quaternion

from utils_common_root import vis_single_head_pose_traj, vis_multiple_head_pose_traj

def eval_sequences(cur_jobs):
    with torch.no_grad():
        traj_ar_net.load_state_dict(model_cp['stateAR_net_dict'], strict=True)
        results = defaultdict(dict)
        pbar = tqdm(cur_jobs)
        counter = 0
        for seq_key, data_dict in pbar:
            data_dict = {k:torch.from_numpy(v).to(device).type(dtype) for k, v in data_dict.items()}
    
            data_acc = defaultdict(list)
   
            feature_pred = traj_ar_net.forward(data_dict)
            
            data_acc['head_pose'] = feature_pred['head_pose'][0].cpu().numpy()
            data_acc['head_pose_gt'] = data_dict['head_pose'][0].cpu().numpy()
            
            results[seq_key] = data_acc
          
            counter += 1
           
    return results

def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50
    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat

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
        args.data = args.mode if args.mode in {'train', 'val', 'test'} else 'train'

    cfg = Config(args.action, args.cfg, wild = args.wild, \
        create_dirs=(args.iter == 0), mujoco_path = "assets/mujoco_models/%s.xml")
    
    """setup"""
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda', index=args.gpu_index) if (torch.cuda.is_available()) else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    tb_logger = Logger(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """Datasets"""
    if args.mode == 'train':
        dataset = HeadMappingDataset(cfg, args.data) 
    else:
        dataset = HeadMappingDataset(cfg, args.data)

    # For validation
    val_dataset = HeadMappingDataset(cfg, "test")

    """networks"""
    state_dim = dataset.traj_dim
    traj_ar_net = HeadSlamScaleTransformer(cfg, device=device, dtype=dtype, mode=args.mode)
    if args.iter > 0:
        cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
        logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp, meta = pickle.load(open(cp_path, "rb"))
        traj_ar_net.load_state_dict(model_cp['stateAR_net_dict'], strict=True)

    traj_ar_net.to(device)
    curr_lr = cfg.lr
    optimizer = torch.optim.Adam(traj_ar_net.parameters(), lr=curr_lr, weight_decay=cfg.weightdecay)

    scheduler = get_scheduler(
        optimizer,
        policy="step",
        decay_step=cfg.step_size
    )

    if args.mode == 'train':
        torch.autograd.set_detect_anomaly(True)
        traj_ar_net.train()
        for i_epoch in range(args.iter, cfg.num_epoch):
            # fr_num = fr_num_start + i_epoch/cfg.num_epoch * (fr_num_end - fr_num_start) // 5 * 5
            # print(f"fr_num: {fr_num}, cfg: {args.cfg}")
            
            t0 = time.time()
            losses = np.array(0)
            val_losses = np.array(0)
            epoch_loss = 0
            epoch_val_loss = 0
            generator = dataset.sampling_generator(num_samples=cfg.num_sample, \
                batch_size=cfg.batch_size, num_workers=2, fr_num=cfg.fr_num)
            
            pbar = tqdm(generator)
            for data_dict in pbar:
                data_dict = {k:v.clone().to(device).type(dtype) for k, v in data_dict.items()}

                feature_pred = traj_ar_net.forward(data_dict)
                loss, loss_idv = traj_ar_net.compute_loss(feature_pred, data_dict)
                
                # mem_info = torch.cuda.mem_get_info()
                # print("Memory info:{0}".format(mem_info))
                # pbar.set_description(f"Memory Info: {mem_info}")

                optimizer.zero_grad()
                loss.backward()   # Testing GT
                torch.nn.utils.clip_grad_norm_(traj_ar_net.parameters(), 0.25)
                optimizer.step()  # Testing GT

                pbar.set_description(f"loss: {loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_idv])}]")
                epoch_loss += loss.cpu().item()
                losses = loss_idv + losses
                # break 

            if (i_epoch+1) % 100 == 0:
                with torch.no_grad():
                    val_generator = val_dataset.sampling_generator_for_val(num_samples=cfg.num_sample, \
                        batch_size=cfg.batch_size, num_workers=0, fr_num=fr_num)

                    # Run validation data 
                    for it, val_data_dict in enumerate(val_generator):
                        val_data_dict = {k:v.clone().to(device).type(dtype) for k, v in val_data_dict.items()}

                        val_feature_pred = traj_ar_net.forward(val_data_dict)
                        val_loss, val_loss_idv = traj_ar_net.compute_loss(val_feature_pred, val_data_dict)

                        # pbar.set_description(f"loss: {loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_idv])}]")
                        epoch_val_loss += val_loss.cpu().item()
                        val_losses = val_loss_idv + val_losses

                        # Add head pose visualization 
                        if (i_epoch+1) % 100 == 0 and it == 0:
                            head_pred_pose = val_feature_pred['head_pose'] # B X T X 7 (3 + 4) 
                            head_gt_pose = val_data_dict['head_pose']
                            # head_pred_pose = val_feature_pred['gt_head_pose']
                            num_seq = head_pred_pose.shape[0]
                            for v_idx in range(4):
                                dest_vis_folder = os.path.join(cfg.log_dir, "head_traj_vis", str(i_epoch)+"_b_"+str(v_idx))
                                if not os.path.exists(dest_vis_folder):
                                    os.makedirs(dest_vis_folder)
                                anim_gt_head_seq_path = os.path.join(dest_vis_folder, "gt_head_traj.gif")
                                anim_pred_head_seq_path = os.path.join(dest_vis_folder, "pred_head_traj.gif")
                                anim_cmp_seq_path = os.path.join(dest_vis_folder, "cmp_head_traj.gif")

                                head_gt_trans = head_gt_pose[v_idx, :, :3] # T X 3
                                head_gt_quat = head_gt_pose[v_idx, :, 3:] # T X 4
                                head_pred_trans = head_pred_pose[v_idx, :, :3] # T X 3
                                head_pred_quat = head_pred_pose[v_idx, :, 3:] # T X 4 

                                head_gt_rot_mat = quat2mat(head_gt_quat)
                                head_pred_rot_mat = quat2mat(head_pred_quat)

                                vis_single_head_pose_traj(head_gt_trans, head_gt_rot_mat, anim_gt_head_seq_path)
                                vis_single_head_pose_traj(head_pred_trans, head_pred_rot_mat, anim_pred_head_seq_path)
                                vis_multiple_head_pose_traj([head_gt_trans, head_pred_trans], \
                                    [head_gt_rot_mat, head_pred_rot_mat], anim_cmp_seq_path)

                    epoch_val_loss /= val_dataset.__len__()
                    val_losses /= val_dataset.__len__() 

                    logger.info(f'epoch {i_epoch:4d}    time {time.time() - t0:.2f}   val loss {epoch_val_loss:.4f} {np.round(val_losses * 100, 4).tolist()} ')
                
                    tb_logger.scalar_summary('val_loss', epoch_val_loss, i_epoch)
                    [tb_logger.scalar_summary('val_loss' + str(i), val_losses[i], i_epoch) for i in range(val_losses.shape[0])] 
            
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
        start = 0
        counter = 0
        traj_ar_net.eval()

        vis_res = False  
        with torch.no_grad():
            if vis_res:
                val_generator = dataset.sampling_generator_for_val(num_samples=cfg.num_sample, \
                    batch_size=cfg.batch_size, num_workers=0, fr_num=cfg.fr_num)

                # Run validation data 
                for it, val_data_dict in enumerate(val_generator):
                    val_data_dict = {k:v.clone().to(device).type(dtype) for k, v in val_data_dict.items()}

                    val_feature_pred = traj_ar_net.forward(val_data_dict)

                    # Add head pose visualization 
                    head_pred_pose = val_feature_pred['head_pose'] # B X T X 7 (3 + 4) 
                    head_gt_pose = val_data_dict['head_pose']
                    num_seq = head_pred_pose.shape[0]
                    for v_idx in range(5):
                        dest_vis_folder = os.path.join(cfg.log_dir, "train_head_traj_vis", str(it)+"_b_"+str(v_idx))
                        if not os.path.exists(dest_vis_folder):
                            os.makedirs(dest_vis_folder)
                        anim_gt_head_seq_path = os.path.join(dest_vis_folder, "gt_head_traj.gif")
                        anim_pred_head_seq_path = os.path.join(dest_vis_folder, "pred_head_traj.gif")
                        anim_cmp_seq_path = os.path.join(dest_vis_folder, "cmp_head_traj.gif")

                        head_gt_trans = head_gt_pose[v_idx, :, :3] # T X 3
                        head_gt_quat = head_gt_pose[v_idx, :, 3:] # T X 4
                        head_pred_trans = head_pred_pose[v_idx, :, :3] # T X 3
                        head_pred_quat = head_pred_pose[v_idx, :, 3:] # T X 4 

                        head_gt_rot_mat = quat2mat(head_gt_quat)
                        head_pred_rot_mat = quat2mat(head_pred_quat)

                        vis_single_head_pose_traj(head_gt_trans, head_gt_rot_mat, anim_gt_head_seq_path)
                        vis_single_head_pose_traj(head_pred_trans, head_pred_rot_mat, anim_pred_head_seq_path)
                        vis_multiple_head_pose_traj([head_gt_trans, head_pred_trans], \
                            [head_gt_rot_mat, head_pred_rot_mat], anim_cmp_seq_path)
            else: 
                vis_debug = False   
                # Compute metrics 
                cur_jobs = list(val_dataset.iter_data().items())
        
                results = defaultdict(dict)

                dest_root_folder = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/slam_scale_res"
                # dest_root_folder = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/head_estimator_res"
                # dest_root_folder = "/viscam/u/jiamanli/datasets/gimo_processed/head_estimator_res"
                # dest_root_folder = "/viscam/u/jiamanli/datasets/kin-poly/head_estimator_res"
                tmp_dict = {}
                for seq_key, data_dict in cur_jobs:
                    
                    val_data_dict = {k:torch.from_numpy(v).to(device).type(dtype) for k, v in data_dict.items()}

                    val_feature_pred = traj_ar_net.forward_for_eval(val_data_dict)

                    head_pred_pose = val_feature_pred['head_pose'] # B X T X 7 (3 + 4) 
                    head_gt_pose = val_data_dict['head_pose']

                    scene_name = seq_key.split("-")[0]
                    curr_folder = os.path.join(dest_root_folder, scene_name)
                    if not os.path.exists(curr_folder):
                        os.makedirs(curr_folder)
                    
                    final_seq_name = "-".join(seq_key.split("-")[1:])
                    # final_seq_name = "_".join(final_seq_name.split("_")[:3])+".npy"
                    # final_seq_name = seq_key.split("-")[1].replace(".npz", ".npy")
                    final_seq_name = final_seq_name.replace(".npz", ".npy")

                    dest_npy_path = os.path.join(curr_folder, final_seq_name)
                    if dest_npy_path not in tmp_dict:
                        tmp_dict[dest_npy_path] = 1
                    else:
                        import pdb 
                        pdb.set_trace() 
                    np.save(dest_npy_path, head_pred_pose[0].data.cpu().numpy()) # T X 7 

                    if vis_debug:
                        dest_vis_folder = os.path.join(cfg.log_dir, "eval_debug_head_traj_vis", seq_key)
                        if not os.path.exists(dest_vis_folder):
                            os.makedirs(dest_vis_folder)
                        anim_gt_head_seq_path = os.path.join(dest_vis_folder, "gt_head_traj.gif")
                        anim_pred_head_seq_path = os.path.join(dest_vis_folder, "pred_head_traj.gif")
                        anim_cmp_seq_path = os.path.join(dest_vis_folder, "cmp_head_traj.gif")

                        e_idx = 120
                        head_gt_trans = head_gt_pose[0, :e_idx, :3] # T X 3
                        head_gt_quat = head_gt_pose[0, :e_idx, 3:] # T X 4
                        head_pred_trans = head_pred_pose[0, :e_idx, :3] # T X 3
                        head_pred_quat = head_pred_pose[0, :e_idx, 3:] # T X 4 

                        head_gt_rot_mat = quat2mat(head_gt_quat)
                        head_pred_rot_mat = quat2mat(head_pred_quat)

                        vis_single_head_pose_traj(head_gt_trans, head_gt_rot_mat, anim_gt_head_seq_path)
                        vis_single_head_pose_traj(head_pred_trans, head_pred_rot_mat, anim_pred_head_seq_path)
                        vis_multiple_head_pose_traj([head_gt_trans, head_pred_trans], \
                            [head_gt_rot_mat, head_pred_rot_mat], anim_cmp_seq_path)
                # import pdb 
                # pdb.set_trace() 



        # jobs = list(dataset.iter_data().items())
        # data_res_full = eval_sequences(jobs)
        # num_jobs = 5
        
        # res_path = '%s/iter_%04d_%s_%s.p' % (cfg.result_dir, args.iter, args.data, cfg.data_file)
        # print(f"results dir: {res_path}")
        # pickle.dump(data_res_full, open(res_path, 'wb'))
