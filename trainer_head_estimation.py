import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import os
from pathlib import Path
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW

import wandb 

import pytorch3d.transforms as transforms 

from egoego.data.realworld_headpose_dataset import RealWorldHeadPoseDataset
from egoego.data.ares_headpose_dataset import ARESHeadPoseDataset
from egoego.data.gimo_headpose_dataset import GIMOHeadPoseDataset
from egoego.model.head_estimation_transformer import HeadFormer 
from egoego.vis.head_motion import vis_single_head_pose_traj, vis_multiple_head_pose_traj

def train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    epochs = opt.epochs
    save_interval = opt.save_interval

    use_wandb = True 
                          
    # Loggers
    if use_wandb:
        wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, name=opt.exp_name, dir=opt.save_dir)

    if opt.train_on_ares:
        # Load HeadPoseDataset and prepare dataloader
        train_dataset = ARESHeadPoseDataset(opt.data_root_folder, train=True, window=opt.window)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.workers, pin_memory=True, drop_last=False)

        val_dataset = ARESHeadPoseDataset(opt.data_root_folder, train=False, window=opt.window)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.workers, pin_memory=True, drop_last=False)
    elif opt.train_on_gimo:
        # Load HeadPoseDataset and prepare dataloader
        train_dataset = GIMOHeadPoseDataset(opt.data_root_folder, train=True, window=opt.window)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.workers, pin_memory=True, drop_last=False)

        val_dataset = GIMOHeadPoseDataset(opt.data_root_folder, train=False, window=opt.window)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.workers, pin_memory=True, drop_last=False)
    else:
        # Load HeadPoseDataset and prepare dataloader
        train_dataset = RealWorldHeadPoseDataset(opt.data_root_folder, train=True, window=opt.window)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.workers, pin_memory=True, drop_last=False)

        val_dataset = RealWorldHeadPoseDataset(opt.data_root_folder, train=False, window=opt.window)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.workers, pin_memory=True, drop_last=False)

    # Define transformer model 
    transformer_encoder = HeadFormer(opt, device)
    transformer_encoder.to(device)

    optim = AdamW(params=transformer_encoder.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.3)

    for epoch in range(1, epochs + 1):
        recon_rot_loss = []
        recon_dist_loss = []
        recon_va_loss = []
        total_loss_list = []
        for it, input_data_dict in enumerate(train_loader):
            output = transformer_encoder(input_data_dict)

            total_g_loss, rot_loss, va_loss, dist_loss = transformer_encoder.compute_loss(output, \
            input_data_dict)

            recon_rot_loss.append(rot_loss)
            recon_va_loss.append(va_loss) 
            recon_dist_loss.append(dist_loss)
            total_loss_list.append(total_g_loss)
        
            optim.zero_grad()
            total_g_loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_encoder.parameters(), 1.0, error_if_nonfinite=False)
            optim.step()

            if it % 10 == 0:
                print("Epoch: {0}, Iter: {1}".format(epoch, it))
                print('Training: Total loss: %.4f, Rotation loss: %.4f, Angular velocity loss: %.4f, Distance loss: %.4f' % \
                (total_g_loss, rot_loss, va_loss, dist_loss))
           
            # Check loss in validation set
            val_recon_rot_loss = []
            val_recon_va_loss = []
            val_recon_dist_loss = []
            val_total_loss_list = []
            if (epoch + 1) % opt.validation_iter == 0 and it == 0:
                transformer_encoder.eval()
                with torch.no_grad():
                    for val_it, val_input_data_dict in enumerate(val_loader):
                        if val_it >= 50:
                            break

                        val_output = transformer_encoder(val_input_data_dict)

                        val_total_g_loss, val_rot_loss, val_va_loss, val_dist_loss = \
                        transformer_encoder.compute_loss(val_output, val_input_data_dict)
                        
                        val_recon_rot_loss.append(val_rot_loss)
                        val_recon_va_loss.append(val_va_loss)
                        val_recon_dist_loss.append(val_dist_loss)
                        val_total_loss_list.append(val_total_g_loss)
                        
                        print("*********************************************************************************************")
                        print('Validation: Total loss: %.4f, Rotation loss: %.4f, Angular velocity loss: %.4f, Distance loss: %.4f' % \
                        (val_total_g_loss, val_rot_loss, val_va_loss, val_dist_loss))

                transformer_encoder.train()

            # Visulization
            if (epoch + 1) % opt.image_save_iter == 0 and it == 0:
                transformer_encoder.eval() # Super important!!!
                with torch.no_grad():
                    for test_it, test_input_data_dict in enumerate(val_loader):
                        if test_it >= 4:
                            break
                        
                        test_output = transformer_encoder(test_input_data_dict)
                            
                        head_pred_rot_quat = test_output['head_rot_quat'] # B X T X 4
                        head_gt_pose = test_input_data_dict['head_pose']
            
                        for v_idx in range(1):
                            dest_vis_folder = os.path.join(opt.save_dir, str(epoch), "test_it_" + str(test_it)+"_"+str(v_idx))
                            if not os.path.exists(dest_vis_folder):
                                os.makedirs(dest_vis_folder)

                            anim_gt_head_seq_path = os.path.join(dest_vis_folder, "gt_head_traj.gif")
                            anim_pred_head_seq_path = os.path.join(dest_vis_folder, "pred_head_traj.gif")
                            anim_cmp_seq_path = os.path.join(dest_vis_folder, "cmp_head_traj.gif")

                            head_gt_quat = head_gt_pose[v_idx, :, 3:] # T X 4
                            head_pred_quat = head_pred_rot_quat[v_idx, :, :] # T X 4 

                            # Tmp use, set translation to 0, only compare rotation
                            head_gt_trans = torch.zeros(head_gt_quat.shape[0], 3).to(head_gt_quat.device)
                            head_pred_trans = torch.zeros(head_pred_quat.shape[0], 3).to(head_pred_quat.device)

                            head_gt_rot_mat = transforms.quaternion_to_matrix(head_gt_quat)
                            head_pred_rot_mat = transforms.quaternion_to_matrix(head_pred_quat)

                            vis_single_head_pose_traj(head_gt_trans, head_gt_rot_mat, anim_gt_head_seq_path)
                            vis_single_head_pose_traj(head_pred_trans, head_pred_rot_mat, anim_pred_head_seq_path)
                            vis_multiple_head_pose_traj([head_gt_trans, head_pred_trans], \
                                [head_gt_rot_mat, head_pred_rot_mat], anim_cmp_seq_path)

                transformer_encoder.train()

            
            # Log
            if len(val_total_loss_list) == 0:
                log_dict = {
                "Train/Loss/Rotatation Loss": torch.stack(recon_rot_loss).mean().item(),
                "Train/Loss/Angular velocity Loss": torch.stack(recon_va_loss).mean().item(),
                "Train/Loss/Distance Loss": torch.stack(recon_dist_loss).mean().item(),
                "Train/Loss/Total Loss": torch.stack(total_loss_list).mean().item(),
                }
            else:
                log_dict = {
                    "Train/Loss/Rotatation Loss": torch.stack(recon_rot_loss).mean().item(),
                    "Train/Loss/Angular velocity Loss": torch.stack(recon_va_loss).mean().item(),
                    "Train/Loss/Distance Loss": torch.stack(recon_dist_loss).mean().item(),
                    "Train/Loss/Total Loss": torch.stack(total_loss_list).mean().item(),
                    "Val/Loss/Rotatation Loss": torch.stack(val_recon_rot_loss).mean().item(),
                    "Val/Loss/Angular velocity Loss": torch.stack(val_recon_va_loss).mean().item(),
                    "Val/Loss/Distance Loss": torch.stack(val_recon_dist_loss).mean().item(),
                    "Val/Loss/Total Loss": torch.stack(val_total_loss_list).mean().item(),
                }
            if use_wandb:
                wandb.log(log_dict)

            # torch.cuda.empty_cache()

        scheduler.step()
        
        # Save model
        if (epoch % save_interval) == 0:
            ckpt = {'epoch': epoch,
                    'transformer_encoder_state_dict': transformer_encoder.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': total_g_loss}
            torch.save(ckpt, os.path.join(wdir, f'train-{epoch}.pt'))
            print(f"[MODEL SAVED at {epoch} Epoch]")

    if use_wandb:
        wandb.run.finish()
    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='/viscam/u/jiamanli/output/headformer_runs/train', help='project/name')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')
    parser.add_argument('--wandb_pj_name', type=str, default='headformer_train', help='project name')
    parser.add_argument('--entity', default="", help='W&B entity')

    parser.add_argument('--data_root_folder', default='', help='folder to data')

    parser.add_argument('--workers', type=int, default=8, help='the number of workers for data loading')
    
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--validation_iter', type=int, default=1, help='validation iter')
    parser.add_argument('--image_save_iter', type=int, default=1, help='image save iter')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=50, help='Log model after every "save_period" epoch')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='generator_learning_rate')

    parser.add_argument('--window', type=int, default=90, help='horizon')

    parser.add_argument('--w_trans', type=float, default=1.0, help='loss_pos_weight')
    parser.add_argument('--w_rotation', type=float, default=1.0, help='loss_rot_weight')
    parser.add_argument('--w_vl', type=float, default=1.0, help='loss_vl_weight')
    parser.add_argument('--w_va', type=float, default=1.0, help='loss_va_weight')
    parser.add_argument('--w_dist', type=float, default=1.0, help='loss_distance_weight')

    parser.add_argument('--dist_scale', type=float, default=10.0, help='scale for prediction of distance scalar')

    parser.add_argument("--freeze_of_cnn", action="store_true")

    parser.add_argument("--train_on_ares", action="store_true")

    parser.add_argument("--train_on_gimo", action="store_true")

    parser.add_argument("--input_of_feats", action="store_true")

    parser.add_argument('--n_dec_layers', type=int, default=2, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=256, help='the dimension of intermediate representation in transformer')
    
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(Path(opt.project) / opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt, device)
    