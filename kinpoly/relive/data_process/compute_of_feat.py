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
import glob
import pdb
import os.path as osp
import yaml
sys.path.append(os.getcwd())

import cv2
import torch.utils.data as data
import torch
import joblib
from tqdm import tqdm
sys.path.append(os.getcwd())
import torch.nn.functional as F
from collections import defaultdict

from relive.utils import *
from relive.models.resnet_traj import ResNet
from relive.data_loaders.of_dataset import OFDatatset
from relive.utils.statear_smpl_config import Config


def load_of(of_files, augment = False):
    ofs = []
    for of_file in of_files:
        of_i = np.load(of_file)
        ofs.append(of_i)
    ofs = np.stack(ofs)
    
    return ofs

if __name__ == "__main__":
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--data', default=None)
    parser.add_argument('--iter', default=0, type=int)
    parser.add_argument('--out', default="resnet_traj")
    parser.add_argument('--wild', action='store_true', default=False)
    args = parser.parse_args()
    
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()    
        else torch.device("cpu")
    )

    if args.data is None:
        args.data = args.mode if args.mode in {'train', 'test'} else 'train'

    cfg = Config("all", args.cfg,  create_dirs=False, mujoco_path = "%s.xml")
    os.makedirs(f'results/{args.out}', exist_ok=True)
    
    resnet_traj = ResNet(6, fix_params= False, running_stats="False", pretrained=True)

    if args.iter > 0:
        cp_path = f'results/{args.out}/iter_{args.iter:04d}.p' 
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        resnet_traj.load_state_dict(model_cp, strict=True)

    num_epoch = 300
    optimizer = torch.optim.Adam(resnet_traj.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)    
    dataset = OFDatatset(cfg, data_mode = "all") # such that augmentation can happend every time
    scheduler = get_scheduler(
        optimizer,
        policy="lambda",
        nepoch_fix=10,
        nepoch=num_epoch,
    )
    

    if args.mode == "train":
        resnet_traj.train()
        resnet_traj.to(device)
        data_out = {}
        for i_epoch in range(args.iter, num_epoch):
            generator = dataset.sampling_generator(batch_size=768)
            epoch_loss = 0
            pbar = tqdm(generator)
            for data_sample in pbar:
                data_sample = {k: v.to(device) for k, v in data_sample.items()}
                of_data = data_sample['of_data']
                of_data = torch.cat((of_data, torch.zeros(of_data.shape[:-1] + (1,), device=device)), dim=-1)
                target_headvel = data_sample['head_vels']
                take_output = resnet_traj(of_data.permute(0, 3, 1, 2))
                loss = torch.square(take_output - target_headvel).sum(dim = 1).mean()

                optimizer.zero_grad()
                loss.backward()   
                optimizer.step()  
                

                pbar.set_description(f"loss: {loss.cpu().detach().numpy():.3f} lr: {scheduler.get_lr()[0]}")
                epoch_loss += loss.cpu().item()
            scheduler.step()
            
            epoch_loss /= dataset.len
            print(f"epoch {i_epoch:4d}   epoch_loss: {epoch_loss:.4f}")
            if  (i_epoch + 1) % 10 == 0:
                with to_cpu(resnet_traj):
                    # cp_path = '%s/iter_%04d.p' % ("results/resnet_traj", i_epoch + 1)
                    cp_path = f'results/{args.out}/iter_{(i_epoch + 1):04d}.p' 
                    pickle.dump(resnet_traj.state_dict(), open(cp_path, 'wb'))
                    
    elif args.mode == "test":
        data_dir = "/insert_directory_here/"
        if args.wild:
            data_file = "traj_wild_smpl"
        else:
            data_file = "expert_smpl_all_all"
        data_file = osp.join(data_dir, "features", f"{data_file}.p")
        print(f"Loading data: {data_file}")
        all_expert_data = joblib.load(data_file)
        data_out = {}
        resnet_traj.to(device)
        resnet_traj.eval()
        pbar = tqdm(all_expert_data.items())
        for take, take_data in pbar:
            # pbar.set_description(take)
            
            of_data = torch.from_numpy(load_of(take_data['of_files'])).to(device)
            head_vels = torch.from_numpy(take_data['head_vels']).to(device)
            of_data = torch.cat((of_data, torch.zeros(of_data.shape[:-1] + (1,), device=device)), dim=-1)
            with torch.no_grad():
                if of_data.shape[0] > 256:
                    splits = torch.split(of_data, 256,dim = 0)
                    splits_acc = []
                    for split in splits:
                        take_output = resnet_traj.get_embedding(split.permute(0, 3, 1, 2))
                        splits_acc.append(take_output)
                    take_output = torch.cat(splits_acc, dim = 0)
                else:
                    take_output = resnet_traj.get_embedding(of_data.permute(0, 3, 1, 2))
                    # take_output = resnet_traj(of_data.permute(0, 3, 1, 2))
                
                # head_vels = torch.from_numpy(take_data['head_vels']).to(device)
                # take_output = resnet_traj(of_data.permute(0, 3, 1, 2))
                # loss = torch.square(take_output - head_vels).sum(dim = 1).mean()
                # pbar.set_description(f"loss: {loss.cpu().detach().numpy():.3f}")



            data_out[take] = take_output.cpu().numpy().squeeze()

        # out_dir = osp.join(data_dir, "features", f"of_feat_smpl_all_all_aug.p")
        # out_dir = osp.join(data_dir, "features", f"of_feat_smpl_wild_mocap.p")
        # out_dir = osp.join(data_dir, "features", f"of_feat_wild_all_all_aug.p")
        out_dir = osp.join(data_dir, "features", f"of_feat_smpl_wild_mocap_wild.p")
        # out_dir = osp.join(data_dir, "features", f"of_feat_smpl_all_all.p")
        # out_dir = osp.join(data_dir, "features", f"of_feat_smpl_all_aug.p")
        # out_dir = osp.join(data_dir, "features", f"of_feat_wild_all_aug.p")
        # out_dir = osp.join(data_dir, "features", f"of_feat_wild_all.p")
        # out_dir = osp.join(data_dir, "features", f"of_feat_wild_all_all.p")
        print(out_dir)
        joblib.dump(data_out, out_dir)