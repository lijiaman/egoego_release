import argparse
import os
from pathlib import Path
import yaml

import wandb

import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import pytorch3d.transforms as transforms 

from ema_pytorch import EMA

from egoego.data.amass_diffusion_dataset import AMASSDataset, quat_ik_torch, run_smpl_model 

from egoego.model.transformer_cond_diffusion_model import CondGaussianDiffusion

from egoego.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file
from egoego.vis.pose import show3Dpose_animation_smpl22

from egoego.lafan1.utils import rotate_at_frame_smplh

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay = 0.995,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 10000000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        ema_update_every = 10,
        save_and_sample_every = 200000,
        results_folder = './results',
        use_wandb=True,
        run_demo=False,
    ):
        super().__init__()

        self.use_wandb = use_wandb           
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, name=opt.exp_name, dir=opt.save_dir)

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt 

        if run_demo:
            self.ds = AMASSDataset(self.opt, train=False, window=opt.window, run_demo=True) 
        else:
            self.prep_dataloader(window_size=opt.window)

        self.window = opt.window 

        self.bm_dict = self.ds.bm_dict 

    def prep_dataloader(self, window_size):
        # Define dataset
        train_dataset = AMASSDataset(self.opt, train=True, window=window_size)
        val_dataset = AMASSDataset(self.opt, train=False, window=window_size)

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=0))

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

    def load(self, milestone):
        data = torch.load(os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def load_weight_path(self, weight_path):
        data = torch.load(weight_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def train(self):
        init_step = self.step 
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()

            nan_exists = False # If met nan in loss or gradient, need to skip to next data. 
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)

                data = data_dict['motion'].cuda()

                padding_mask = self.prep_padding_mask(data, data_dict['seq_len'])

                with autocast(enabled = self.amp):
                   
                    cond_mask = self.prep_head_condition_mask(data) # BS X T X D 
                    loss_diffusion = self.model(data, cond_mask, padding_mask=padding_mask)
                    
                    loss = loss_diffusion

                    if torch.isnan(loss).item():
                        print('WARNING: NaN loss. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()
                  
                    # check gradients
                    parameters = [p for p in self.model.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(data.device) for p in parameters]), 2.0)
                    if torch.isnan(total_norm):
                        print('WARNING: NaN gradients. Skipping to next data...')
                        nan_exists = True 

                        torch.cuda.empty_cache()
                        continue

                    if self.use_wandb:
                        log_dict = {
                            "Train/Loss/Total Loss": loss.item(),
                            "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                        }
                        wandb.log(log_dict)

                    if idx % 50 == 0 and i == 0:
                        print("Step: {0}".format(idx))
                        print("Loss: %.4f" % (loss.item()))

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.ema.ema_model.eval()
                with torch.no_grad():
                    milestone = self.step // self.save_and_sample_every
                  
                    val_data_dict = next(self.val_dl) 
                    val_data = val_data_dict['motion'].cuda() 
                    cond_mask = self.prep_head_condition_mask(val_data) # BS X T X D 

                    padding_mask = self.prep_padding_mask(val_data, val_data_dict['seq_len'])

                    all_res_list = self.ema.ema_model.sample(val_data, cond_mask, padding_mask=padding_mask)
                    
                self.save(milestone)

                # Visualization
                bs_for_vis = 4
                for_vis_gt_data = val_data[:bs_for_vis]
                self.gen_vis_res(for_vis_gt_data, self.step, vis_gt=True)

                self.gen_vis_res(all_res_list[:bs_for_vis], self.step)

            self.step += 1

        print('training complete')

        if self.use_wandb:
            wandb.run.finish()
    
    def prep_head_condition_mask(self, data, joint_idx=15):
        # data: BS X T X D 
        # head_idx = 15 
        # Condition part is zeros, while missing part is ones. 
        mask = torch.ones_like(data).to(data.device)

        cond_pos_dim_idx = joint_idx * 3 
        cond_rot_dim_idx = 22 * 3 + joint_idx * 6
        mask[:, :, cond_pos_dim_idx:cond_pos_dim_idx+3] = torch.zeros(data.shape[0], data.shape[1], 3).to(data.device)
        mask[:, :, cond_rot_dim_idx:cond_rot_dim_idx+6] = torch.zeros(data.shape[0], data.shape[1], 6).to(data.device)

        return mask 

    def prep_padding_mask(self, val_data, seq_len):
        # Generate padding mask 
        actual_seq_len = seq_len + 1 # BS, + 1 since we need additional timestep for noise level 
        tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
        self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
        # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :].to(val_data.device)

        return padding_mask 

    def cond_sample_res(self):
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
        self.load(milestone)
        self.ema.ema_model.eval()
        num_sample = 4
        with torch.no_grad():
            for s_idx in range(num_sample):
                val_data_dict = next(self.val_dl)
                val_data = val_data_dict['motion'].cuda() 
                cond_mask = self.prep_head_condition_mask(val_data) # BS X T X D 

                padding_mask = self.prep_padding_mask(val_data, val_data_dict['seq_len'])
                
                all_res_list = self.ema.ema_model.sample(x_start=val_data, cond_mask=cond_mask, padding_mask=padding_mask)
                
                vis_tag = "test_head_cond_sample_"+str(s_idx)
                   
                max_num = 1
                self.gen_vis_res(val_data[:max_num], vis_tag, vis_gt=True)
                self.gen_vis_res(all_res_list[:max_num], vis_tag)

    def full_body_gen_cond_head_pose_sliding_window(self, head_pose, seq_name):
        # head_pose: BS X T X 7 
        self.ema.ema_model.eval()

        global_head_jpos = head_pose[:, :, :3] # BS X T X 3 
        global_head_quat = head_pose[:, :, 3:] # BS X T X 4 

        data = torch.zeros(head_pose.shape[0], head_pose.shape[1], 22*3+22*6).to(head_pose.device) # BS X T X D 

        with torch.no_grad():
            cond_mask = self.prep_head_condition_mask(data) # BS X T X D 

            local_aa_rep, seq_root_pos = self.ema.ema_model.sample_sliding_window_w_canonical(self.ds, \
            global_head_jpos, global_head_quat, x_start=data, cond_mask=cond_mask) 
            # BS X T X 22 X 3, BS X T X 3       

        return local_aa_rep, seq_root_pos # T X 22 X 3, T X 3  

    def gen_vis_res(self, all_res_list, step, vis_gt=False):
        # all_res_list: N X T X D 
        num_seq = all_res_list.shape[0]
        normalized_global_jpos = all_res_list[:, :, :22*3].reshape(num_seq, -1, 22, 3)

        global_jpos = self.ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, 22, 3))
      
        global_jpos = global_jpos.reshape(num_seq, -1, 22, 3) # N X T X 22 X 3 
        global_root_jpos = global_jpos[:, :, 0, :].clone() # N X T X 3 

        global_rot_6d = all_res_list[:, :, 22*3:].reshape(num_seq, -1, 22, 6)
        
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # N X T X 22 X 3 X 3 

        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx] # T X 22 X 3 X 3 
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat) # T X 22 X 3 X 3 
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 22 X 3 
            
            curr_global_root_jpos = global_root_jpos[idx] # T X 3
            move_xy_trans = curr_global_root_jpos.clone()[0:1] # 1 X 3 
            move_xy_trans[:, 2] = 0 
            root_trans = curr_global_root_jpos - move_xy_trans # T X 3 

            # Generate global joint position 
            bs = 1
            betas = torch.zeros(bs, 16).to(root_trans.device)
            gender = ["male"] * bs 

            mesh_jnts, mesh_verts, mesh_faces = \
            run_smpl_model(root_trans[None], \
            curr_local_rot_aa_rep[None], betas, gender, \
            self.bm_dict)
            # BS(1) X T' X 22 X 3, BS(1) X T' X Nv X 3
            
            dest_mesh_vis_folder = os.path.join(self.vis_folder, "blender_mesh_vis", str(step))
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)

            if vis_gt:
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt.mp4")
            else:
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+".mp4")

            # Visualize the skeleton 
            if vis_gt:
                dest_skeleton_vis_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_skeleton_gt.gif")
            else:
                dest_skeleton_vis_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_skeleton.gif")

            channels = global_jpos[idx:idx+1] # 1 X T X 22 X 3 
            # show3Dpose_animation_smpl22(channels.data.cpu().numpy(), dest_skeleton_vis_path) 

            # For visualizing human mesh only 
            save_verts_faces_to_mesh_file(mesh_verts.data.cpu().numpy()[0], mesh_faces.data.cpu().numpy(), mesh_save_folder)
            run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, out_vid_file_path)

    def gen_full_body_vis(self, root_trans, curr_local_rot_aa_rep, dest_mesh_vis_folder, seq_name, vis_gt=False):
        # root_trans: T X 3 
        # curr_local_rot_aa_rep: T X 22 X 3 

        # Generate global joint position 
        bs = 1
        betas = torch.zeros(bs, 16).to(root_trans.device)
        gender = ["male"] * bs 

        mesh_jnts, mesh_verts, mesh_faces = run_smpl_model(root_trans[None].float(), \
        curr_local_rot_aa_rep[None].float(), betas.float(), gender, self.ds.bm_dict)
        # BS(1) X T' X 22 X 3, BS(1) X T' X Nv X 3
    
        if vis_gt:
            mesh_save_folder = os.path.join(dest_mesh_vis_folder, seq_name, \
                            "objs_gt")
            out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, seq_name, \
                            "imgs_gt")
            out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                            seq_name+"_vid_gt.mp4")
        else:
            mesh_save_folder = os.path.join(dest_mesh_vis_folder, seq_name, \
                            "objs")
            out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, seq_name, \
                            "imgs")
            out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                            seq_name+"_vid.mp4")

        # For visualizing human mesh only 
        save_verts_faces_to_mesh_file(mesh_verts.data.cpu().numpy()[0], \
        mesh_faces.data.cpu().numpy(), mesh_save_folder)
        run_blender_rendering_and_save2video(mesh_save_folder, \
        out_rendered_img_folder, out_vid_file_path)

        return mesh_jnts, mesh_verts 

def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model  
    repr_dim = 22 * 3 + 22 * 6 
  
    loss_type = "l1"
    
    diffusion_model = CondGaussianDiffusion(d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                batch_size=opt.batch_size)
  
    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=8000000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
    )

    trainer.train()

    torch.cuda.empty_cache()

def run_sample(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # Define model     
    repr_dim = 22 * 3 + 22 * 6 
   
    loss_type = "l1"
  
    diffusion_model = CondGaussianDiffusion(d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                batch_size=opt.batch_size)

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=8000000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False 
    )
  
    trainer.cond_sample_res()

    torch.cuda.empty_cache()

def get_trainer(opt, run_demo=False):
    opt.window = opt.diffusion_window 

    opt.diffusion_save_dir = os.path.join(opt.diffusion_project, opt.diffusion_exp_name)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # Prepare Directories
    save_dir = Path(opt.diffusion_save_dir)
    wdir = save_dir / 'weights'

    # Define model 
    repr_dim = 22 * 3 + 22 * 6 
   
    transformer_diffusion = CondGaussianDiffusion(d_feats=repr_dim, d_model=opt.diffusion_d_model, \
                n_dec_layers=opt.diffusion_n_dec_layers, n_head=opt.diffusion_n_head, \
                d_k=opt.diffusion_d_k, d_v=opt.diffusion_d_v, \
                max_timesteps=opt.diffusion_window+1, out_dim=repr_dim, timesteps=1000, objective="pred_x0", \
                batch_size=opt.diffusion_batch_size)

    transformer_diffusion.to(device)

    trainer = Trainer(
        opt,
        transformer_diffusion,
        train_batch_size=opt.diffusion_batch_size, # 32
        train_lr=opt.diffusion_learning_rate, # 1e-4
        train_num_steps=8000000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False,
        run_demo=run_demo,
    )

    return trainer 

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--wandb_pj_name', type=str, default='', help='project name')
    parser.add_argument('--entity', default='', help='W&B entity')
    parser.add_argument('--exp_name', default='', help='save to project/name')
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--data_root_folder', default='', help='')

    parser.add_argument('--window', type=int, default=120, help='horizon')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")
   
    # For data representation
    parser.add_argument("--canonicalize_init_head", action="store_true")
    parser.add_argument("--use_min_max", action="store_true")

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    else:
        run_train(opt, device)
