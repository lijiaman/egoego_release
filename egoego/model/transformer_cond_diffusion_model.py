import torch
from torch import nn

from egoego.model.transformer_module import Decoder 

import os 
import math 

from tqdm.auto import tqdm

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from inspect import isfunction

import torch.nn.functional as F

import pytorch3d.transforms as transforms 

from egoego.vis.mesh_motion import get_mesh_verts_faces_for_human_only 

from body_model.body_model import BodyModel

from egoego.data.amass_diffusion_dataset import quat_ik_torch 

from egoego.lafan1.utils import rotate_at_frame_smplh
 
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
        
class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_feats,
        d_model,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        max_timesteps,
    ):
        super().__init__()
        
        self.d_feats = d_feats 
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.max_timesteps = max_timesteps 

        # Input: BS X D X T 
        # Output: BS X T X D'
        self.motion_transformer = Decoder(d_feats=self.d_feats*2, d_model=self.d_model, \
            n_layers=self.n_dec_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True)  

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

        # For noise level t embedding
        dim = 64
        time_dim = dim * 4

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )

    def forward(self, src, noise_t, padding_mask=None):
        # src: BS X T X D
        # noise_t: int 
       
        noise_t_embed = self.time_mlp(noise_t) # BS X d_model 
        noise_t_embed = noise_t_embed[:, None, :] # BS X 1 X d_model 

        bs = src.shape[0]
        num_steps = src.shape[1] + 1

        if padding_mask is None:
            # In training, no need for masking 
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool() # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps)+1 # timesteps
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1) # BS X 1 X timesteps

        data_input = src.transpose(1, 2).detach() # BS X D X T 
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, obj_embedding=noise_t_embed)
    
        output = self.linear_out(feat_pred[:, 1:]) # BS X T X D

        return output # predicted noise, the same size as the input 

class CondGaussianDiffusion(nn.Module):
    def __init__(
        self,
        d_feats,
        d_model,
        n_head,
        n_dec_layers,
        d_k,
        d_v,
        max_timesteps,
        out_dim,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        batch_size=None,
    ):
        super().__init__()

        self.denoise_fn = TransformerDiffusionModel(d_feats=d_feats, d_model=d_model, n_head=n_head, \
                    d_k=d_k, d_v=d_v, n_dec_layers=n_dec_layers, max_timesteps=max_timesteps) 
        # Input condition and noisy motion, noise level t, predict gt motion
        
        self.objective = objective

        self.seq_len = max_timesteps - 1 
        self.out_dim = out_dim 

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_cond, clip_denoised, padding_mask=None):
        x_all = torch.cat((x, x_cond), dim=-1)
        model_output = self.denoise_fn(x_all, t, padding_mask=padding_mask)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, x_cond, clip_denoised=True, padding_mask=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_cond=x_cond, \
            clip_denoised=clip_denoised, padding_mask=padding_mask)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_start, cond_mask, padding_mask=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)
        x_cond = x_start * (1. - cond_mask) + \
            cond_mask * torch.randn_like(x_start).to(x_start.device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), x_cond, padding_mask=padding_mask)     

        return x # BS X T X D

    @torch.no_grad()
    def p_sample_loop_sliding_window(self, shape, x_start, cond_mask):
        device = self.betas.device

        b = shape[0]
        assert b == 1
        
        x_all = torch.randn(shape, device=device)
        x_cond_all = x_start * (1. - cond_mask) + \
            cond_mask * torch.randn_like(x_start).to(x_start.device)

        x_blocks = []
        x_cond_blocks = []
        # Divide to blocks to form a batch, then just need run model once to get all the results. 
        num_steps = x_start.shape[1]
        stride = self.window // 2
        for t_idx in range(0, num_steps, stride):
            x = x_all[0, t_idx:t_idx+self.window]
            x_cond = x_cond_all[0, t_idx:t_idx+self.window]

            x_blocks.append(x) # T X D 
            x_cond.append(x_cond) # T X D 

        last_window_x = None 
        last_window_cond = None 
        if x_blocks[-1].shape[0] != x_blocks[0].shape[0]:
            last_window_x = x_blocks[-1][None] # 1 X T X D 
            last_window_cond = x_cond_blocks[-1][None] 

            x_blocks = torch.stack(x_blocks[:-1]) # K X T X D 
            x_cond_blocks = torch.stack(x_cond_blocks[:-1]) # K X T X D 
        else:
            x_blocks = torch.stack(x_blocks) # K X T X D 
            x_cond_blocks = torch.stack(x_cond_blocks) # K X T X D 
       
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x_blocks = self.p_sample(x_blocks, torch.full((b,), i, device=device, dtype=torch.long), x_cond_blocks)    

        if last_window_x is not None:
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                last_window_x = self.p_sample(last_window_x, torch.full((b,), i, device=device, dtype=torch.long), last_window_cond)    

        # Convert from K X T X D to a single sequence.
        seq_res = None  
        # for t_idx in range(0, num_steps, stride):
        num_windows = x_blocks.shape[0]
        for w_idx in range(num_windows):
            if w_idx == 0:
                seq_res = x_blocks[w_idx] # T X D 
            else:
                seq_res = torch.cat((seq_res, x_blocks[self.window-stride:]), dim=0)

        if last_window_x is not None:
            seq_res = torch.cat((seq_res, last_window_x[self.window-stride:]), dim=0)

        return seq_res # BS X T X D

    @torch.no_grad()
    def p_sample_loop_sliding_window_w_canonical(self, ds, shape, global_head_jpos, global_head_jquat, cond_mask):
        # shape: BS X T X D 
        # global_head_jpos: BS X T X 3 
        # global_head_jquat: BS X T X 4 
        # cond_mask: BS X T X D 

        device = self.betas.device

        b = shape[0]
        # assert b == 1
        
        x_all = torch.randn(shape, device=device)

        whole_seq_aa_rep = None 
        whole_seq_root_pos = None 
        whole_seq_head_pos = None 

        # Divide to blocks to form a batch, then just need run model once to get all the results. 
        num_steps = global_head_jpos.shape[1]
        # stride = self.seq_len // 2
        overlap_frame_num = 10
        stride = self.seq_len - overlap_frame_num 
        for t_idx in range(0, num_steps, stride):
            curr_x = x_all[:, t_idx:t_idx+self.seq_len]

            if curr_x.shape[1] <= self.seq_len - stride:
                break 

            # Canonicalize current window 
            curr_global_head_quat = global_head_jquat[:, t_idx:t_idx+self.seq_len] # BS X T X 4
            curr_global_head_jpos = global_head_jpos[:, t_idx:t_idx+self.seq_len] # BS X T X 3 

            aligned_head_trans, aligned_head_quat, recover_rot_quat = \
            rotate_at_frame_smplh(curr_global_head_jpos.data.cpu().numpy(), \
            curr_global_head_quat.data.cpu().numpy(), cano_t_idx=0)
            # BS X T' X 3, BS X T' X 4, BS X 1 X 1 X 4  

            aligned_head_trans = torch.from_numpy(aligned_head_trans).to(global_head_jpos.device)
            aligned_head_quat = torch.from_numpy(aligned_head_quat).to(global_head_jpos.device)

            move_to_zero_trans = aligned_head_trans[:, 0:1, :].clone() # Move the head joint x, y to 0,  BS X 1 X 3
            move_to_zero_trans[:, :, 2] = 0 

            aligned_head_trans = aligned_head_trans - move_to_zero_trans # BS X T X 3 

            aligned_head_rot_mat = transforms.quaternion_to_matrix(aligned_head_quat) # BS X T X 3 X 3 
            aligned_head_rot_6d = transforms.matrix_to_rotation_6d(aligned_head_rot_mat) # BS X T X 6  

            head_idx = 15 
            curr_x_start = torch.zeros(aligned_head_rot_6d.shape[0], \
            aligned_head_rot_6d.shape[1], 22*3+22*6).to(aligned_head_rot_6d.device)
            curr_x_start[:, :, head_idx*3:head_idx*3+3] = aligned_head_trans # BS X T X 3
            curr_x_start[:, :, 22*3+head_idx*6:22*3+head_idx*6+6] = aligned_head_rot_6d # BS X T X 6 

            # Normalize data to [-1, 1]
            normalized_jpos = ds.normalize_jpos_min_max(curr_x_start[:, :, :22*3].reshape(-1, 22, 3))
            curr_x_start[:, :, :22*3] = normalized_jpos.reshape(b, -1, 22*3) # BS X T X (22*3)

            curr_cond_mask = cond_mask[:, t_idx:t_idx+self.seq_len] # BS X T X D 
            curr_x_cond = curr_x_start * (1. - curr_cond_mask) + \
            curr_cond_mask * torch.randn_like(curr_x_start).to(curr_x_start.device)
       
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                curr_x = self.p_sample(curr_x, torch.full((b,), i, device=device, dtype=torch.long), curr_x_cond)    
                # Apply previous window prediction as additional condition, direcly replacement. 
                if t_idx > 0:
                    curr_x[:, :self.seq_len-stride, 22*3:] = prev_res_rot_6d.reshape(b, -1, 22*6)
                    curr_x[:, :self.seq_len-stride, :22*3] = prev_res_jpos.reshape(b, -1, 22*3)

            curr_seq_local_aa_rep, curr_seq_root_pos, curr_seq_head_pos = \
            self.convert_model_res_to_data(ds, curr_x, \
            recover_rot_quat, curr_global_head_jpos) 
            
            if t_idx == 0:
                whole_seq_aa_rep = curr_seq_local_aa_rep # BS X T X 22 X 3
                whole_seq_root_pos = curr_seq_root_pos # BS X T X 3 
                whole_seq_head_pos = curr_seq_head_pos # BS X T X 3 
            else:
                prev_last_pos = whole_seq_head_pos[:, -1:, :].clone() # BS X 1 X 3 
                curr_first_pos = curr_seq_head_pos[:, self.seq_len-stride-1:self.seq_len-stride, :].clone() # BS X 1 X 3
                
                move_trans = prev_last_pos - curr_first_pos # BS X 1 X 3 
                curr_seq_root_pos += move_trans # BS X T X 3 
                curr_seq_head_pos += move_trans 
                
                whole_seq_aa_rep = torch.cat((whole_seq_aa_rep, \
                curr_seq_local_aa_rep[:, self.seq_len-stride:]), dim=1)
                whole_seq_root_pos = torch.cat((whole_seq_root_pos, \
                curr_seq_root_pos[:, self.seq_len-stride:]), dim=1)
                whole_seq_head_pos = torch.cat((whole_seq_head_pos, \
                curr_seq_head_pos[:, self.seq_len-stride:]), dim=1)

            # Convert results to normalized representation for sampling in next window
            tmp_global_rot_quat, tmp_global_jpos = ds.fk_smpl(curr_seq_root_pos.reshape(-1, 3), \
            curr_seq_local_aa_rep.reshape(-1, 22, 3)) 
            # (BS*T) X 22 X 4, (BS*T) X 22 X 3 
            tmp_global_rot_quat = tmp_global_rot_quat.reshape(b, -1, 22, 4)
            tmp_global_jpos = tmp_global_jpos.reshape(b, -1, 22, 3)

            tmp_global_rot_quat = tmp_global_rot_quat[:, -self.seq_len+stride:].clone()
            tmp_global_jpos = tmp_global_jpos[:, -self.seq_len+stride:].clone()
            
            tmp_global_head_quat = tmp_global_rot_quat[:, :, 15, :] # BS X T X 4 
            tmp_global_head_jpos = tmp_global_jpos[:, :, 15, :] # BS X T X 3 

            tmp_aligned_head_trans, tmp_aligned_head_quat, tmp_recover_rot_quat = \
            rotate_at_frame_smplh(tmp_global_head_jpos.data.cpu().numpy(), \
            tmp_global_head_quat.data.cpu().numpy(), cano_t_idx=0)
            # BS X T' X 3, BS X T' X 4, BS X 1 X 1 X 4  

            tmp_aligned_head_trans = torch.from_numpy(tmp_aligned_head_trans).to(tmp_global_head_jpos.device)

            tmp_move_to_zero_trans = tmp_aligned_head_trans[:, 0:1, :].clone() 
            # Move the head joint x, y to 0,  BS X 1 X 3
            tmp_move_to_zero_trans[:, :, 2] *= 0 # 1 X 1 X 3 

            tmp_aligned_head_trans = tmp_aligned_head_trans - tmp_move_to_zero_trans # BS X T X 3 

            tmp_recover_rot_quat = torch.from_numpy(tmp_recover_rot_quat).float().to(tmp_global_rot_quat.device)

            tmp_global_jpos = transforms.quaternion_apply(transforms.quaternion_invert(\
            tmp_recover_rot_quat).repeat(1, tmp_global_jpos.shape[1], \
            tmp_global_jpos.shape[2], 1), tmp_global_jpos) # BS X T X 22 X 3

            tmp_global_jpos -= tmp_move_to_zero_trans[:, :, None, :] 

            prev_res_jpos = tmp_global_jpos.clone() 
            prev_res_jpos = ds.normalize_jpos_min_max(prev_res_jpos.reshape(-1, 22, 3)).reshape(b, -1, 22, 3) # BS X T X 22 X 3 

            prev_res_global_rot_quat = transforms.quaternion_multiply(transforms.quaternion_invert(\
            tmp_recover_rot_quat).repeat(1, tmp_global_rot_quat.shape[1], \
            tmp_global_rot_quat.shape[2], 1), \
            tmp_global_rot_quat) # BS X T X 22 X 4
            prev_res_rot_mat = transforms.quaternion_to_matrix(prev_res_global_rot_quat) # BS X T X 22 X 3 X 3 
            prev_res_rot_6d = transforms.matrix_to_rotation_6d(prev_res_rot_mat) # BS X T X 22 X 6 

        return whole_seq_aa_rep, whole_seq_root_pos
        # T X 22 X 3, T X 3 

    def convert_model_res_to_data(self, ds, all_res_list, recover_rot_quat, curr_global_head_jpos):
        # all_res_list: BS X T X D 
        # recover_rot_quat: BS X 1 X 1 X 4 
        # curr_global_head_jpos: BS X T X 3 

        # De-normalize jpos 
        use_global_head_pos_for_root_trans = False 

        bs = all_res_list.shape[0]
        normalized_global_jpos = all_res_list[:, :, :22*3].reshape(bs, -1, 22, 3) # BS X T X 22 X 3 
      
        global_jpos = ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, 22, 3)) # (BS*T) X 22 X 3
        global_jpos = global_jpos.reshape(bs, -1, 22, 3) # BS X T X 22 X 3 

        global_rot_6d = all_res_list[:, :, 22*3:] # BS X T X (22*6)
        
        bs, num_steps, _, _ = global_jpos.shape
        global_rot_6d = global_rot_6d.reshape(bs, num_steps, 22, 6) # BS X T X 22 X 6 
        
        global_root_jpos = global_jpos[:, :, 0, :] # BS X T X 3 

        head_idx = 15 
        global_head_jpos = global_jpos[:, :, head_idx, :] # BS X T X 3 

        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # BS X T X 22 X 3 X 3
        global_quat = transforms.matrix_to_quaternion(global_rot_mat) # BS X T X 22 X 4 
        recover_rot_quat = torch.from_numpy(recover_rot_quat).to(global_quat.device) # BS X 1 X 1 X 4 
        ori_global_quat = transforms.quaternion_multiply(recover_rot_quat, global_quat) # BS X T X 22 X 4 
        ori_global_root_jpos = global_root_jpos # BS X T X 3 
        ori_global_root_jpos = transforms.quaternion_apply(recover_rot_quat.squeeze(1).repeat(1, num_steps, 1), \
                        ori_global_root_jpos) # BS X T X 3 

        ori_global_head_jpos = transforms.quaternion_apply(recover_rot_quat.squeeze(1).repeat(1, num_steps, 1), \
                        global_head_jpos) # BS X T X 3 

        # Convert global join rotation to local joint rotation
        ori_global_rot_mat = transforms.quaternion_to_matrix(ori_global_quat) # BS X T X 22 X 3 X 3
        ori_local_rot_mat = quat_ik_torch(ori_global_rot_mat.reshape(-1, 22, 3, 3)).reshape(bs, -1, 22, 3, 3) # BS X T X 22 X 3 X 3 
        ori_local_aa_rep = transforms.matrix_to_axis_angle(ori_local_rot_mat) # BS X T X 22 X 3 

        if use_global_head_pos_for_root_trans: 
            zero_root_trans = torch.zeros(ori_local_aa_rep.shape[0], ori_local_aa_rep.shape[1], 3).to(ori_local_aa_rep.device).float()
            betas = torch.zeros(bs, 10).to(zero_root_trans.device).float()
            gender = ["male"] * bs 

            _, mesh_jnts = ds.fk_smpl(zero_root_trans.reshape(-1, 3), ori_local_aa_rep.reshape(-1, 22, 3))
            # (BS*T) X 22 X 4, (BS*T) X 22 X 3 
            mesh_jnts = mesh_jnts.reshape(bs, -1, 22, 3) # BS X T X 22 X 3 

            head_idx = 15 
            wo_root_trans_head_pos = mesh_jnts[:, :, head_idx, :] # BS X T X 3 

            calculated_root_trans = ori_global_head_jpos - wo_root_trans_head_pos # BS X T X 3 

            return ori_local_aa_rep, calculated_root_trans, ori_global_head_jpos

        return ori_local_aa_rep, ori_global_root_jpos, ori_global_head_jpos

    @torch.no_grad()
    def sample(self, x_start, cond_mask, padding_mask=None):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.denoise_fn.eval() 
        sample_res = self.p_sample_loop(x_start.shape, \
                x_start, cond_mask)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res  

    @torch.no_grad()
    def sample_sliding_window(self, x_start, cond_mask):
        # If the sequence is longer than trained max window, divide 
        self.denoise_fn.eval()
        sample_res = self.p_sample_loop_sliding_window(x_start.shape, \
                x_start, cond_mask)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res  

    @torch.no_grad()
    def sample_sliding_window_w_canonical(self, ds, global_head_jpos, global_head_jquat, x_start, cond_mask):
        # If the sequence is longer than trained max window, divide 
        self.denoise_fn.eval()
        sample_res = self.p_sample_loop_sliding_window_w_canonical(ds, x_start.shape, \
                global_head_jpos, global_head_jquat, cond_mask)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res  

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, cond_mask, t, noise=None, padding_mask=None):
        # x_start: BS X T X D
        # cond_mask: BS X T X D, missing regions are 1, head pose conditioned regions are 0.  
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise) # noisy motion in noise level t. 
       
        noisy_x_start = x_start.clone() 
        masked_x_input = x 
        x_cond = noisy_x_start * (1. - cond_mask) + cond_mask * torch.randn_like(noisy_x_start).to(noisy_x_start.device)
        
        x_all = torch.cat((masked_x_input, x_cond), dim=-1)
        model_out = self.denoise_fn(x_all, t, padding_mask)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # Predicting both head pose and other joints' pose. 
        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, reduction = 'none') * padding_mask[:, 0, 1:][:, :, None]
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none') # BS X T X D 
           
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        
        return loss.mean()

    def de_normalize_root_jpos(self, root_jpos):
        # root_jpos: BS X T X 3 
        normalized_jpos = (root_jpos + 1) * 0.5 # [0, 1] range
        root_jpos_min = self.global_jpos_min[:, 0:1, :] # 1 X 1 X 3 
        root_jpos_max = self.global_jpos_max[:, 0:1, :] # 1 X 1 X 3 
        de_jpos = normalized_jpos * (root_jpos_max.to(normalized_jpos.device)-\
        root_jpos_min.to(normalized_jpos.device)) + root_jpos_min.to(normalized_jpos.device)

        return de_jpos # BS X T X 3 

    def forward(self, x_start, cond_mask, padding_mask=None):
        # x_start: BS X T X D 
        # cond_mask: BS X T X D 
        bs = x_start.shape[0] 
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()
        # print("t:{0}".format(t))
        curr_loss = self.p_losses(x_start, cond_mask, t, padding_mask=padding_mask)

        return curr_loss
        