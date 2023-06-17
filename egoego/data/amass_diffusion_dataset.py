import os
import numpy as np
import joblib 

import torch
from torch.utils.data import Dataset
import pytorch3d.transforms as transforms 

from human_body_prior.body_model.body_model import BodyModel

from egoego.lafan1.utils import rotate_at_frame_smplh

SMPLH_PATH = "smpl_models/smplh_amass"

def run_smpl_model(root_trans, aa_rot_rep, betas, gender, bm_dict):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3 
    # betas: BS X 16
    # gender: BS 
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device) # BS X T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2) # BS X T X 52 X 3 

    aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3) # (BS*T) X n_joints X 3 
    betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs*num_steps, -1) # (BS*T) X 16 
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist() # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3  
    smpl_betas = betas # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :] # (BS*T) X 3 
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63) # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90) # (BS*T) X 90 

    B = smpl_trans.shape[0] # (BS*T) 

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female']
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int64)*-1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int64)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue
        
        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(pose_body=cur_pred_pose, pose_hand=cur_pred_pose_hand, \
                betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
        
        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]
        
    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 22 X 3 

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3 
    
    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3) # BS X T X 22 X 3/BS X T X 24 X 3  
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3 

    mesh_faces = pred_body.f 
    
    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces 

def get_smpl_parents():
    bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 
    parents = ori_kintree_table[0, :22] # 22 
    parents[0] = -1 # Assign -1 for the root joint's parent idx.

    return parents

def local2global_pose(local_pose):
    # local_pose: T X J X 3 X 3 
    kintree = get_smpl_parents() 

    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose # T X J X 3 X 3 

def quat_ik_torch(grot_mat):
    # grot: T X J X 3 X 3 
    parents = get_smpl_parents() 

    grot = transforms.matrix_to_quaternion(grot_mat) # T X J X 4 

    res = torch.cat(
            [
                grot[..., :1, :],
                transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), \
                grot[..., 1:, :]),
            ],
            dim=-2) # T X J X 4 

    res_mat = transforms.quaternion_to_matrix(res) # T X J X 3 X 3 

    return res_mat 

def quat_fk_torch(lrot_mat, lpos):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    parents = get_smpl_parents() 

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res

class AMASSDataset(Dataset):
    def __init__(
        self,
        opt,
        train,
        window=120,
        for_eval=False,
        run_demo=False, 
    ):
        self.opt = opt 

        self.train = train
        
        self.window = window

        # Prepare SMPLH model 
        surface_model_male_fname = os.path.join(SMPLH_PATH, "male", 'model.npz')
        surface_model_female_fname = os.path.join(SMPLH_PATH, "female", 'model.npz')
        dmpl_fname = None
        num_dmpls = None 
        num_expressions = None
        num_betas = 16 

        self.male_bm = BodyModel(bm_fname=surface_model_male_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname)
        self.female_bm = BodyModel(bm_fname=surface_model_female_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname)

        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False 

        self.male_bm = self.male_bm.cuda()
        self.female_bm = self.female_bm.cuda()
        
        self.bm_dict = {'male' : self.male_bm, 'female' : self.female_bm}

        self.rest_human_offsets = self.get_rest_pose_joints() # 1 X J X 3 

        data_root_folder = opt.data_root_folder 
        if self.train:
            self.data_path = os.path.join(data_root_folder, "amass_same_shape_egoego_processed", "train_amass_smplh_motion.p")
        else:
            self.data_path = os.path.join(data_root_folder, "amass_same_shape_egoego_processed", "test_amass_smplh_motion.p")
        
        if run_demo:
            self.data_path = os.path.join(data_root_folder, "demo_ares_data.p")

        ori_data_dict = joblib.load(self.data_path)
        # self.data_dict = self.filter_data(ori_data_dict)
        self.data_dict = ori_data_dict 

        if self.train:
            if self.opt.canonicalize_init_head:
                processed_data_path = os.path.join(data_root_folder, "cano_train_diffusion_amass_window_"+str(self.window)+".p")
            else:
                processed_data_path = os.path.join(data_root_folder, "train_diffusion_amass_window_"+str(self.window)+".p")
        else: 
            if self.opt.canonicalize_init_head:
                processed_data_path = os.path.join(data_root_folder, "cano_test_diffusion_amass_window_"+str(self.window)+".p")
            else:
                processed_data_path = os.path.join(data_root_folder, "test_diffusion_amass_window_"+str(self.window)+".p")

        if self.opt.canonicalize_init_head:
            min_max_mean_std_data_path = os.path.join(data_root_folder, "cano_min_max_mean_std_data_window_"+str(self.window)+".p")
        else:
            min_max_mean_std_data_path = os.path.join(data_root_folder, "min_max_mean_std_data_window_"+str(self.window)+".p")
        
        if run_demo:
            processed_data_path = os.path.join(data_root_folder, "demo_ares_data.p")

        if os.path.exists(processed_data_path):
            self.window_data_dict = joblib.load(processed_data_path)
        else:
            self.cal_normalize_data_input()
            joblib.dump(self.window_data_dict, processed_data_path)

        if self.train and not os.path.exists(min_max_mean_std_data_path):
            stats_dict = self.extract_min_max_mean_std_from_data()
            joblib.dump(stats_dict, min_max_mean_std_data_path)

        if os.path.exists(min_max_mean_std_data_path):
            min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)

            self.global_jpos_min = torch.from_numpy(min_max_mean_std_jpos_data['global_jpos_min']).float().reshape(22, 3)[None]
            self.global_jpos_max = torch.from_numpy(min_max_mean_std_jpos_data['global_jpos_max']).float().reshape(22, 3)[None]
            self.global_jvel_min = torch.from_numpy(min_max_mean_std_jpos_data['global_jvel_min']).float().reshape(22, 3)[None]
            self.global_jvel_max = torch.from_numpy(min_max_mean_std_jpos_data['global_jvel_max']).float().reshape(22, 3)[None]
       
        self.for_eval = for_eval

        if self.train:
            print("Total number of windows for training:{0}".format(len(self.window_data_dict)))
        else:
            print("Total number of windows for validation:{0}".format(len(self.window_data_dict)))

    def get_rest_pose_joints(self):
        zero_root_trans = torch.zeros(1, 1, 3).cuda().float()
        zero_rot_aa_rep = torch.zeros(1, 1, 22, 3).cuda().float()
        bs = 1 
        betas = torch.zeros(1, 16).cuda().float()
        gender = ["male"] * bs 

        rest_human_jnts, _, _ = \
        run_smpl_model(zero_root_trans, zero_rot_aa_rep, betas, gender, self.bm_dict)
        # 1 X 1 X J X 3 

        parents = get_smpl_parents()
        parents[0] = 0 # Make root joint's parent itself so that after deduction, the root offsets are 0
        rest_human_offsets = rest_human_jnts.squeeze(0) - rest_human_jnts.squeeze(0)[:, parents, :]

        return rest_human_offsets # 1 X J X 3 

    def fk_smpl(self, root_trans, lrot_aa):
        # root_trans: N X 3 
        # lrot_aa: N X J X 3 

        # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
        # lpos: N X J X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
        
        parents = get_smpl_parents() 

        lrot_mat = transforms.axis_angle_to_matrix(lrot_aa) # N X J X 3 X 3 

        lrot = transforms.matrix_to_quaternion(lrot_mat)

        # Generate global joint position 
        lpos = self.rest_human_offsets.repeat(lrot_mat.shape[0], 1, 1) # T' X 22 X 3 

        gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
        for i in range(1, len(parents)):
            gp.append(
                transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
            )
            gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

        global_rot = torch.cat(gr, dim=-2) # T X 22 X 4 
        global_jpos = torch.cat(gp, dim=-2) # T X 22 X 3 

        global_jpos += root_trans[:, None, :] # T X 22 X 3

        return global_rot, global_jpos 

    def filter_data(self, ori_data_dict):
        new_cnt = 0
        new_data_dict = {}
        max_len = 0 
        for k in ori_data_dict:
            curr_data = ori_data_dict[k]
            seq_len = curr_data['head_qpos'].shape[0]
         
            if seq_len >= self.window:
                new_data_dict[new_cnt] = curr_data 
                new_cnt += 1 

            if seq_len > max_len:
                max_len = seq_len 

        print("The numer of sequences in original data:{0}".format(len(ori_data_dict)))
        print("After filtering, remaining sequences:{0}".format(len(new_data_dict)))
        print("Max length:{0}".format(max_len))

        return new_data_dict 

    def cal_normalize_data_input(self):
        self.window_data_dict = {}
        s_idx = 0 

        for index in self.data_dict:
            seq_name = self.data_dict[index]['seq_name']

            seq_root_trans = self.data_dict[index]['trans'] # T X 3 
            seq_root_orient = self.data_dict[index]['root_orient'] # T X 3 
            seq_pose_body = self.data_dict[index]['body_pose'].reshape(-1, 21, 3) # T X 21 X 3

            num_steps = seq_root_trans.shape[0]
            for start_t_idx in range(0, num_steps, self.window//2):
                end_t_idx = start_t_idx + self.window - 1
                if end_t_idx >= num_steps:
                    end_t_idx = num_steps 

                # Skip the segment that has a length < 30 
                if end_t_idx - start_t_idx < 30:
                    continue 

                self.window_data_dict[s_idx] = {}
                
                query = self.process_window_data(seq_root_trans, seq_root_orient, seq_pose_body, start_t_idx, end_t_idx)
                
                curr_global_jpos = query['global_jpos']
                curr_global_jvel = query['global_jvel']
                curr_global_rot_6d = query['global_rot_6d']

                self.window_data_dict[s_idx]['seq_name'] = seq_name
                self.window_data_dict[s_idx]['start_t_idx'] = start_t_idx
                self.window_data_dict[s_idx]['end_t_idx'] = end_t_idx 

                self.window_data_dict[s_idx]['global_jpos'] = curr_global_jpos.reshape(-1, 66).detach().cpu().numpy()
                self.window_data_dict[s_idx]['global_jvel'] = curr_global_jvel.reshape(-1, 66).detach().cpu().numpy()
                self.window_data_dict[s_idx]['global_rot_6d'] = curr_global_rot_6d.reshape(-1, 22*6).detach().cpu().numpy()

                s_idx += 1 

    def extract_min_max_mean_std_from_data(self):
        all_global_jpos_data = []
        all_global_jvel_data = []

        for s_idx in self.window_data_dict: 
            all_global_jpos_data.append(self.window_data_dict[s_idx]['global_jpos'])
            all_global_jvel_data.append(self.window_data_dict[s_idx]['global_jvel'])

        all_global_jpos_data = np.vstack(all_global_jpos_data).reshape(-1, 66) # (N*T) X 66 
        all_global_jvel_data = np.vstack(all_global_jvel_data).reshape(-1, 66)

        min_jpos = all_global_jpos_data.min(axis=0)
        max_jpos = all_global_jpos_data.max(axis=0)
        min_jvel = all_global_jvel_data.min(axis=0)
        max_jvel = all_global_jvel_data.max(axis=0)

        stats_dict = {}
        stats_dict['global_jpos_min'] = min_jpos 
        stats_dict['global_jpos_max'] = max_jpos 
        stats_dict['global_jvel_min'] = min_jvel 
        stats_dict['global_jvel_max'] = max_jvel  

        return stats_dict 

    def normalize_jpos_min_max(self, ori_jpos):
        # ori_jpos: T X 22 X 3 
        normalized_jpos = (ori_jpos - self.global_jpos_min.to(ori_jpos.device))/(self.global_jpos_max.to(ori_jpos.device)\
        -self.global_jpos_min.to(ori_jpos.device))
        normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 

        return normalized_jpos # T X 22 X 3 

    def de_normalize_jpos_min_max(self, normalized_jpos):
        normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range
        de_jpos = normalized_jpos * (self.global_jpos_max.to(normalized_jpos.device)-\
        self.global_jpos_min.to(normalized_jpos.device)) + self.global_jpos_min.to(normalized_jpos.device)

        return de_jpos # T X 22 X 3 

    def normalize_jvel_min_max(self, ori_jvel):
        # ori_jpos: T X 22 X 3 
        normalized_jvel = (ori_jvel - self.global_jvel_min.to(ori_jvel.device))/(self.global_jvel_max.to(ori_jvel.device)\
        -self.global_jvel_min.to(ori_jvel.device))
        normalized_jvel = normalized_jvel * 2 - 1 # [-1, 1] range 

        return normalized_jvel # T X 22 X 3 

    def de_normalize_jvel_min_max(self, normalized_jvel):
        normalized_jvel = (normalized_jvel + 1) * 0.5 # [0, 1] range
        de_jvel = normalized_jvel * (self.global_jvel_max.to(normalized_jvel.device)-\
        self.global_jpos_min.to(normalized_jvel.device)) + self.global_jvel_min.to(normalized_jvel.device)

        return de_jvel # T X 22 X 3 

    def process_window_data(self, seq_root_trans, seq_root_orient, seq_pose_body, random_t_idx, end_t_idx):
        window_root_trans = torch.from_numpy(seq_root_trans[random_t_idx:end_t_idx+1]).float().cuda()
        window_root_orient = torch.from_numpy(seq_root_orient[random_t_idx:end_t_idx+1]).float().cuda()
        window_pose_body  = torch.from_numpy(seq_pose_body[random_t_idx:end_t_idx+1]).float().cuda()

        window_root_rot_mat = transforms.axis_angle_to_matrix(window_root_orient) # T' X 3 X 3 
        window_root_quat = transforms.matrix_to_quaternion(window_root_rot_mat)

        window_pose_rot_mat = transforms.axis_angle_to_matrix(window_pose_body) # T' X 21 X 3 X 3 

        # Generate global joint rotation 
        local_joint_rot_mat = torch.cat((window_root_rot_mat[:, None, :, :], window_pose_rot_mat), dim=1) # T' X 22 X 3 X 3 
        global_joint_rot_mat = local2global_pose(local_joint_rot_mat) # T' X 22 X 3 X 3 
        global_joint_rot_quat = transforms.matrix_to_quaternion(global_joint_rot_mat) # T' X 22 X 4 

        if self.opt.canonicalize_init_head:
            # Canonicalize first frame's facing direction based on global head joint rotation. 
            head_idx = 15 
            global_head_joint_rot_quat = global_joint_rot_quat[:, head_idx, :].detach().cpu().numpy() # T' X 4 

            aligned_root_trans, aligned_head_quat, recover_rot_quat = \
            rotate_at_frame_smplh(window_root_trans.detach().cpu().numpy()[np.newaxis], \
            global_head_joint_rot_quat[np.newaxis], cano_t_idx=0)
            # BS(1) X T' X 3, BS(1) X T' X 4, BS(1) X 1 X 1 X 4  
            # recover_rot_quat: from [1, 0, 0] to the actual forward direction 

            # Apply the rotation to the root orientation 
            cano_window_root_quat = transforms.quaternion_multiply( \
            transforms.quaternion_invert(torch.from_numpy(recover_rot_quat[0, 0]).float().to(\
            window_root_quat.device)).repeat(window_root_quat.shape[0], 1), window_root_quat) # T' X 4 
            cano_window_root_rot_mat = transforms.quaternion_to_matrix(cano_window_root_quat) # T' X 3 X 3 

            cano_local_joint_rot_mat = torch.cat((cano_window_root_rot_mat[:, None, :, :], window_pose_rot_mat), dim=1) # T' X 22 X 3 X 3 
            cano_global_joint_rot_mat = local2global_pose(cano_local_joint_rot_mat) # T' X 22 X 3 X 3 
            
            cano_local_rot_aa_rep = transforms.matrix_to_axis_angle(cano_local_joint_rot_mat) # T' X 22 X 3 

            cano_local_rot_6d = transforms.matrix_to_rotation_6d(cano_local_joint_rot_mat)
            cano_global_rot_6d = transforms.matrix_to_rotation_6d(cano_global_joint_rot_mat)

            # Generate global joint position 
            local_jpos = self.rest_human_offsets.repeat(cano_local_rot_aa_rep.shape[0], 1, 1) # T' X 22 X 3 
            _, human_jnts = quat_fk_torch(cano_local_joint_rot_mat, local_jpos) # T' X 22 X 3 
            human_jnts = human_jnts + torch.from_numpy(aligned_root_trans[0][:, None, :]).float().to(human_jnts.device)# T' X 22 X 3 

            # Move the trajectory based on global head position. Make the head joint to x = 0, y = 0. 
            global_head_jpos = human_jnts[:, head_idx, :] # T' X 3 
            move_to_zero_trans = global_head_jpos[0:1].clone() # 1 X 3
            move_to_zero_trans[:, 2] = 0  
        
            global_jpos = human_jnts - move_to_zero_trans[None] # T' X 22 X 3  

            global_jvel = global_jpos[1:] - global_jpos[:-1] # (T'-1) X 22 X 3 

            query = {}

            query['local_rot_mat'] = cano_local_joint_rot_mat # T' X 22 X 3 X 3 
            query['local_rot_6d'] = cano_local_rot_6d # T' X 22 X 6

            query['global_jpos'] = global_jpos # T' X 22 X 3 
            query['global_jvel'] = torch.cat((global_jvel, \
                torch.zeros(1, 22, 3).to(global_jvel.device)), dim=0) # T' X 22 X 3 
            
            query['global_rot_mat'] = cano_global_joint_rot_mat # T' X 22 X 3 X 3 
            query['global_rot_6d'] = cano_global_rot_6d # T' X 22 X 6
        else:
            curr_seq_pose_aa = torch.cat((window_root_orient[:, None, :], window_pose_body), dim=1) # T' X 22 X 3 
            curr_seq_local_jpos = self.rest_human_offsets.repeat(curr_seq_pose_aa.shape[0], 1, 1) # T' X 22 X 3 

            curr_seq_pose_rot_mat = transforms.axis_angle_to_matrix(curr_seq_pose_aa)
            _, human_jnts = quat_fk_torch(curr_seq_pose_rot_mat, curr_seq_local_jpos)
            human_jnts = human_jnts + window_root_trans[:, None, :] # T' X 22 X 3  

            head_idx = 15 
            # Move the trajectory based on global head position. Make the head joint to x = 0, y = 0. 
            global_head_jpos = human_jnts[:, head_idx, :] # T' X 3 
            move_to_zero_trans = global_head_jpos[0:1].clone() # 1 X 3
            move_to_zero_trans[:, 2] = 0  
        
            global_jpos = human_jnts - move_to_zero_trans[None] # T' X 22 X 3  

            global_jvel = global_jpos[1:] - global_jpos[:-1] # (T'-1) X 22 X 3 

            local_joint_rot_mat = transforms.axis_angle_to_matrix(curr_seq_pose_aa) # T' X 22 X 3 X 3 
            global_joint_rot_mat = local2global_pose(local_joint_rot_mat) # T' X 22 X 3 X 3 

            local_rot_6d = transforms.matrix_to_rotation_6d(local_joint_rot_mat)
            global_rot_6d = transforms.matrix_to_rotation_6d(global_joint_rot_mat)

            query = {}

            query['local_rot_mat'] = local_joint_rot_mat # T' X 22 X 3 X 3 
            query['local_rot_6d'] = local_rot_6d # T' X 22 X 6

            query['global_jpos'] = global_jpos # T' X 22 X 3 
            query['global_jvel'] = torch.cat((global_jvel, \
                torch.zeros(1, 22, 3).to(global_jvel.device)), dim=0) # T' X 22 X 3 
            
            query['global_rot_mat'] = global_joint_rot_mat # T' X 22 X 3 X 3 
            query['global_rot_6d'] = global_rot_6d # T' X 22 X 6

        return query 

    def __len__(self):
        return len(self.window_data_dict)

    def __getitem__(self, index):
        globla_jpos = torch.from_numpy(self.window_data_dict[index]['global_jpos']).float()
        global_rot6d = torch.from_numpy(self.window_data_dict[index]['global_rot_6d']).float()

        data_input = torch.cat((globla_jpos, global_rot6d), dim=-1) 

        num_joints = 22
        normalized_jpos = self.normalize_jpos_min_max(data_input[:, :num_joints*3].reshape(-1, num_joints, 3)) # T X 22 X 3 

        new_data_input = torch.cat((normalized_jpos.reshape(-1, 66), global_rot6d), dim=1)

        actual_seq_len = new_data_input.shape[0]

        if actual_seq_len < self.window:
            # Add padding
            padded_data_input = torch.zeros(self.window-actual_seq_len, new_data_input.shape[1]) 

            new_data_input = torch.cat((new_data_input, padded_data_input), dim=0)

        data_input_dict = {}
        data_input_dict['motion'] = new_data_input # T X (22*3+22*6) range [-1, 1]
        data_input_dict['seq_len'] = actual_seq_len 

        return data_input_dict 
        