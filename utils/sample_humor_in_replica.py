
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import argparse
import importlib, time

import json
import random 

import numpy as np

from plyfile import PlyData

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.config import TestConfig
from utils.logging import Logger, class_name_to_file_name, mkdir, cp_files
from utils.torch import get_device, save_state, load_state
from utils.stats import StatTracker
from utils.transforms import rotation_matrix_to_angle_axis
from body_model.utils import SMPL_JOINTS
from datasets.amass_utils import NUM_KEYPT_VERTS, CONTACT_INDS
from losses.humor_loss import CONTACT_THRESH

NUM_WORKERS = 0

def parse_args(argv):
    # create config and parse args
    config = TestConfig(argv)
    known_args, unknown_args = config.parse()
    print('Unrecognized args: ' + str(unknown_args))
    return known_args

def write_to_obj(dest_obj_file, vertices, faces):
    # vertices: N X 3, faces: N X 3
    w_f = open(dest_obj_file, 'w')

    # print("Total vertices:{0}".format(vertices.shape[0]))

    # Write vertices to file 
    for idx in range(vertices.shape[0]):
        w_f.write("v "+str(vertices[idx, 0])+" "+str(vertices[idx, 1])+" "+str(vertices[idx, 2])+"\n")

    # Write faces to file 
    for idx in range(faces.shape[0]):
        w_f.write("f "+str(faces[idx, 0]+1)+" "+str(faces[idx, 1]+1)+" "+str(faces[idx, 2]+1)+"\n")

    w_f.close() 

def check_if_valid(vertices, sdf, centroid, extents, grid_dim, voxel_size, sdf_penetration_weight):
    # vertices: bs(1) X N X 3
    # Compute scene penetration using signed distance field (SDF)
    sdf_penetration_loss = 0.0
    nv = vertices.shape[1]

    # sdf_ids = torch.round(
    #     (vertices.squeeze() - grid_min) / voxel_size).to(dtype=torch.long)
    # sdf_ids.clamp_(min=0, max=grid_dim-1)

    centroid = torch.from_numpy(centroid).float().to(vertices.device)
    extents = torch.from_numpy(extents).float().to(vertices.device)

    norm_vertices = vertices - centroid
    norm_vertices *= 2 / extents.max()

    # norm_vertices = (vertices - grid_min) / (grid_max - grid_min) * 2 - 1 # normalize to range(-1, 1)

    body_sdf = F.grid_sample(sdf.view(1, 1, grid_dim, grid_dim, grid_dim),
                                norm_vertices[:, :, [2, 1, 0]].view(1, nv, 1, 1, 3),
                                #norm_vertices.view(1, nv, 1, 1, 3),
                                padding_mode='border') # Grid sample return z, y, x!!!
   
    # if there are no penetrating vertices then set sdf_penetration_loss = 0
    if body_sdf.lt(0).sum().item() < 1:
        sdf_penetration_loss = torch.tensor(0.0, dtype=vertices.dtype, device=vertices.device)
    else:
        sdf_penetration_loss = sdf_penetration_weight * body_sdf[body_sdf < 0].abs().sum()

    in_penetration = (body_sdf < 0)
    return sdf_penetration_loss, in_penetration

def gen_data_npz(x_pred_dict, meta, head_cam_verts, actual_t, end_idx, dest_npz_path, meta_data):
    # x_pred_dict: contains data in the canonical frame of the first frame's pose. 
    # ['trans', 'trans_vel', 'root_orient', 'root_orient_vel', 'pose_body', 'joints', 'joints_vel', 'contacts']
    # cano_rot_inv: 4 X 4, transform the canonical frame of the first pose to the aligned_floor frame. 
    # new_world2aligned_rot: T X 3 X 3 
    # new_cano_rot_mat_arr: T X 4 X 4 
    # head_cam_verts: T X 3 
    from body_model.body_model import BodyModel
    from body_model.utils import SMPLH_PATH

    new_trans = x_pred_dict['trans'][0, :end_idx][:actual_t] # T X 3
    # new_trans_vel= x_pred_dict['trans_vel'][0, :end_idx][:actual_t] # T X 3
    new_root_orient = x_pred_dict['root_orient'][0, :end_idx][:actual_t] # T X 9 
    # new_root_orient_vel = x_pred_dict['root_orient_vel'][0, :end_idx][:actual_t] # T X 3
    new_pose_body = x_pred_dict['pose_body'][0, :end_idx][:actual_t] # T X 189
    new_joints = x_pred_dict['joints'][0, :end_idx][:actual_t] # T X 66
    # new_joints_vel = x_pred_dict['joints_vel'][0, :end_idx][:actual_t] # T X 66 

    np.savez(dest_npz_path, fps=meta['fps'], 
        path=meta['path'],
        gender=meta['gender'],
        start_frame_idx=meta['start_frame_idx']+1, # Since the first frame is not in global_out from dataloader. 
        trans=new_trans.data.cpu().numpy(), # T X 3
        root_orient=new_root_orient.data.cpu().numpy(), # T X 3 
        pose_body=new_pose_body.data.cpu().numpy(), # T X 63
        betas=meta['betas'][0, 0].data.cpu().numpy(), # 10
        head_cam_v_pos=head_cam_verts.data.cpu().numpy(), # T X 3 
        joints=new_joints.reshape(-1, 22, 3).data.cpu().numpy(),) # T X 22 X 3
        # joints_vel=new_joints_vel.data.cpu().numpy(), # T X 22 X 3
        # trans_vel=new_trans.data.cpu().numpy(), # T X 3
        # root_orient_vel=new_root_orient_vel.data.cpu().numpy()) # T X 3
        # joint_orient_vel_seq=joint_orient_vel_seq, # T: Based on joints_world2aligned_rot, calculate angular velocity for z-axis rotation. 
        # pose_body_vel=pose_body_vel_seq, # T X 21 X 3

def test(args_obj, config_file):

    # set up output
    args = args_obj.base
    mkdir(args.out)

    # create logging system
    test_log_path = os.path.join(args.out, 'test_'+args.house_name+'.log')
    Logger.init(test_log_path)

    # save arguments used
    Logger.log('Base args: ' + str(args))
    Logger.log('Model args: ' + str(args_obj.model))
    Logger.log('Dataset args: ' + str(args_obj.dataset))
    Logger.log('Loss args: ' + str(args_obj.loss))

    # save training script/model/dataset/config used
    test_scripts_path = os.path.join(args.out, 'test_scripts')
    mkdir(test_scripts_path)
    pkg_root = os.path.join(cur_file_path, '..')
    dataset_file = class_name_to_file_name(args.dataset)
    dataset_file_path = os.path.join(pkg_root, 'datasets/' + dataset_file + '.py')
    model_file = class_name_to_file_name(args.model)
    loss_file = class_name_to_file_name(args.loss)
    model_file_path = os.path.join(pkg_root, 'models/' + model_file + '.py')
    train_file_path = os.path.join(pkg_root, 'test/sample_humor_in_replica.py')
    # cp_files(test_scripts_path, [train_file_path, model_file_path, dataset_file_path, config_file])

    # load model class and instantiate
    model_class = importlib.import_module('models.' + model_file)
    Model = getattr(model_class, args.model)
    model = Model(**args_obj.model_dict,
                    model_smpl_batch_size=args.batch_size) # assumes model is HumorModel

    # load loss class and instantiate
    loss_class = importlib.import_module('losses.' + loss_file)
    Loss = getattr(loss_class, args.loss)
    loss_func = Loss(**args_obj.loss_dict,
                      smpl_batch_size=args.batch_size*args_obj.dataset.sample_num_frames) # assumes loss is HumorLoss

    device = get_device(args.gpu)
    model.to(device)
    loss_func.to(device)

    print(model)

    # count params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    Logger.log('Num model params: ' + str(params))

    # freeze params in loss
    for param in loss_func.parameters():
        param.requires_grad = False

    # load in pretrained weights if given
    if args.ckpt is not None:
        start_epoch, min_val_loss, min_train_loss = load_state(args.ckpt, model, optimizer=None, map_location=device, ignore_keys=model.ignore_keys)
        Logger.log('Successfully loaded saved weights...')
        Logger.log('Saved checkpoint is from epoch idx %d with min val loss %.6f...' % (start_epoch, min_val_loss))
    else:
        Logger.log('ERROR: No weight specified to load!!')
        # return

    # load dataset class and instantiate training and validation set
    if args.test_on_train:
        Logger.log('WARNING: running evaluation on TRAINING data as requested...should only be used for debugging!')
    elif args.test_on_val:
        Logger.log('WARNING: running evaluation on VALIDATION data as requested...should only be used for debugging!')
    Dataset = getattr(importlib.import_module('datasets.' + dataset_file), args.dataset)
    split = 'test'
    if args.test_on_train:
        split = 'train'
    elif args.test_on_val:
        split = 'val'
    split = 'all' # For generate the whole dataset, use different split for locomotion sicne before we generated training data for locomotion, keep it consistent 
    test_dataset = Dataset(split=split, **args_obj.dataset_dict)

    # only select a subset of data
    # subset_indices = np.random.choice(len(test_dataset), size=args.num_batches, replace=False)
    # subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
    
    # create loaders
    test_loader = DataLoader(test_dataset, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            # sampler=subset_sampler,
                            num_workers=NUM_WORKERS,
                            pin_memory=True,
                            drop_last=False,
                            worker_init_fn=lambda _: np.random.seed())

    test_dataset.return_global = True
    model.dataset = test_dataset

    house_name = args.house_name
    if args.eval_sampling or args.eval_sampling_debug:
        eval_sampling(model, test_dataset, test_loader, device, house_name, 
                            out_dir=args.out if args.eval_sampling else None,
                            num_samples=args.eval_num_samples,
                            samp_len=args.eval_sampling_len,
                            viz_contacts=args.viz_contacts,
                            viz_pred_joints=args.viz_pred_joints,
                            viz_smpl_joints=args.viz_smpl_joints,
                            write_obj=args.write_obj, 
                            save_seq_len=args.seq_len)

    Logger.log('Finished!')

def eval_sampling(model, test_dataset, test_loader, device, house_name, 
                  out_dir=None,
                  num_samples=1,
                  samp_len=10.0,
                  viz_contacts=False,
                  viz_pred_joints=False,
                  viz_smpl_joints=False,
                  write_obj=False,
                  save_seq_len=None):
    Logger.log('Evaluating sampling qualitatively...')
    from body_model.body_model import BodyModel
    from body_model.utils import SMPLH_PATH

    # eval_qual_samp_len = int(samp_len * 30.0) # at 30 Hz
    eval_qual_samp_len = 150

    if out_dir is not None:
        #res_out_dir = os.path.join(out_dir, 'eval_sampling')
        house_out_dir = os.path.join(out_dir, house_name)
        if not os.path.exists(house_out_dir):
            os.mkdir(house_out_dir)

    J = len(SMPL_JOINTS)
    V = NUM_KEYPT_VERTS
    male_bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    female_bm_path = os.path.join(SMPLH_PATH, 'female/model.npz')
    male_bm = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=eval_qual_samp_len).to(device)
    female_bm = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=eval_qual_samp_len).to(device)

    with torch.no_grad():
        test_dataset.pre_batch()
        model.eval()
        for i, data in enumerate(test_loader):
            # get inputs
            batch_in, batch_out, meta = data
            print(meta['path'])
            seq_name_list = [spath[:-4] for spath in meta['path']]
            
            batch_res_out_list = [os.path.join(house_out_dir, seq_name.replace('/', '_') + '_b' + str(i) + 'seq' + str(sidx)) for sidx, seq_name in enumerate(seq_name_list)]
            print(batch_res_out_list)
            # continue
            x_past, _, gt_dict, input_dict, global_gt_dict = model.prepare_input(batch_in, device, 
                                                                                data_out=batch_out,
                                                                                return_input_dict=True,
                                                                                return_global_dict=True)

            # roll out predicted motion
            B, T, _, _ = x_past.size()
            x_past = x_past[:,0,:,:] # only need input for first step
            rollout_input_dict = dict()
            for k in input_dict.keys():
                rollout_input_dict[k] = input_dict[k][:,0,:,:] # only need first step

            # load scene sdf
            sdf_dir = "/viscam/u/jiamanli/datasets/replica_processed/replica_fixed_poisson_sdfs_res256"
            scene_name = house_name
            # with open(os.path.join(sdf_dir, scene_name + ".json"), "r") as f:
            #     sdf_data = json.load(f)
            #     grid_min = torch.tensor((sdf_data['min'][0], -sdf_data['min'][2], sdf_data['min'][1]), dtype=torch.float32, device=device)
            #     grid_max = torch.tensor((sdf_data['max'][0], -sdf_data['max'][2], sdf_data['max'][1]), dtype=torch.float32, device=device)
            #     grid_dim = sdf_data['dim']
            grid_dim = 256 
            # voxel_size = (grid_max - grid_min) / grid_dim
            voxel_size = 256 
            sdf = np.load(os.path.join(sdf_dir, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
            sdf = torch.tensor(sdf, dtype=torch.float32, device=device)
            sdf_penetration_weight = 1
            penetration_loss_threshold = 2
            min_seq_len = 70

            json_sdf_info_path = os.path.join(sdf_dir, scene_name+"_sdf_info.json")
            json_sdf_info = json.load(open(json_sdf_info_path, 'r'))
            centroid = np.asarray(json_sdf_info['centroid']) # 3
            extents = np.asarray(json_sdf_info['extents']) # 3 

            use_gt_data = True 

            # sample same trajectory multiple times and save the joints/contacts output
            for samp_idx in range(num_samples):
                if use_gt_data:
                    x_pred_dict = {}
                    for tmp_k in global_gt_dict:
                        x_pred_dict[tmp_k] = global_gt_dict[tmp_k].squeeze(2)
                else:
                    x_pred_dict = model.roll_out(x_past, rollout_input_dict, eval_qual_samp_len, gender=meta['gender'], betas=meta['betas'].to(device))

                # translate the human to the scene
                x_pred_dict = translate_to_scene(x_pred_dict, house_name)

                # visualize and save
                print('Visualizing sample %d/%d!' % (samp_idx+1, num_samples))
                imsize = (1080, 1080)
                cur_res_out_list = batch_res_out_list
                cur_res_out_path = cur_res_out_list[0]+"_samp_"+str(samp_idx)
                human_verts, human_faces = viz_eval_samp(global_gt_dict, x_pred_dict, meta, male_bm, female_bm)
                human_verts = torch.from_numpy(human_verts).float().to(device).squeeze(0)

                eval_qual_samp_len = human_verts.shape[0]

                valid_verts_list = []
                end_idx = 0                           # the index till where is the valid sequence
                for f_idx in range(eval_qual_samp_len):
                    penetration_loss, in_penetration = check_if_valid(human_verts[f_idx: f_idx + 1], sdf, centroid, extents, grid_dim, \
                        voxel_size, sdf_penetration_weight)

                    # print("penetration loss:", penetration_loss)
                    if penetration_loss > penetration_loss_threshold:
                        # if len(valid_verts_list) >= min_valid_seq_len:
                        end_idx = f_idx
                        break
                    else:
                        valid_verts_list.append(human_verts[f_idx])
                    #valid_verts_list.append(human_verts[f_idx])
              
                if len(valid_verts_list) == eval_qual_samp_len:
                    end_idx = eval_qual_samp_len

                # if save_seq_len:
                #     seq_len = save_seq_len
                # else:
                seq_len = end_idx - 10 

                if len(valid_verts_list) >= min_seq_len: 
                    # If longer than minimum sequence length, save motion sequence and collision supervision
                    # valid_verts_seq = torch.stack(valid_verts_list[-seq_len:]).to(device).squeeze(0)
                    #valid_verts_seq = torch.stack(valid_verts_list).to(device).squeeze(0)
                    #valid_verts_seq = torch.stack(valid_verts_list).to(device).squeeze(0)[:20, ...]

                    # We discard the motion sequences which is close to penetration... 
                    valid_verts_seq = torch.stack(valid_verts_list[:seq_len]).to(device).squeeze(0)
                    os.makedirs(cur_res_out_path, exist_ok=True)
                    if write_obj: 
                        # for t_idx in range(0, seq_len, 10):
                        for t_idx in range(seq_len):
                            dest_mesh_path = os.path.join(cur_res_out_path, "%05d"%(t_idx) + ".obj")
                            write_to_obj(dest_mesh_path, valid_verts_seq[t_idx].data.cpu().numpy(), human_faces.data.cpu().numpy())
                 
                    # Extract the head vertex position of human mesh 
                    head_v_idx = 232 
                    head_cam_verts = valid_verts_seq[:, head_v_idx, :] # T X 3 
                    dest_npz_path = os.path.join(cur_res_out_path, "motion_seq.npz") # This motion seq file is only used for rendering using habitat, for real gt motion, still use original gt data. 
                    gen_data_npz(x_pred_dict, meta, head_cam_verts, seq_len, end_idx, dest_npz_path, meta) 

                    break # If current motion is valid, no need for more sampling for the same motion sequence. 

def trans2velocity(root_trans):
    # root_trans: BS X T X 3 
    root_velocity = root_trans[:, 1:, :] - root_trans[:, :-1, :]
    return root_velocity # BS X (T-1) X 3  

def velocity2trans(init_root_trans, root_velocity):
    # init_root_trans: 3
    # root_velocity: (T-1) X 3

    timesteps = root_velocity.shape[0] + 1
    absolute_pose_data = torch.zeros(timesteps, 3).to(root_velocity.device) # T X 3
    absolute_pose_data[0, :] = init_root_trans.clone() 

    root_trans = init_root_trans[None].clone() # 1 X 3
    for t_idx in range(1, timesteps):
        root_trans += root_velocity[t_idx-1:t_idx, :] # 1 X 3
        absolute_pose_data[t_idx, :] = root_trans # 1 X 3  

    return absolute_pose_data # T X 3
    
def translate_to_scene(x_pred_dict, house_name):
    """Translate the human location to a random location on the floor."""
    # scene_dir = "/orion/u/bxpan/exoskeleton/habitat_resources/mp3d/v1/scans/" + house_name
    # region_dir = os.path.join(scene_dir, "region_segmentations")
    region_dir = "/viscam/u/jiamanli/datasets/replica"

    # region_ply_path = os.path.join(region_dir, "region{}.ply".format(region_index))
    region_ply_path = os.path.join(region_dir, house_name, "habitat", "mesh_semantic.ply")
    region_ply = PlyData.read(region_ply_path)

    # Read semantic json
    sem_json = os.path.join(region_dir, house_name, "habitat", "info_semantic.json")
    sem_data = json.load(open(sem_json, 'r'))
    obj_data = sem_data['objects']
    num_objs = len(obj_data)
    floor_obj_id_list = []
    for o_idx in range(num_objs):
        if obj_data[o_idx]['class_name'] == "floor":
            floor_obj_id_list.append(obj_data[o_idx]['id'])

    # Random pick one idx fron floor obj idx list 
    floor_idx = random.sample(floor_obj_id_list, 1)[0]

    floor_face_ids = np.where(region_ply['face']['object_id'] == floor_idx)
    floor_vertex_ids = np.stack(region_ply['face']['vertex_indices'][floor_face_ids], axis=0).reshape(-1,)

    rand_vertex_id = np.random.choice(floor_vertex_ids)

    rand_vertex_x = region_ply['vertex']['x'][rand_vertex_id]
    rand_vertex_y = region_ply['vertex']['y'][rand_vertex_id]
    rand_vertex_z = region_ply['vertex']['z'][rand_vertex_id]

    # Randomize the initial frame's root orientation 
    rot_angle = random.sample(list(range(0, 360, 20)), 1)[0] 
    rot_angle = torch.tensor(rot_angle)
    rot_z = torch.deg2rad(rot_angle)

    curr_rot_mat = torch.zeros(3, 3).float()
    curr_rot_mat[0, 0] = torch.cos(rot_z)
    curr_rot_mat[1, 0] = torch.sin(rot_z)
    curr_rot_mat[0, 1] = -torch.sin(rot_z)
    curr_rot_mat[1, 1] = torch.cos(rot_z)
    curr_rot_mat[2, 2] = 1
    curr_rot_mat = curr_rot_mat[None] # 1 X 3 X 3

    # Rotate along with the z axis 
    ori_root_rot_mat = x_pred_dict['root_orient'] # 1 X T X 9 
    rotated_root = torch.matmul(curr_rot_mat.to(ori_root_rot_mat.device).repeat(ori_root_rot_mat.shape[1], 1, 1), \
        ori_root_rot_mat.squeeze(0).reshape(-1, 3, 3)) # T X 3 X 3
    x_pred_dict['root_orient'] = rotated_root.reshape(1, -1, 9) # 1 X T X 9  

    # Rotate trans
    ori_root_trans = x_pred_dict['trans'] # 1 X T X 3 
    aligned_root_trans = torch.matmul(curr_rot_mat[0].to(ori_root_trans.device), ori_root_trans[0].T).T # T X 3 
    # print(global_trans.shape)

    # Get the offsets between joints and trans 
    trans2joint = torch.zeros(1, 1, 3).to(ori_root_trans.device) # 1 X 1 X 3 
    trans2joint[0, 0, :2] = ori_root_trans[0, 0, :2] - x_pred_dict['joints'][0, 0, :2] # Translation from origin to root joint

    # Rotate joints
    ori_joints = x_pred_dict['joints'].reshape(1, -1, 22, 3) # 1 X T X 22 X 3  
    aligned_joints = ori_joints.clone() 
    aligned_joints += trans2joint[None]
    aligned_joints = torch.matmul(curr_rot_mat[0].to(aligned_joints.device), aligned_joints.reshape((-1, 3)).T).T.reshape((-1, 22, 3)) # T X 22 X 3 
    aligned_joints -= trans2joint # T X 22 X 3 

    # Translate SMPL mesh   
    init_delta_joints_x = rand_vertex_x - aligned_joints[0, 0, 0].data.cpu().item()
    init_delta_joints_y = rand_vertex_y - aligned_joints[0, 0, 1].data.cpu().item() 
    init_delta_joints_z = rand_vertex_z 
    init_delta_trans = torch.from_numpy(np.asarray([init_delta_joints_x, init_delta_joints_y, init_delta_joints_z])).float() 

    aligned_root_trans += init_delta_trans.to(aligned_root_trans.device)[None]
    aligned_joints += init_delta_trans.to(aligned_joints.device)[None, None] 

    # Assign aligned value to dict 
    x_pred_dict['trans'] = aligned_root_trans[None] # 1 X T X 3
    x_pred_dict['joints'] = aligned_joints.reshape(-1, 22*3)[None] # 1 X T X (22*3)

    return x_pred_dict

def viz_eval_samp(global_gt_dict, x_pred_dict, meta, male_bm, female_bm):
    '''
    Given x_pred_dict from the model rollout and the ground truth dict, runs through SMPL model to visualize
    '''
    J = len(SMPL_JOINTS)
    V = NUM_KEYPT_VERTS

    pred_world_root_orient = x_pred_dict['root_orient']
    B, T, _ = pred_world_root_orient.size()
    pred_world_root_orient = rotation_matrix_to_angle_axis(pred_world_root_orient.reshape((B*T, 3, 3))).reshape((B, T, 3))
    pred_world_pose_body = x_pred_dict['pose_body']
    pred_world_pose_body = rotation_matrix_to_angle_axis(pred_world_pose_body.reshape((B*T*(J-1), 3, 3))).reshape((B, T, (J-1)*3))
    pred_world_trans = x_pred_dict['trans']
    # pred_world_joints = x_pred_dict['joints'].reshape((B, T, J, 3))

    betas = meta['betas'].to(global_gt_dict[list(global_gt_dict.keys())[0]].device)
    human_verts_list = []
    human_faces = None
    for b in range(B):
        bm_world = male_bm if meta['gender'][b] == 'male' else female_bm
        # pred
        body_pred = bm_world(pose_body=pred_world_pose_body[b], 
                        pose_hand=None,
                        betas=betas[b,0].reshape((1, -1)).expand((T, 16)),
                        root_orient=pred_world_root_orient[b],
                        trans=pred_world_trans[b])

        pred_smpl_joints = body_pred.Jtr[:, :J]
        pred_smpl_verts = body_pred.v
        if human_faces is None:
            human_faces = body_pred.f
        human_verts_list.append(pred_smpl_verts.data.cpu().numpy())
    human_verts = np.asarray(human_verts_list)

    return human_verts, human_faces
      
def main(args, config_file):
    test(args, config_file)

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    config_file = sys.argv[1:][0][1:]

    main(args, config_file)
