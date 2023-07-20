import os, sys
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import argparse
import math

import numpy as np
import torch

import subprocess

import habitat_sim
from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
from habitat_sim import geo

import magnum as mn
import quaternion

from habitat_utils.utils import batch_rigid_transform, REGION_NAME_DICT

# define some globals the first time we run.
if "sim" not in globals():
    global sim
    sim = None

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []
    if settings["color_sensor_1st_person"]:
        color_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        color_sensor_1st_person_spec.uuid = "color_sensor_1st_person"
        color_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        color_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        color_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_1st_person_spec)
    if settings["depth_sensor_1st_person"]:
        depth_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_1st_person_spec.uuid = "depth_sensor_1st_person"
        depth_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        depth_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        depth_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_1st_person_spec)
    if settings["semantic_sensor_1st_person"]:
        semantic_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_1st_person_spec.uuid = "semantic_sensor_1st_person"
        semantic_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        semantic_sensor_1st_person_spec.position = [
            0.0,
            settings["sensor_height"],
            0.0,
        ]
        semantic_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        semantic_sensor_1st_person_spec.sensor_subtype = (
            habitat_sim.SensorSubType.PINHOLE
        )
        sensor_specs.append(semantic_sensor_1st_person_spec)
    if settings["color_sensor_3rd_person"]:
        color_sensor_3rd_person_spec = habitat_sim.CameraSensorSpec()
        color_sensor_3rd_person_spec.uuid = "color_sensor_3rd_person"
        color_sensor_3rd_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_3rd_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        color_sensor_3rd_person_spec.position = [
            0.0,
            settings["sensor_height"] + 0.2,
            0.2,
        ]
        color_sensor_3rd_person_spec.orientation = [-math.pi / 4, 0, 0]
        color_sensor_3rd_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_3rd_person_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def make_default_settings():
    settings = {
        "width": 224,  # Spatial resolution of the observations
        "height": 224,
        "scene": "/viscam/u/jiamanli/datasets/replica/hotel_0/habitat/mesh_semantic.ply",  # Scene path
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "sensor_pitch": -math.pi / 8.0,  # sensor pitch (x rotation in rads)
        "color_sensor_1st_person": True,  # RGB sensor
        "color_sensor_3rd_person": False,  # RGB sensor 3rd person
        "depth_sensor_1st_person": False,  # Depth sensor
        "semantic_sensor_1st_person": False,  # Semantic sensor
        "seed": 1,
        "enable_physics": False,  # enable dynamics simulation
    }
    return settings

def make_simulator_from_settings(sim_settings):
    cfg = make_cfg(sim_settings)
    # clean-up the current simulator instance if it exists
    global sim

    if sim != None:
        sim.close()
    # initialize the simulator
    sim = habitat_sim.Simulator(cfg)

def main(args):
    # load scene
    sim_settings = make_default_settings()
    sim_settings["scene"] = "/viscam/u/jiamanli/datasets/replica/" + args.house_name + "/habitat/mesh_semantic.ply"
    sim_settings["sensor_pitch"] = 0
    sim_settings["sensor_height"] = 1. 
    
    make_simulator_from_settings(sim_settings)
    
    visual_sensor = sim._sensors["color_sensor_1st_person"]
    
    # set sensor initial position
    visual_sensor._spec.position = np.array([0, 0, 0]) 
    visual_sensor._spec.orientation = np.array([0, 0, 0]) 
    visual_sensor._sensor_object.set_transformation_from_spec()
   
    house_dir = os.path.join(args.data_dir, args.house_name)

    region_dir = house_dir
    motion_names = sorted(os.listdir(region_dir))

    TRAIN_DATASETS = ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje', 'BMLmovi', 
                    'EKUT', 'ACCAD']
    TEST_DATASETS = ['Transitions_mocap', 'HumanEva']
    VAL_DATASETS = ['MPI_HDM05', 'SFU', 'MPI_mosh']

    ALL_DATASETS = ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje', 'BMLmovi', 
                    'EKUT', 'ACCAD', 'Transitions_mocap', 'HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh']

    dataset_name_list = None
    if args.data_split == "train":
        dataset_name_list = TRAIN_DATASETS
    elif args.data_split == "val":
        dataset_name_list = VAL_DATASETS
    elif args.data_split == "test":
        dataset_name_list = TEST_DATASETS 
    elif args.data_split == "all":
        dataset_name_list = ALL_DATASETS 
    
    # Tmp, only select specific sequence
    motion_names = [] 
    for tmp_name in sorted(os.listdir(region_dir)):
        # if "Eyes_Japan_Dataset_kudo_jump-06-rope_normal_run_fast" in tmp_name:
        #     motion_names.append(tmp_name)
        for dataset_name in dataset_name_list:
            if dataset_name in tmp_name:
                motion_names.append(tmp_name)

    for motion_name in motion_names:
        motion_dir = os.path.join(region_dir, motion_name)
        
        # if current motion already has observations saved, skip to the next motion
        observation_dir = os.path.join(motion_dir, "observations")
        if os.path.exists(observation_dir): continue
        
        # load motion sequence
        motion_seq = np.load(os.path.join(motion_dir, "motion_seq.npz"))

        root_ori = motion_seq["root_orient"].reshape(-1, 3, 3)				# T x 3 x 3 
            
        # load kinematic tree file
        parents = np.load('humor/body_models/smplh/male/model.npz')['kintree_table'][0][:22]
        parents[0] = -1

        # move tensors to gpu
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        #parents = torch.from_numpy(parents).to(device).expand(all_trans.shape[1], 22)           # T x 22
        parents = torch.from_numpy(parents).to(device)          # 22
        rot_mats = torch.from_numpy(motion_seq["pose_body"]).to(device).view(-1, 21, 3, 3)      # T x 21 x 3 x 3
        root_rot = torch.from_numpy(root_ori).to(device).unsqueeze(1)
        rot_mats = torch.cat((root_rot, rot_mats), dim=1)
        J = torch.from_numpy(motion_seq["joints"]).to(device)                                   # T x 22 x 3
        dtype = J.dtype
        
        J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

        # get corresponding joint orientations
        ori_head = A[:, 15, :3, :3].cpu().numpy()
       
        quat_mp3d_to_habitat = quat_from_two_vectors(np.array([0, 0, -1]), geo.GRAVITY)
        rot_mat_mp3d_to_habitat = quaternion.as_rotation_matrix(quat_mp3d_to_habitat)

        head_vert_pos = motion_seq["head_cam_v_pos"] # T X 3

        # output directory
        os.makedirs(observation_dir)
       
        # For joint head, use the vertex position instead, we only need the head-mounted camera for now. 
        trans = head_vert_pos  
                        # T x 3
        ori = ori_head
        joint_name = "head"
        joint_observation_dir = os.path.join(observation_dir, joint_name)
        os.makedirs(joint_observation_dir, exist_ok=True)

        for t in range(trans.shape[0]):
            camera_pos = trans[t]
            #camera_rot = root_ori[t]
            camera_rot = ori[t]
            
            root_ori_t_x = -camera_rot[:, 0]
            root_ori_t_y = camera_rot[:, 1]
            root_ori_t_z = -camera_rot[:, 2]
            camera_rot = np.stack((root_ori_t_x, root_ori_t_y, root_ori_t_z), axis=1)

            #camera_pos_habitat = quat_rotate_vector(quat_mp3d_to_habitat, camera_pos)
            camera_pos_habitat = rot_mat_mp3d_to_habitat @ camera_pos
            camera_rot_habitat = rot_mat_mp3d_to_habitat @ camera_rot

            sim.get_agent(0).scene_node.translation = camera_pos_habitat
            sim.get_agent(0).scene_node.rotation = mn.Quaternion.from_matrix(camera_rot_habitat)
            observation = habitat_sim.utils.viz_utils.observation_to_image(sim.get_sensor_observations()['color_sensor_1st_person'], \
                                        observation_type='color')
            
            output_path = os.path.join(joint_observation_dir, "%05d.png"%t)
            observation.save(output_path)

            # if args.save_video:
            # ffmpeg.input(joint_observation_dir + '/*.png', pattern_type='glob', framerate=12).output(os.path.join(motion_dir, '{}_{}_observations.mp4'.format(motion_name, joint_name))).overwrite_output().run()
            output_vid_file = os.path.join(motion_dir, '{}_{}_observations.mp4'.format(motion_name, joint_name))
            command = [
                'ffmpeg', '-r', '30', '-y', '-threads', '16', '-i', f'{joint_observation_dir}/%05d.png', '-profile:v', 'baseline',
                '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
            ]

            print(f'Running \"{" ".join(command)}\"')
            subprocess.call(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--house_name', type=str, required=True)
    parser.add_argument('--data_split', type=str, required=True)
    parser.add_argument('--save_video', action='store_true')
    args = parser.parse_args()

    main(args)
