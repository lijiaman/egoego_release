import sys 
sys.path.append("../../")
import os 
import joblib 

from copycat.khrylib.utils import *
from collections import defaultdict

from scipy.spatial.transform import Rotation as sRot
from copycat.smpllib.smpl_mujoco import smpl_to_qpose, SMPL_M_Viewer
from mujoco_py import load_model_from_path, MjSim
from copycat.utils.config import Config
from copycat.envs.humanoid_im import HumanoidEnv
from copycat.utils.tools import get_expert
from copycat.data_loaders.dataset_amass_single import DatasetAMASSSingle
from copycat.data_loaders.dataset_smpl_obj import DatasetSMPLObj
from copycat.envs.humanoid_im import HumanoidEnv as CC_HumanoidEnv

from copycat.utils.config import Config as CC_Config

def get_height_data(data_path):
    data = joblib.load(data_path)

    for seq_name in data:
        qpos = data[seq_name]['qpos']
        root_trans = qpos[:, :3] # T X 3 
        root_height = root_trans[:, 2].min()
        print("Root height:{0}".format(root_height))

def get_floor_height(data_path):

    # Initialize config, humanoid simulator 
    cc_cfg = CC_Config("copycat", "/viscam/u/jiamanli/github/egomotion/kin-poly", create_dirs=False)
    # cc_cfg = CC_Config("copycat", "/Users/jiamanli/github/egomotion/kin-poly", create_dirs=False)

    # cc_cfg.data_specs['test_file_path'] = "/Users/jiamanli/github/kin-polysample_data/h36m_train_no_sit_30_qpos.pkl"
    cc_cfg.data_specs['test_file_path'] = "/viscam/u/jiamanli/github/egomotion/kin-poly/sample_data/h36m_test.pkl"
    # cc_cfg.data_specs['test_file_path'] = "/Users/jiamanli/github/egomotion/kin-poly/sample_data/h36m_test.pkl"
    cc_cfg.data_specs['neutral_path'] = "/viscam/u/jiamanli/github/egomotion/kin-poly/sample_data/standing_neutral.pkl"

    cc_cfg.mujoco_model_file = "/viscam/u/jiamanli/github/egomotion/kin-poly/assets/mujoco_models/humanoid_smpl_neutral_mesh_all_step.xml"
    
    data_loader = DatasetSMPLObj(cc_cfg.data_specs, data_mode="test")
    init_expert = data_loader.sample_seq()
    env = CC_HumanoidEnv(cc_cfg, init_expert = init_expert, data_specs = cc_cfg.data_specs, mode="test")

    model_file = f'/viscam/u/jiamanli/github/egomotion/kin-poly/assets/mujoco_models/humanoid_smpl_neutral_mesh.xml'
    humanoid_model =load_model_from_path(model_file)

    data = joblib.load(data_path)

    all_floor_height_list = []
    for seq_name in data:
        qpos = data[seq_name]['qpos']

        floor_height_list = []
        timesteps = qpos.shape[0]
        for t_idx in range(timesteps):
            env.data.qpos[:76] = qpos[t_idx] # 3(trans) + 4(root quat) + 69(other joints' euler angles with respect to parent)
            env.sim.forward()
            wbpos = env.get_wbody_pos() # 72 

            wbpos = wbpos.reshape(24, 3) 

            floor_height = wbpos[:, 2].min()
            floor_height_list.append(floor_height)

        mean_floor_height = np.median(np.asarray(floor_height_list))
        print("Mean Floor height:{0}".format(mean_floor_height)) # Kinpoly: 0.01-0.04

        all_floor_height_list.append(mean_floor_height)

    all_mean_floor_height = np.asarray(all_floor_height_list).mean()
    print("Total data mean floor height:{0}".format(all_mean_floor_height))

if __name__ == "__main__":
    # data_path = "/viscam/u/jiamanli/datasets/kin-poly/MoCapData/features/mocap_annotations.p"
    # data_path = "/viscam/u/jiamanli/datasets/gimo_processed/gimo_motion_for_kinpoly/MoCapData/features/mocap_annotations.p"
    data_path = "/viscam/u/jiamanli/datasets/egomotion_syn_dataset/ego_syn_amass_for_kinpoly/MoCapData/features/mocap_annotations.p"
    # get_height_data(data_path)
    get_floor_height(data_path)
