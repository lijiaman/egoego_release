import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from PIL import Image
import torch
import numpy as np
import argparse
import time
import random
import copy
import glob
import pickle as pk
import joblib
from collections import defaultdict
from tqdm import tqdm
from copycat.khrylib.utils import *

## AMASS Datatset with a Single input sequence
class DatasetAMASSSingle:
    def __init__(self, data_specs, mode="all", data_mode="train"):
        print("******* Reading AMASS Class Data, Single Instance! ***********")
        np.random.seed(0)  # train test split need to stablize
        random.seed(0)
        if data_mode == "train":
            self.data_root = data_specs["file_path"]
        elif data_mode == "test":
            self.data_root = data_specs["test_file_path"]

        self.pickle_data = joblib.load(open(self.data_root, "rb"))
        self.data_specs = data_specs
        # self.amass_data = amass_path
        self.has_smpl_root = data_specs["has_smpl_root"]
        self.flip_cnd = data_specs["flip_cnd"]
        self.t_min = data_specs.get("t_min", 90)
        self.t_max = data_specs.get("t_max", -1)
        self.flip_time = data_specs["flip_time"]

        self.to_one_hot = data_specs.get("to_one_hot", True)
        self.mode = data_specs.get("mode", "all")
        self.adaptive_iter = data_specs.get("adaptive_iter", -1)
        self.netural_path = data_specs.get(
            "neutral_path", "/Users/jiamanli/github/kin-poly/sample_data/standing_neutral.pkl"
        )
        self.netural_data = joblib.load(self.netural_path)
        self.data_mode = data_mode

        self.init_infos = None
        self.init_states = None
        self.init_probs = None

        self.prepare_data()
        self.iter_keys = {}
        self.seq_counter = 0
        self.curr_key = ""
        # if self.adaptive_iter != -1 and self.data_mode == "train":
        #     print("loading states...")
        #     state_file = data_specs['state_file_path']
        #     self.states = joblib.load(state_file)

        print("Dataset Root: ", self.data_root)
        print("Dataset Flip setting: ", self.flip_cnd)
        print("Dataset has SMPL root?: ", self.has_smpl_root)
        print("Dataset Num Sequences: ", self.seq_len)
        print("Traj Dimsnion: ", self.traj_dim)
        print("Data mode: ", self.mode)
        print("Adaptive_iter", self.adaptive_iter)
        print("T Max: ", self.t_max)
        print("T Min: ", self.t_min)
        print("******* Finished AMASS Class Data ***********")

    def prepare_data(self):
        self.data = self.process_data_pickle(self.pickle_data)
        self.traj_dim = list(self.data["pose_6d"].values())[0].shape[1]
        self.seq_len = len(list(self.data["pose_6d"].values()))

    def process_data_pickle(self, pk_data):
        self.sample_keys = []
        self.data_keys = []
        data_out = defaultdict(dict)

        if self.mode == "all":
            process_keys = pk_data.keys()
        elif self.mode == "singles":
            process_keys = self.data_specs["key_subsets"]

        for k in tqdm(process_keys):
            v = pk_data[k]
            smpl_squence = v["pose_aa"] # T X 72
            seq_len = smpl_squence.shape[0]
            if seq_len < self.t_min:
                print(k, seq_len)
                continue

            data_out["pose_6d"][k] = v["pose_6d"] # T X 144
            data_out["pose_aa"][k] = v["pose_aa"] # T X 72
            data_out["trans"][k] = v["trans"] # T X 3
            data_out["qpos"][k] = v["qpos"]# T X 76
            data_out["obj_pose"][k] = (
                v["obj_pose"]
                if ("obj_pose" in v) and (not v["obj_pose"] is None) in v
                else v["qpos"]
            ) # T X 7

            # data_out['entry_names'][k] = k
            if self.t_max != -1:
                [
                    self.sample_keys.append((k, [-1]))
                    for i in range(smpl_squence.shape[0] // self.t_max + 1)
                ]
            else:
                self.sample_keys.append((k, [-1]))

            self.data_keys.append(k)
            # if len(self.data_keys) > 10:
            # break

        return data_out

    def hard_negative_mining(
        self, value_net, env, device, dtype, running_state=None, sampling_temp=0.2
    ):
        print("Hard negative mininig..")
        self.init_values = []
        self.init_info = []
        with torch.no_grad():
            for k, v in tqdm(self.states.items()):
                if v.shape[0] > self.t_min:
                    curr_seq = v[: -self.t_min]
                    if running_state != None:
                        curr_seq = (curr_seq - running_state.rs.mean[None, :]) / (
                            running_state.rs.std[None, :] + 1e-8
                        )
                    state_seq = torch.from_numpy(curr_seq).type(dtype).to(device)
                    curr_values = value_net(state_seq)
                    self.init_values.append(curr_values.cpu().numpy())
                    self.init_info += [(k, i) for i in range(curr_values.shape[0])]
            self.init_values = np.concatenate(self.init_values)

        self.init_probs = np.exp(-self.init_values / sampling_temp)
        self.init_probs = self.init_probs / self.init_probs.sum()

        return (self.init_probs, self.init_info)

    def sample_seq(self, full_sample=False, freq_dict=None):
        # if self.init_probs is None:
        #     sample = random.choice(self.sample_keys)
        #     self.curr_key = sample[0]
        #     pre_start = random.choice(sample[1])
        # else:
        #     rnd_idx = np.random.choice(len(self.init_info), p=self.init_probs.flatten())
        #     self.curr_key, pre_start = self.init_info[rnd_idx]
        self.fr_start = fr_start = 0
        if freq_dict is None:
            sample = random.choice(self.sample_keys)
            self.curr_key = sample[0]
        else:
            sampling_temp = 0.2
            sampling_freq = 0.75
            init_probs = np.exp(
                -np.array(
                    [
                        ewma(np.array(freq_dict[k])[:, 0] == 1)
                        if len(freq_dict[k]) > 0
                        else 0
                        for k in freq_dict.keys()
                    ]
                )
                / sampling_temp
            )
            init_probs = init_probs / init_probs.sum()

            self.curr_key = curr_key = (
                np.random.choice(self.data_keys, p=init_probs)
                if np.random.binomial(1, sampling_freq)
                else np.random.choice(self.data_keys)
            )
            curr_qpos = self.data["qpos"][self.curr_key]
            seq_len = curr_qpos.shape[0]

            #######
            # perfs = np.array(freq_dict[curr_key])
            # if len(perfs) > 0 and len(perfs[perfs[:, 0] != 1][:, 1]) > 0 and np.random.binomial(1, sampling_freq):
            #     perfs = perfs[perfs[:, 0] != 1][:, 1]
            #     chosen_idx = np.random.choice(perfs)
            #     self.fr_start = fr_start = np.random.randint(max(chosen_idx- 20, 0), min(chosen_idx + 20, seq_len - self.t_min))
            # else:
            #     self.fr_start = fr_start =  np.random.randint(0, seq_len - self.t_min)
            #######
            if full_sample:
                self.fr_start = fr_start = 0
            else:
                self.fr_start = fr_start = np.random.randint(0, seq_len - self.t_min)

        return self.get_sample_from_key(
            self.curr_key, full_sample=full_sample, fr_start=fr_start
        )

    def get_sample_from_key(self, take_key, full_sample=False, fr_start=0):
        self.curr_key = take_key

        curr_qpos = self.data["qpos"][take_key] # T X 76 
        seq_len = curr_qpos.shape[0]
        if full_sample:
            self.fr_start = fr_start
            self.fr_end = fr_end = self.data["qpos"][self.curr_key].shape[0]
        else:
            self.fr_start = fr_start
            self.fr_end = fr_end = (
                fr_start + self.t_max if fr_start + self.t_max < seq_len else seq_len
            )

        sample = {}

        for key in self.data.keys():
            sample[key] = self.data[key][self.curr_key][fr_start:fr_end]
        curr_qpos = curr_qpos[fr_start:fr_end]
        sample["seq_name"] = self.curr_key
        sample["has_obj"] = sample["obj_pose"].shape != curr_qpos.shape
        sample["num_obj"] = sample["obj_pose"].shape[1] // 7 if sample["has_obj"] else 0

        return sample

    def set_singles(self, seq_name):
        self.data_keys = [seq_name]

    def set_seq_counter(self, idx):
        self.seq_counter = idx

    def iter_seq(self):
        self.iter_keys = self.data_keys
        self.curr_key = self.iter_keys[self.seq_counter % len(self.iter_keys)]
        self.seq_counter += 1
        return self.get_sample_from_key(self.curr_key, full_sample=True, fr_start=0)

    def get_len(self):
        return len(self.data_keys)

    def remove_base_rot(self, quat):
        return quaternion_multiply(quat, quaternion_inverse(self.base_rot))

    def match_heading_and_pos(self, qpos_1, qpos_2):
        posxy_1 = qpos_1[:2]
        qpos_1_quat = self.remove_base_rot(qpos_1[3:7])
        qpos_2_quat = self.remove_base_rot(qpos_2[3:7])
        heading_1 = get_heading_q(qpos_1_quat)
        qpos_2[3:7] = de_heading(qpos_2[3:7])
        qpos_2[3:7] = quaternion_multiply(heading_1, qpos_2[3:7])
        qpos_2[:2] = posxy_1
        return qpos_2

    def random_heading_seq(self, qposes):
        rand_heading = random_heading()
        qposes = qposes.copy()
        qpose_start = qposes[0].copy()
        q_dehead = de_heading(qpose_start[3:7])
        q_target = quaternion_multiply(rand_heading, q_dehead)
        quat_delta = quaternion_multiply(q_target, quaternion_inverse(qpose_start[3:7]))
        qposes[:, 3:7] = quaternion_multiply_batch(
            np.repeat(
                quat_delta[
                    None,
                ],
                qposes.shape[0],
                axis=0,
            ),
            qposes[:, 3:7],
        )

        start_xy = qposes[0, :2].copy()
        # qposes[:,:2] -= qposes[0, :2]
        qposes[:, :3] = transform_vec_batch(
            qposes[:, :3], quaternion_inverse(quat_delta)
        ).T
        # qposes[:,:2] += start_xy

        return qposes

    def random_heading(self, qpos):
        rand_heading = random_heading()
        qpos[3:7] = de_heading(qpos[3:7])
        qpos[3:7] = quaternion_multiply(rand_heading, qpos[3:7])
        return qpos

    def iter_generator(self, batch_size=8, num_workers=8):
        loader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return loader


if __name__ == "__main__":
    np.random.seed(0)
    data_specs = {
        "dataset_name": "amass_rf",
        "file_path": "/insert_directory_here/amass_copycat_take1_test.pkl",
        "flip_cnd": 0,
        "has_smpl_root": True,
        "traj_dim": 144,
        "t_min": 90,
        "nc": 2,
        "mode": "all",
        "load_class": -1,
        "root_dim": 6,
        "flip_time": False,
    }
    smpl_viewer = SMPL_M_Viewer()
    model_file = f"assets/mujoco_models/humanoid_smpl_neutral_mesh.xml"
    dataset = DatasetAMASSSingle(data_specs, model_file=model_file)
    # for i in range(10):
    #     generator = dataset.sampling_generator(num_samples=5000)
    #     for data in generator:
    #         qpos = data["qpos"]
    #         smpl_viewer.set_qpose(qpos)
    #         smpl_viewer.show_pose()

    data = dataset.sample_seq()
    qpos = data["qpos"]
    smpl_viewer.set_qpose(qpos)
    smpl_viewer.show_pose(loop=True)