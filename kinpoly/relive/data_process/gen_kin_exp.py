import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import sys
import time
import pickle
sys.path.append(os.getcwd())

from relive.utils import *
#from relive.envs.humanoid_v1 import HumanoidEnv
from relive.envs.humanoid_v2 import HumanoidEnv
from relive.envs.visual.humanoid_vis import HumanoidVisEnv

from relive.utils.egomimic_config import Config as EgoConfig


from relive.utils.math_utils import get_heading_q
import matplotlib.pyplot as plt

def get_head(head_name):
    bone_id = env_vis.model._body_name2id[head_name]
    head_pos = env_vis.data.body_xpos[bone_id]
    head_quat = env_vis.data.body_xquat[bone_id]
    return head_pos, head_quat

def get_vel(prev_pos, prev_quat, curr_pos, curr_quat):
    v = (curr_pos - prev_pos) / dt
    qrel = quaternion_multiply(curr_quat, quaternion_inverse(prev_quat))
    axis, angle = rotation_from_quaternion(qrel, True)
    
    if angle > np.pi: # -180 < angle < 180
        angle -= 2 * np.pi # 
    elif angle < -np.pi:
        angle += 2 * np.pi
    rv = (axis * angle) / dt
    rv = transform_vec(rv, curr_quat, 'root')
    rlinv_local = transform_vec(v, curr_quat, 'heading')
    return rlinv_local, rv

def get_expert(reg_qpos, lb, ub, take):
    from scipy.special import softmax
    from scipy.signal import medfilt
    expert = {'qpos': reg_qpos}
    nq = 59 + off_obj_qpos
    
    linv, rlinv = [], []
    for i in range(reg_qpos.shape[0]):
        qpos = reg_qpos[i]
        env_vis.data.qpos[:nq] = qpos
        env_vis.sim.forward()
        if args.render:
            env_vis.render()

        curr_pos, curr_quat = get_head('Head')
        

        if i > 0:
            rl, ra = get_vel(prev_pos, prev_quat, curr_pos, curr_quat)
            linv.append(rl)
            rlinv.append(ra)
        prev_pos, prev_quat = curr_pos.copy(), curr_quat.copy()

    linv.insert(0, linv[0].copy())
    rlinv.insert(0, rlinv[0].copy())
    linv = np.array(linv)
    rlinv = np.array(rlinv)

    return linv, rlinv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-id', default='meta_all')
    parser.add_argument('--cfg', default='all_traj_3')
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--model-id', default='1213')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--test-feat', type=str, default=None)
    parser.add_argument('--action', default='sit')
    args = parser.parse_args()

    # cfg_dict = {
    #     'meta_id': args.meta_id,
    #     'model_id': args.model_id,
    #     'obs_coord': 'heading',
    # }
    # cfg = EgoConfig(args.action, None, create_dirs=False, cfg_dict=cfg_dict)
    cfg = EgoConfig(args.action, args.cfg, create_dirs=False)
    obj_num     = len(cfg.object)
    off_obj_qpos = obj_num * 7
    dt          = 1 / cfg.meta['capture']['fps']

    vis_model_file = 'assets/mujoco_models/%s/%s/humanoid_vis_double_v1.xml' % (args.model_id, args.action)
    env_vis = HumanoidVisEnv(vis_model_file, 10, focus=True)



    if args.test_feat is None:
        sr_res_path = './results/%s/statereg/%s/results/iter_%04d_all.p' % (args.action, args.cfg, args.iter)
        sr_out_path = './results/%s/statereg/%s/results/iter_%04d_exp.p' % (args.action, args.cfg, args.iter)
    else:
        sr_res_path = './results/%s/statereg/%s/results/iter_%04d_%s.p' % (args.action, args.cfg, args.iter, args.test_feat)

    print(sr_res_path)
    sr_res, meta = pickle.load(open(sr_res_path, 'rb'))
    traj_pred = sr_res['traj_pred']
    sr_res['linv_local'] = {}
    sr_res['rangv_local'] = {}
    num_sample = 0
    expert_dict = {}
    dlist = []
    rdlist = []
    for take in traj_pred.keys():
        qpos = traj_pred[take]
        lb = 0
        ub = qpos.shape[0]
        linv, rangv = get_expert(qpos, lb, ub, take)
        print(linv.shape)
        print(qpos.shape)
        sr_res['linv_local'][take] = linv
        sr_res['rangv_local'][take] = rangv
    print(sr_out_path)
    pickle.dump((sr_res, meta), open(sr_out_path, 'wb'))

