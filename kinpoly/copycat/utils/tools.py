from readline import insert_text
from copycat.khrylib.utils import *
from collections import defaultdict

def get_expert(expert_qpos, expert_meta, env):
    old_state = env.sim.get_state()
    expert = defaultdict(list)
    expert['qpos'] = expert_qpos
    expert['meta'] = expert_meta
    feat_keys = {'qvel', 'rlinv', 'rlinv_local', 'rangv', 'rq_rmh',
                 'com', "body_com", 'head_pose', 'ee_pos', 'ee_wpos', 'bquat', 'bangvel', 'wbpos', "wbquat"}

    for i in range(expert_qpos.shape[0]):
        qpos = expert_qpos[i]
        env.data.qpos[:76] = qpos # 3(trans) + 4(root quat) + 69(other joints' euler angles with respect to parent)
        env.sim.forward()
        rq_rmh = de_heading(qpos[3:7]) # root quaternion, remove heading. 4
        ee_pos = env.get_ee_pos(env.cfg.obs_coord) # obs_coord: heading
        wbpos = env.get_wbody_pos()
        wbquat = env.get_wbody_quat()
        
        ee_wpos = env.get_ee_pos(None)
        bquat = env.get_body_quat() # current pose (body) in quaternion
        com = env.get_com()
        head_pose = env.get_head().copy()
        body_com = env.get_body_com()
        
        if i > 0:
            prev_qpos = expert_qpos[i - 1]
            qvel = get_qvel_fd_new(prev_qpos, qpos, env.dt)
            qvel = qvel.clip(-10.0, 10.0)
            rlinv = qvel[:3].copy() # root position linear velocity 
            rlinv_local = transform_vec(qvel[:3].copy(), qpos[3:7], env.cfg.obs_coord) # obs_coord: heading 
            rangv = qvel[3:6].copy() # root joint angular velocity 
            expert['qvel'].append(qvel)
            expert['rlinv'].append(rlinv)
            expert['rlinv_local'].append(rlinv_local)
            expert['rangv'].append(rangv)
        
        expert['wbquat'].append(wbquat)
        expert['wbpos'].append(wbpos)
        expert['ee_pos'].append(ee_pos)
        expert['ee_wpos'].append(ee_wpos)
        expert['bquat'].append(bquat)
        expert['com'].append(com)
        expert['body_com'].append(body_com)
        expert['head_pose'].append(head_pose)
        expert['rq_rmh'].append(rq_rmh)

    expert['qvel'].insert(0, expert['qvel'][0].copy())
    expert['rlinv'].insert(0, expert['rlinv'][0].copy())
    expert['rlinv_local'].insert(0, expert['rlinv_local'][0].copy())
    expert['rangv'].insert(0, expert['rangv'][0].copy())

    # get expert body quaternions
    for i in range(1, expert_qpos.shape[0]):
        bangvel = get_angvel_fd(expert['bquat'][i - 1], expert['bquat'][i], env.dt)
        expert['bangvel'].append(bangvel) # body joint angular velocity 
    
    expert['bangvel'].insert(0, expert['bangvel'][0].copy())

    for key in feat_keys:
        expert[key] = np.vstack(expert[key])

    expert['len'] = expert['qpos'].shape[0]
    expert['height_lb'] = expert['qpos'][:, 2].min()
    expert['head_height_lb'] = expert['head_pose'][:, 2].min()
    if expert_meta['cyclic']:
        expert['init_heading'] = get_heading_q(expert_qpos[0, 3:7])
        expert['init_pos'] = expert_qpos[0, :3].copy()
    env.sim.set_state(old_state)
    env.sim.forward()
    return expert


def get_expert_v2(expert_qpos, expert_meta, env):
    old_state = env.sim.get_state()
    expert = defaultdict(list)
    expert['qpos'] = expert_qpos
    expert['meta'] = expert_meta
    feat_keys = {'qvel', 'rlinv', 'rlinv_local', 'rangv', 'rq_rmh',
                 'com', "body_com", 'head_pose', 'ee_pos', 'ee_wpos', 'bquat', 'bangvel', 'wbpos', "wbquat"}

    for i in range(expert_qpos.shape[0]):
        qpos = expert_qpos[i]
        env.data.qpos[:76] = qpos # 3(trans) + 4(root quat) + 69(other joints' euler angles with respect to parent)
        env.sim.forward()
        rq_rmh = de_heading(qpos[3:7]) # root quaternion, remove heading. 4
        ee_pos = env.get_ee_pos(env.cfg.obs_coord) # obs_coord: heading
        wbpos = env.get_wbody_pos()
        wbquat = env.get_wbody_quat()
        
        ee_wpos = env.get_ee_pos(None)
        bquat = env.get_body_quat() # current pose (body) in quaternion
        com = env.get_com()
        head_pose = env.get_head().copy()
        body_com = env.get_body_com()
        
        if i > 0:
            prev_qpos = expert_qpos[i - 1]
            qvel = get_qvel_fd_new(prev_qpos, qpos, env.dt)
            qvel = qvel.clip(-10.0, 10.0)
            rlinv = qvel[:3].copy() # root position linear velocity 
            rlinv_local = transform_vec(qvel[:3].copy(), qpos[3:7], env.cfg.obs_coord) # obs_coord: heading 
            rangv = qvel[3:6].copy() # root joint angular velocity 
            expert['qvel'].append(qvel)
            expert['rlinv'].append(rlinv)
            expert['rlinv_local'].append(rlinv_local)
            expert['rangv'].append(rangv)
        
            expert['wbquat'].append(wbquat)
            expert['wbpos'].append(wbpos)
            expert['ee_pos'].append(ee_pos)
            expert['ee_wpos'].append(ee_wpos)
        
            expert['com'].append(com)
            expert['body_com'].append(body_com)
            expert['head_pose'].append(head_pose)
            expert['rq_rmh'].append(rq_rmh)
        
        expert['bquat'].append(bquat)

    # get expert body quaternions
    for i in range(1, expert_qpos.shape[0]):
        bangvel = get_angvel_fd(expert['bquat'][i - 1], expert['bquat'][i], env.dt)
        expert['bangvel'].append(bangvel) # body joint angular velocity 

    expert['bquat'] = expert['bquat'][1:]
    expert['qpos'] = expert['qpos'][1:]
    
    for key in feat_keys:
        expert[key] = np.vstack(expert[key])

    expert['len'] = expert['qpos'].shape[0]
    expert['height_lb'] = expert['qpos'][:, 2].min()
    expert['head_height_lb'] = expert['head_pose'][:, 2].min()
    if expert_meta['cyclic']:
        expert['init_heading'] = get_heading_q(expert_qpos[0, 3:7])
        expert['init_pos'] = expert_qpos[0, :3].copy()
    env.sim.set_state(old_state)
    env.sim.forward()
    return expert