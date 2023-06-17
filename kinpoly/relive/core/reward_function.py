from relive.utils import *
from relive.utils.flags import flags


def quat_space_reward_v2(env, state, action, info):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_c, w_r = ws.get('w_p', 0.5), ws.get('w_v', 0.05), ws.get('w_e', 0.15), ws.get('w_c', 0.1), ws.get('w_r', 0.2)
    k_p, k_v, k_e, k_c, k_r = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20), ws.get('k_c', 1000), ws.get('k_r', 1.0)
    w_rq, w_rlinv, w_rangv = ws.get('w_rq', 2.0), ws.get('w_rlinv', 1.0), ws.get('w_rangv', 0.1)
    v_ord = ws.get('v_ord', 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)
    cur_rlinv_local = cur_qvel[:3]
    cur_rangv = cur_qvel[3:6]
    cur_rq_rmh = de_heading(cur_qpos[3:7])
    cur_ee = env.get_ee_pos(cfg.obs_coord)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_com = env.get_com()
    # expert
    e_rlinv_local = env.get_expert_attr('rlinv_local', ind)
    e_rangv = env.get_expert_attr('rangv', ind)
    e_rq_rmh = env.get_expert_attr('rq_rmh', ind)
    e_ee = env.get_expert_attr('ee_pos', ind)
    e_bquat = env.get_expert_attr('bquat', ind)
    e_bangvel = env.get_expert_attr('bangvel', ind)
    e_com = env.get_expert_attr('com', ind)
    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))    # ignore root
    pose_diff *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel[3:] - e_bangvel[3:], ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # com reward
    com_dist = cur_com[2] - e_com[2]
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # root reward
    rq_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]
    rlinv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)
    rangv_dist = np.linalg.norm(cur_rangv - e_rangv)
    root_dist = w_rq * rq_dist + w_rlinv * rlinv_dist + w_rangv * rangv_dist
    root_reward = math.exp(-k_r * (root_dist ** 2))
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_c * com_reward + w_r * root_reward
    reward /= w_p + w_v + w_e + w_c + w_r
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_reward, vel_reward, ee_reward, com_reward, root_reward])

def quat_space_reward_v3(env, state, action, info):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_rp, w_rv = ws.get('w_p', 0.5), ws.get('w_v', 0.1), ws.get('w_e', 0.2), ws.get('w_rp', 0.1), ws.get('w_rv', 0.1)
    k_p, k_v, k_e = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20)
    k_rh, k_rq, k_rl, k_ra = ws.get('k_rh', 300), ws.get('k_rq', 300), ws.get('k_rl', 5.0), ws.get('k_ra', 0.5)
    v_ord = ws.get('v_ord', 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)
    cur_rlinv_local = cur_qvel[:3]
    cur_rangv = cur_qvel[3:6]
    cur_rq_rmh = de_heading(cur_qpos[3:7])
    cur_ee = env.get_ee_pos(cfg.obs_coord)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # expert
    e_qpos = env.get_expert_attr('qpos', ind)
    e_rlinv_local = env.get_expert_attr('rlinv_local', ind)
    e_rangv = env.get_expert_attr('rangv', ind)
    e_rq_rmh = env.get_expert_attr('rq_rmh', ind)
    e_ee = env.get_expert_attr('ee_pos', ind)
    e_bquat = env.get_expert_attr('bquat', ind)
    e_bangvel = env.get_expert_attr('bangvel', ind)
    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))    # ignore root
    pose_diff *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel[3:] - e_bangvel[3:], ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # root position reward
    root_height_dist = cur_qpos[2] - e_qpos[2]
    root_quat_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]
    root_pose_reward = math.exp(-k_rh * (root_height_dist ** 2) - k_rq * (root_quat_dist ** 2))
    # root velocity reward
    root_linv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)
    root_angv_dist = np.linalg.norm(cur_rangv - e_rangv)
    root_vel_reward = math.exp(-k_rl * (root_linv_dist ** 2) - k_ra * (root_angv_dist ** 2))
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_rp * root_pose_reward + w_rv * root_vel_reward
    reward /= w_p + w_v + w_e + w_rp + w_rv
    if ws.get('decay', False):
        reward *= 1.0 - t / cfg.env_episode_len
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_reward, vel_reward, ee_reward, root_pose_reward, root_vel_reward])

def deep_mimic_reward(env, state, action, info):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_c = ws.get('w_p', 0.65), ws.get('w_v', 0.1), ws.get('w_e', 0.15), ws.get('w_c', 0.1)
    k_p, k_v, k_e, k_c = ws.get('k_p', 2), ws.get('k_v', 0.1), ws.get('k_e', 10), ws.get('k_c', 10)
    v_ord = ws.get('v_ord', 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_rpos = cur_qpos[env.off_obj_qpos:env.off_obj_qpos+3]
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # expert
    e_rpos = env.get_expert_attr('qpos', ind)[env.off_obj_qpos:env.off_obj_qpos+3]
    e_ee = env.get_expert_attr('ee_wpos', ind).copy()
    e_bquat = env.get_expert_attr('bquat', ind).copy()
    e_bangvel = env.get_expert_attr('bangvel', ind)

    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat, e_bquat))
    pose_dist = np.linalg.norm(pose_diff)
    pose_diff[1:] *= cfg.b_diffw
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    #root reward
    root_dist = np.linalg.norm(cur_rpos - e_rpos)
    root_reward = math.exp(-k_c * (root_dist ** 2))
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_c * root_reward
    reward /= w_p + w_v + w_e + w_c
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_reward, vel_reward, ee_reward, root_reward])


def deep_mimic_reward_v2(env, state, action, info):
    # reward coefficients
    

    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_rp, w_rq = ws.get('w_p', 0.65), ws.get('w_v', 0.1), ws.get('w_e', 0.15), ws.get('w_rp', 0.1), ws.get('w_rq', 0.1)
    k_p, k_v, k_e, k_rp, k_rq = ws.get('k_p', 2), ws.get('k_v', 0.1), ws.get('k_e', 10), ws.get('k_rp', 10), ws.get('k_rq', 10)

    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_rpos  = cur_qpos[env.off_obj_qpos  :env.off_obj_qpos+3]
    cur_rquat = cur_qpos[env.off_obj_qpos+3:env.off_obj_qpos+7]
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # expert
    e_rpos  = env.get_expert_attr('qpos', ind)[env.off_obj_qpos  :env.off_obj_qpos+3]
    e_rquat = env.get_expert_attr('qpos', ind)[env.off_obj_qpos+3:env.off_obj_qpos+7]
    e_ee = env.get_expert_attr('ee_wpos', ind).copy()
    e_bquat = env.get_expert_attr('bquat', ind).copy()
    e_bangvel = env.get_expert_attr('bangvel', ind)

    # pose reward (exclude root)
    #pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))
    pose_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    #root pos reward
    rp_dist = np.linalg.norm(cur_rpos - e_rpos)
    rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    #root quat reward
    rq_dist = multi_quat_norm_v2(multi_quat_diff(cur_bquat[:4], e_bquat[:4]))
    rq_reward = math.exp(-k_rq * (rq_dist ** 2))

    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_rp * rp_reward + w_rq * rq_reward
    reward /= (w_p + w_v + w_e + w_rp + w_rq)

    return reward, np.array([pose_reward, vel_reward, ee_reward, rp_reward, rq_reward])

def deep_mimic_reward_v2_vf(env, state, action, info):
    # reward coefficients
    
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_rp, w_rq, w_vf = ws.get('w_p', 0.65), ws.get('w_v', 0.1), ws.get('w_e', 0.15), ws.get('w_rp', 0.1), ws.get('w_rq', 0.1), ws.get('w_vf', 0.1)
    k_p, k_v, k_e, k_rp, k_rq, k_vf= ws.get('k_p', 2), ws.get('k_v', 0.1), ws.get('k_e', 10), ws.get('k_rp', 10), ws.get('k_rq', 10), ws.get('k_vf', 10)

    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_rpos  = cur_qpos[env.off_obj_qpos  :env.off_obj_qpos+3]
    cur_rquat = cur_qpos[env.off_obj_qpos+3:env.off_obj_qpos+7]
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # expert
    e_rpos  = env.get_expert_attr('qpos', ind)[env.off_obj_qpos  :env.off_obj_qpos+3]
    e_rquat = env.get_expert_attr('qpos', ind)[env.off_obj_qpos+3:env.off_obj_qpos+7]
    e_ee = env.get_expert_attr('ee_wpos', ind).copy()
    e_bquat = env.get_expert_attr('bquat', ind).copy()
    e_bangvel = env.get_expert_attr('bangvel', ind)

    # pose reward (exclude root)
    #pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))
    pose_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    #root pos reward
    rp_dist = np.linalg.norm(cur_rpos - e_rpos)
    rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    #root quat reward
    rq_dist = multi_quat_norm_v2(multi_quat_diff(cur_bquat[:4], e_bquat[:4]))
    rq_reward = math.exp(-k_rq * (rq_dist ** 2))

    if env.cfg.action_v == 2:
        vf = action[-env.vf_dim:]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    elif env.cfg.action_v == 3:
        vf = action[-6:]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        print("action version not supported")
        exit()


    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_rp * rp_reward + w_rq * rq_reward + w_vf * vf_reward
    reward /= (w_p + w_v + w_e + w_rp + w_rq + w_vf)

    return reward, np.array([pose_reward, vel_reward, ee_reward, rp_reward, rq_reward, vf_reward])


def deep_mimic_reward_v2_vf_vq(env, state, action, info):
    # reward coefficients
    

    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_rp, w_rq = ws.get('w_p', 0.65), ws.get('w_v', 0.1), ws.get('w_e', 0.15), ws.get('w_rp', 0.1), ws.get('w_rq', 0.1)
    k_p, k_v, k_e, k_rp, k_rq = ws.get('k_p', 2), ws.get('k_v', 0.1), ws.get('k_e', 10), ws.get('k_rp', 10), ws.get('k_rq', 10)

    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_rpos  = cur_qpos[env.off_obj_qpos  :env.off_obj_qpos+3]
    cur_rquat = cur_qpos[env.off_obj_qpos+3:env.off_obj_qpos+7]
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # expert
    e_rpos  = env.get_expert_attr('qpos', ind)[env.off_obj_qpos  :env.off_obj_qpos+3]
    e_rquat = env.get_expert_attr('qpos', ind)[env.off_obj_qpos+3:env.off_obj_qpos+7]
    e_ee = env.get_expert_attr('ee_wpos', ind).copy()
    e_bquat = env.get_expert_attr('bquat', ind).copy()
    e_bangvel = env.get_expert_attr('bangvel', ind)

    # pose reward (exclude root)
    #pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))
    pose_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    #root pos reward
    rp_dist = np.linalg.norm(cur_rpos - e_rpos)
    rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    #root quat reward
    rq_dist = multi_quat_norm_v2(multi_quat_diff(cur_bquat[:4], e_bquat[:4]))
    rq_reward = math.exp(-k_rq * (rq_dist ** 2))

    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_rp * rp_reward + w_rq * rq_reward
    reward /= (w_p + w_v + w_e + w_rp + w_rq)

    return reward, np.array([pose_reward, vel_reward, ee_reward, rp_reward, rq_reward])



def multiplicable_reward(env, state, action, info):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    k_p, k_v, k_e, k_rp, k_rq, k_rl, k_ra = ws.get('k_p', 2), ws.get('k_v', 0.1), ws.get('k_e', 10), ws.get('k_rp', 10), ws.get('k_rq', 10), ws.get('k_rl', 5.0), ws.get('k_ra', 0.5)


    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd(prev_qpos, cur_qpos, env.dt)
    cur_rlinv = cur_qvel[:3]
    cur_rangv = cur_qvel[3:6]
    cur_rpos  = cur_qpos[env.off_obj_qpos  :env.off_obj_qpos+3]
    cur_rquat = cur_qpos[env.off_obj_qpos+3:env.off_obj_qpos+7]
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)

    # expert
    t = env.cur_t
    ind = env.get_expert_index(t)
    e_rpos  = env.get_expert_attr('qpos', ind)[env.off_obj_qpos  :env.off_obj_qpos+3]
    e_rquat = env.get_expert_attr('qpos', ind)[env.off_obj_qpos+3:env.off_obj_qpos+7]
    e_ee = env.get_expert_attr('ee_wpos', ind).copy()
    e_bquat = env.get_expert_attr('bquat', ind).copy()
    e_bangvel = env.get_expert_attr('bangvel', ind).copy()
    e_rlinv = env.get_expert_attr('rlinv', ind).copy()
    e_rangv = env.get_expert_attr('rangv', ind).copy()

    # pose reward (exclude root)
    #pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    #root pos reward
    rp_dist = np.linalg.norm(cur_rpos - e_rpos)
    rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    #root quat reward
    rq_dist = multi_quat_norm(multi_quat_diff(cur_rquat, e_rquat))
    rq_reward = math.exp(-k_rq * (rq_dist ** 2))
    # root velocity reward
    #root_linv_dist = np.linalg.norm(cur_rlinv - e_rlinv)
    #root_angv_dist = np.linalg.norm(cur_rangv - e_rangv)
    #root_vel_reward = math.exp(-k_rl * (root_linv_dist ** 2) - k_ra * (root_angv_dist ** 2))
    #rq_dist = multi_quat_norm(multi_quat_diff(cur_rquat, e_rquat))


    # overall reward
    reward = pose_reward * vel_reward * ee_reward * rp_reward * rq_reward
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_reward, vel_reward, ee_reward, rp_reward, rq_reward])

def local_world_reward_v1(env, state, action, info):
    expert = env.expert
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_we, w_c, w_r = ws.get('w_p', 0.4), ws.get('w_v', 0.05), ws.get('w_e', 0.15), ws.get('w_we', 0.1), ws.get('w_c', 0.1), ws.get('w_r', 0.2)
    k_p, k_v, k_e, k_we, k_c, k_r = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20), ws.get('k_we', 20), ws.get('k_c', 1000), ws.get('k_r', 1.0)
    w_rq, w_rlinv, w_rangv = ws.get('w_rq', 2.0), ws.get('w_rlinv', 1.0), ws.get('w_rangv', 0.1)
    v_ord = ws.get('v_ord', 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)
    cur_rlinv_local = cur_qvel[:3]
    cur_rangv = cur_qvel[3:6]
    cur_rq_rmh = de_heading(cur_qpos[3:7])
    cur_ee = env.get_ee_pos(cfg.obs_coord)
    cur_wee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_com = env.get_com()
    # expert
    e_rlinv_local = env.get_expert_attr('rlinv_local', ind)
    e_rangv = env.get_expert_attr('rangv', ind)
    e_rq_rmh = env.get_expert_attr('rq_rmh', ind)
    e_ee = env.get_expert_attr('ee_pos', ind)
    e_wee = env.get_expert_attr('ee_wpos', ind).copy()
    e_bquat = env.get_expert_attr('bquat', ind)
    e_bangvel = env.get_expert_attr('bangvel', ind)
    e_com = env.get_expert_attr('com', ind).copy()
    # sync expert
    start_pos = expert['start_pos']
    rel_h = expert['rel_heading']
    sim_pos = expert['sim_pos']
    e_com = quat_mul_vec(rel_h, e_com - start_pos) + sim_pos
    for i in range(e_ee.shape[0] // 3):
        e_wee[3*i: 3*i+3] = quat_mul_vec(rel_h, e_wee[3*i: 3*i+3] - start_pos) + sim_pos

    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))    # ignore root
    pose_diff *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel[3:] - e_bangvel[3:], ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # world ee reward
    wee_dist = np.linalg.norm(cur_wee - e_wee)
    wee_reward = math.exp(-k_we * (wee_dist ** 2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # root reward
    rq_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]
    rlinv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)
    rangv_dist = np.linalg.norm(cur_rangv - e_rangv)
    root_dist = w_rq * rq_dist + w_rlinv * rlinv_dist + w_rangv * rangv_dist
    root_reward = math.exp(-k_r * (root_dist ** 2))
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_we * wee_reward + w_c * com_reward + w_r * root_reward
    reward /= w_p + w_v + w_e + w_we + w_c + w_r
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_reward, vel_reward, ee_reward, wee_reward, com_reward, root_reward])


def local_world_reward_v2(env, state, action, info):
    expert = env.expert
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_h, w_c, w_r = ws.get('w_p', 0.4), ws.get('w_v', 0.05), ws.get('w_e', 0.15), ws.get('w_h', 0.1), ws.get('w_c', 0.1), ws.get('w_r', 0.2)
    k_p, k_v, k_e, k_h, k_c, k_r = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20), ws.get('k_h', 20), ws.get('k_c', 1000), ws.get('k_r', 1.0)
    w_rq, w_rlinv, w_rangv = ws.get('w_rq', 2.0), ws.get('w_rlinv', 1.0), ws.get('w_rangv', 0.1)
    v_ord = ws.get('v_ord', 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)
    cur_rlinv_local = cur_qvel[:3]
    cur_rangv = cur_qvel[3:6]
    cur_rq_rmh = de_heading(cur_qpos[3:7])
    cur_ee = env.get_ee_pos(cfg.obs_coord)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_com = env.get_com()
    cur_heading = get_heading(cur_qpos[3:7])
    # expert
    e_rq = env.get_expert_attr('qpos', ind)[3:7]
    e_rlinv_local = env.get_expert_attr('rlinv_local', ind)
    e_rangv = env.get_expert_attr('rangv', ind)
    e_rq_rmh = env.get_expert_attr('rq_rmh', ind)
    e_ee = env.get_expert_attr('ee_pos', ind)
    e_bquat = env.get_expert_attr('bquat', ind)
    e_bangvel = env.get_expert_attr('bangvel', ind)
    e_com = env.get_expert_attr('com', ind).copy()
    # sync expert
    start_pos = expert['start_pos']
    rel_h = expert['rel_heading']
    sim_pos = expert['sim_pos']
    e_com = quat_mul_vec(rel_h, e_com - start_pos) + sim_pos
    e_rq = quaternion_multiply(rel_h, e_rq)
    e_heading = get_heading(e_rq)

    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))    # ignore root
    pose_diff *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel[3:] - e_bangvel[3:], ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # heading reward
    h_dist = cur_heading - e_heading
    h_reward = math.exp(-k_h * (h_dist ** 2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # root reward
    rq_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]
    rlinv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)
    rangv_dist = np.linalg.norm(cur_rangv - e_rangv)
    root_dist = w_rq * rq_dist + w_rlinv * rlinv_dist + w_rangv * rangv_dist
    root_reward = math.exp(-k_r * (root_dist ** 2))
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_h * h_reward + w_c * com_reward + w_r * root_reward
    reward /= w_p + w_v + w_e + w_h + w_c + w_r
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_reward, vel_reward, ee_reward, h_reward, com_reward, root_reward])


def local_world_reward_v3(env, state, action, info):
    expert = env.expert
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_h, w_c, w_r = ws.get('w_p', 0.4), ws.get('w_v', 0.05), ws.get('w_e', 0.15), ws.get('w_h', 0.1), ws.get('w_c', 0.1), ws.get('w_r', 0.2)
    k_p, k_v, k_e, k_h, k_c, k_r = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20), ws.get('k_h', 20), ws.get('k_c', 1000), ws.get('k_r', 1.0)
    w_rq, w_rlinv, w_rangv = ws.get('w_rq', 2.0), ws.get('w_rlinv', 1.0), ws.get('w_rangv', 0.1)
    v_ord = ws.get('v_ord', 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)
    cur_rlinv_local = cur_qvel[:3]
    cur_rangv = cur_qvel[3:6]
    cur_rq_rmh = de_heading(cur_qpos[3:7])
    cur_ee = env.get_ee_pos(cfg.obs_coord)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_com = env.get_com()
    cur_heading = get_heading(cur_qpos[3:7])
    # expert
    e_rq = env.get_expert_attr('qpos', ind)[3:7]
    e_rlinv_local = env.get_expert_attr('rlinv_local', ind)
    e_rangv = env.get_expert_attr('rangv', ind)
    e_rq_rmh = env.get_expert_attr('rq_rmh', ind)
    e_ee = env.get_expert_attr('ee_pos', ind)
    e_bquat = env.get_expert_attr('bquat', ind)
    e_bangvel = env.get_expert_attr('bangvel', ind)
    e_com = env.get_expert_attr('com', ind).copy()
    # sync expert
    start_pos = expert['start_pos']
    rel_h = expert['rel_heading']
    sim_pos = expert['sim_pos']
    e_com = quat_mul_vec(rel_h, e_com - start_pos) + sim_pos
    e_rq = quaternion_multiply(rel_h, e_rq)
    e_heading = get_heading(e_rq)

    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))    # ignore root
    pose_diff *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel[3:] - e_bangvel[3:], ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # heading reward
    h_dist = cur_heading - e_heading
    h_reward = math.exp(-k_h * (h_dist ** 2))
    # com reward
    com_dist = cur_com[2] - e_com[2]
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # root reward
    rq_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]
    rlinv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)
    rangv_dist = np.linalg.norm(cur_rangv - e_rangv)
    root_dist = w_rq * rq_dist + w_rlinv * rlinv_dist + w_rangv * rangv_dist
    root_reward = math.exp(-k_r * (root_dist ** 2))
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_h * h_reward + w_c * com_reward + w_r * root_reward
    reward /= w_p + w_v + w_e + w_h + w_c + w_r
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_reward, vel_reward, ee_reward, h_reward, com_reward, root_reward])


def world_quat_space_reward(env, state, action, info):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_c = ws.get('w_p', 0.6), ws.get('w_v', 0.1), ws.get('w_e', 0.2), ws.get('w_c', 0.1)
    k_p, k_v, k_e, k_c = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20), ws.get('k_c', 1000)
    v_ord = ws.get('v_ord', 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    # learner
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_com = env.get_com()
    # expert
    e_ee = env.get_expert_attr('ee_wpos', ind).copy()
    e_com = env.get_expert_attr('com', ind).copy()
    e_bquat = env.get_expert_attr('bquat', ind).copy()
    e_bangvel = env.get_expert_attr('bangvel', ind)
    expert = env.expert
    # sync expert
    start_pos = expert['start_pos']
    rel_h = expert['rel_heading']
    sim_pos = expert['sim_pos']
    e_bquat[:4] = quaternion_multiply(rel_h, e_bquat[:4])
    e_com = quat_mul_vec(rel_h, e_com - start_pos) + sim_pos
    for i in range(e_ee.shape[0] // 3):
        e_ee[3*i: 3*i+3] = quat_mul_vec(rel_h, e_ee[3*i: 3*i+3] - start_pos) + sim_pos

    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff[1:] *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel, ord=v_ord)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_c * com_reward
    reward /= w_p + w_v + w_e + w_c
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_reward, vel_reward, ee_reward, com_reward])


def world_quat_space_reward_v2(env, state, action, info):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_c, w_r = ws.get('w_p', 0.3), ws.get('w_v', 0.1), ws.get('w_e', 0.3), ws.get('w_c', 0.1), ws.get('w_r', 0.2)
    k_p, k_v, k_e, k_c, k_r = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20), ws.get('k_c', 1000), ws.get('k_r', 1.0)
    w_rpos, w_rq, w_rlinv, w_rangv = ws.get('w_rpos', 5.0), ws.get('w_rq', 2.0), ws.get('w_rlinv', 1.0), ws.get('w_rangv', 0.1)
    v_ord = ws.get('v_ord', 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd(prev_qpos, cur_qpos, env.dt)
    cur_rlinv = cur_qvel[:3]
    cur_rangv = cur_qvel[3:6]
    cur_rpos = cur_qpos[:3]
    cur_rq = cur_qpos[3:7]
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_com = env.get_com()
    # expert
    e_qpos = env.get_expert_attr('qpos', ind)
    e_rlinv = env.get_expert_attr('rlinv', ind)
    e_rangv = env.get_expert_attr('rangv', ind)
    e_rpos = e_qpos[:3]
    e_rq = e_qpos[3:7]
    e_ee = env.get_expert_attr('ee_wpos', ind).copy()
    e_com = env.get_expert_attr('com', ind).copy()
    e_bquat = env.get_expert_attr('bquat', ind)
    e_bangvel = env.get_expert_attr('bangvel', ind)
    expert = env.expert
    start_pos = expert['start_pos']
    rel_h = expert['rel_heading']
    sim_pos = expert['sim_pos']
    e_rq = quaternion_multiply(rel_h, e_rq)
    e_rlinv = quat_mul_vec(rel_h, e_rlinv)
    e_com = quat_mul_vec(rel_h, e_com - start_pos) + sim_pos
    for i in range(e_ee.shape[0] // 3):
        e_ee[3*i: 3*i+3] = quat_mul_vec(rel_h, e_ee[3*i: 3*i+3] - start_pos) + sim_pos

    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))  # ignore root
    pose_diff *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel[3:] - e_bangvel[3:], ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # root reward
    rpos_dist = np.linalg.norm(cur_rpos - e_rpos)
    rq_dist = multi_quat_norm(multi_quat_diff(cur_rq, e_rq))[0]
    rlinv_dist = np.linalg.norm(cur_rlinv - e_rlinv)
    rangv_dist = np.linalg.norm(cur_rangv - e_rangv)
    root_dist = w_rpos * rpos_dist + w_rq * rq_dist + w_rlinv * rlinv_dist + w_rangv * rangv_dist
    root_reward = math.exp(-k_r * (root_dist ** 2))
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_c * com_reward + w_r * root_reward
    reward /= w_p + w_v + w_e + w_c + w_r
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_reward, vel_reward, ee_reward, com_reward, root_reward])


def fine_tune_kin_action_reward(env, state, action, old_action, info):
    #root reward
    cfg = env.cfg
    ws = cfg.reward_weights
    w_rp, w_rq, w_a, w_p, w_v, w_end = ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_a', 0.05), ws.get('w_p', 1.0), ws.get('w_v', 1.0), ws.get('w_end', 0.0)
    k_rp, k_rq, k_a, k_p, k_v = ws.get('k_rp', 1.0), ws.get('k_rq', 1.0), ws.get('k_a', 1.0 ), ws.get('k_p', 1.0), ws.get('k_v', 0.1)

    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    cur_hpos = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hpvel = (cur_hpos[:3] - prev_hpos[:3]) / env.dt
    hqvel = get_angvel_fd(prev_hpos[3:], cur_hpos[3:], env.dt)

    cur_bquat = env.get_body_quat()[4:]
    if env.fix_start_ind is not None:
        assert ind + env.fix_start_ind < env.expert['len'], "the index is negative value!"
        e_hpos = env.get_expert_attr('head_info', ind + env.fix_start_ind)
        e_hvel =  env.get_expert_attr('hvel', ind + env.fix_start_ind)
    else:
        e_hpos = env.get_expert_attr('head_info', ind)
    e_bquat = env.convert_body_quat(env.get_kinematic_pose_ind(ind).copy())
    

    if env.cfg.adap_weight:
        e_hvel_local = env.get_expert_attr('hvel_local', ind + env.fix_start_ind)
        kin_lvel = env.get_kin_vel(ind).copy()        
        w_p = math.exp(-1.0 * np.linalg.norm(kin_lvel - e_hvel_local))
        w_a = 1.0 - w_p
        w_a *= 0.1
        #w_a = 0.0

    # head position reward
    hp_dist = np.linalg.norm(cur_hpos[:3] - e_hpos[:3])
    hp_reward = math.exp(-k_rp * (hp_dist ** 2)) 

    # head orientation reward
    hq_dist = np.linalg.norm(multi_quat_norm_v2(multi_quat_diff(cur_hpos[3:], e_hpos[3:])))
    hq_reward = math.exp(-k_rq * (hq_dist ** 2))


    # head velocity reward 
    hpvel_dist = np.linalg.norm(hpvel - e_hvel[:3])
    hqvel_dist = np.linalg.norm(hqvel - e_hvel[3:])
    hvel_reward = math.exp(-hpvel_dist - k_v * hqvel_dist)


    # Action reward
    action_dist = np.linalg.norm(action - old_action)
    action_reward = math.exp(-k_a * (action_dist ** 2))

    pose_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat, e_bquat))
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))


    reward = w_rp * hp_reward + w_rq * hq_reward + w_v * hvel_reward + w_p * pose_reward + w_a * action_reward
    reward /= (w_rp + w_rq + w_v + w_p + w_a)
    if info['end']:
        reward = reward + w_end * env.end_reward


    return reward, np.array([hp_reward, hq_reward, hvel_reward, pose_reward, action_reward])


def fine_tune_action_reward(env, state, action, old_action, info):
    #root reward
    cfg = env.cfg
    ws = cfg.reward_weights
    w_rp, w_rq, w_a, w_p, w_v, w_end = ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_a', 0.05), ws.get('w_p', 1.0), ws.get('w_v', 1.0), ws.get('w_end', 1.0)
    k_rp, k_rq, k_a, k_p, k_v = ws.get('k_rp', 1.0), ws.get('k_rq', 1.0), ws.get('k_a', 1.0 ), ws.get('k_p', 1.0), ws.get('k_v', 0.1)

    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    cur_hpos = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hpvel = (cur_hpos[:3] - prev_hpos[:3]) / env.dt
    hqvel = get_angvel_fd(prev_hpos[3:], cur_hpos[3:], env.dt)

    if env.fix_start_ind is not None:
        assert ind + env.fix_start_ind < env.expert['len'], "the index is negative value!"
        e_hpos = env.get_expert_attr('head_info', ind + env.fix_start_ind)
        e_hvel =  env.get_expert_attr('hvel', ind + env.fix_start_ind)
    else:
        e_hpos = env.get_expert_attr('head_info', ind)
    

    # head position reward
    hp_dist = np.linalg.norm(cur_hpos[:3] - e_hpos[:3])
    hp_reward = math.exp(-k_rp * (hp_dist ** 2)) 

    # head orientation reward
    hq_dist = np.linalg.norm(multi_quat_norm_v2(multi_quat_diff(cur_hpos[3:], e_hpos[3:])))
    hq_reward = math.exp(-k_rq * (hq_dist ** 2))


    # head velocity reward 
    hpvel_dist = np.linalg.norm(hpvel - e_hvel[:3])
    hqvel_dist = np.linalg.norm(hqvel - e_hvel[3:])
    hvel_reward = math.exp(-hpvel_dist - k_v * hqvel_dist)


    # Action reward
    action_dist = np.linalg.norm(action - old_action)
    #print(action_dist)
    action_reward = math.exp(-k_a * (action_dist ** 2))


    #reward = w_rp * hp_reward + w_rq * hq_reward + w_v * hvel_reward + w_a * action_reward
    #reward /= (w_rp + w_rq + w_v + w_a + w_p)

    reward = hp_reward * hq_reward * hvel_reward + w_a * action_reward
    #reward /= (w_rp + w_rq + w_v + w_a)
    if info['end']:
        reward = reward + w_end * env.end_reward


    return reward, np.array([hp_reward, hq_reward, hvel_reward, action_reward])

def fine_tune_reward(env, state, action, info):
    #root reward
    cfg = env.cfg
    ws = cfg.reward_weights
    w_rp, w_rq, w_a, w_p, w_v, w_end = ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_a', 0.05), ws.get('w_p', 1.0), ws.get('w_v', 1.0), ws.get('w_end', 1.0)
    k_rp, k_rq, k_a, k_p, k_v = ws.get('k_rp', 1.0), ws.get('k_rq', 1.0), ws.get('k_a', 1.0 ), ws.get('k_p', 1.0), ws.get('k_v', 0.1)

    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    cur_hpos = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hpvel = (cur_hpos[:3] - prev_hpos[:3]) / env.dt
    hqvel = get_angvel_fd(prev_hpos[3:], cur_hpos[3:], env.dt)

    cur_bquat = env.get_body_quat()[4:]
    if env.fix_start_ind is not None:
        assert ind + env.fix_start_ind < env.expert['len'], "the index is negative value!"
        e_hpos = env.get_expert_attr('head_info', ind + env.fix_start_ind)
        e_hvel =  env.get_expert_attr('hvel', ind + env.fix_start_ind)
    else:
        e_hpos = env.get_expert_attr('head_info', ind)
    e_bquat = env.convert_body_quat(env.get_kinematic_pose_ind(ind).copy())
    

    if env.cfg.adap_weight:
        e_hvel_local = env.get_expert_attr('hvel_local', ind + env.fix_start_ind)
        kin_lvel = env.get_kin_vel(ind).copy()        
        kin_weight = math.exp(-1.0 * np.linalg.norm(kin_lvel - e_hvel_local))
        w_p = kin_weight

    # head position reward
    hp_dist = np.linalg.norm(cur_hpos[:3] - e_hpos[:3])
    hp_reward = math.exp(-k_rp * (hp_dist ** 2)) 
    

    # head orientation reward
    hq_dist = np.linalg.norm(multi_quat_norm_v2(multi_quat_diff(cur_hpos[3:], e_hpos[3:])))
    hq_reward = math.exp(-k_rq * (hq_dist ** 2))


    # head velocity reward 
    hpvel_dist = np.linalg.norm(hpvel - e_hvel[:3])
    hqvel_dist = np.linalg.norm(hqvel - e_hvel[3:])
    hvel_reward = math.exp(-hpvel_dist - k_v * hqvel_dist)


    # Action reward
    # action_dist = np.linalg.norm(action - old_action)
    # action_reward = math.exp(-k_a * (action_dist ** 2))

    pose_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat, e_bquat))
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))


    # reward = w_rp * hp_reward + w_rq * hq_reward + w_v * hvel_reward + w_p * pose_reward
    # reward /= (w_rp + w_rq + w_v + w_p)
    # if info['end']:
    #     reward = reward + w_end * env.end_reward
    reward =  hp_reward *  hq_reward  * hvel_reward * pose_reward
    if info['end']:
        reward = reward * env.end_reward

    return reward, np.array([hp_reward, hq_reward, hvel_reward, pose_reward])
 

def dynamic_supervision_v1(env, state, action, info):
    # V1 uses GT 
    # V1 now does not regulate the action using GT, and only has act_v 
    cfg = env.cfg
    ws = cfg.policy_specs['reward_weights']
    w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_p, w_act_v = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
         ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0), ws.get('w_act_v', 1.0)
    k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_p, k_act_v = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1), ws.get('k_act_v', 0.1)
    v_ord = ws.get('v_ord', 2)

    ind = env.cur_t
    # Head losses
    tgt_hpose = env.ar_context['head_pose'][ind]

    cur_hpose = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hpvel = (cur_hpose[:3] - prev_hpos[:3]) / env.dt

    hp_dist = np.linalg.norm(cur_hpose[:3] - tgt_hpose[:3])
    hp_reward = math.exp(-k_hp * (hp_dist ** 2)) 
    
    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpose[3:], tgt_hpose[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist ** 2))

    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))


    # Comparing with GT
    gt_bquat = env.ar_context['bquat'][ind].flatten()
    gt_prev_bquat = env.ar_context['bquat'][ind - 1].flatten()
    prev_bquat = env.prev_bquat

    pose_gt_diff = multi_quat_norm_v2(multi_quat_diff(gt_bquat, cur_bquat)).mean()
    
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    tgt_bangvel = get_angvel_fd(gt_prev_bquat, gt_bquat, env.dt)
    vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    act_v_reward = math.exp(-k_act_v * (vel_dist ** 2))

    # rp_dist = np.linalg.norm(tgt_qpos[:3] - act_qpos[:3])
    # rq_dist = multi_quat_norm_v2(multi_quat_diff(tgt_qpos[3:7], act_qpos[3:7])).mean()
    # rq_reward = math.exp(-k_rq * (rq_dist ** 2))
    # rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    gt_p_reward = math.exp(-k_act_p * pose_gt_diff)

    reward = w_hp * hp_reward + w_hq * hq_reward + w_p * p_reward + \
        w_jp * jp_reward + w_act_p * gt_p_reward + w_act_v * act_v_reward

    if flags.debug:
        import pdb; pdb.set_trace()
        np.set_printoptions(precision=4, suppress=1)
        print(reward, np.array([hp_reward, hq_reward, p_reward, jp_reward, gt_p_reward, act_v_reward]))
    
    return reward, np.array([hp_reward, hq_reward, p_reward, jp_reward, gt_p_reward, act_v_reward])



def dynamic_supervision_v2(env, state, action, info):
    # V2 uses no GT
    # velocity loss is from AR-Net , reguralize the actions by running the model kinematically
    # This thing makes 0 sense rn 
    # cfg = env.cfg
    # ws = cfg.policy_specs['reward_weights']
    # w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_v, w_act_p = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
    #      ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_v', 1.0),  ws.get('w_act_p', 1.0)
    # k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_v, k_act_p = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
    #     ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_v', 0.1), ws.get('k_act_p', 0.1)
    # v_ord = ws.get('v_ord', 2)
    
    # ind = env.cur_t
    # # Head losses
    # tgt_hpos = env.ar_context['head_pose'][ind]
    # tgt_hvel = env.ar_context['head_vels'][ind]

    # cur_hpos = env.get_head().copy()
    # prev_hpos = env.prev_hpos.copy()

    # hp_dist = np.linalg.norm(cur_hpos[:3] - tgt_hpos[:3])
    # hp_reward = math.exp(-k_hp * (hp_dist ** 2)) 
    
    # # head orientation reward
    # hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpos[3:], tgt_hpos[3:])).mean()
    # hq_reward = math.exp(-k_hq * (hq_dist ** 2))

    # # head velocity reward 
    # # hpvel = (cur_hpos[:3] - prev_hpos[:3]) / env.dt
    # # hqvel = get_angvel_fd(prev_hpos[3:], cur_hpos[3:], env.dt)
    # # hpvel_dist = np.linalg.norm(hpvel - tgt_hvel[:3])
    # # hqvel_dist = np.linalg.norm(hqvel - tgt_hvel[3:])
    # # hv_reward = math.exp(-hpvel_dist - k_hv * hqvel_dist)
    # hv_reward = 0
    
    # cur_bquat = env.get_body_quat()
    # cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    # tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    # pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    # pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    # p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    # jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))

    # ## ARNet Action supervision
    # act_qpos = env.target['qpos']
    # tgt_qpos = env.ar_context['ar_qpos'][ind]

    # act_bquat = env.target['bquat'].flatten()
    # tgt_bquat = env.ar_context['ar_bquat'][ind].flatten()
    # tgt_prev_bquat = env.ar_context['ar_bquat'][ind - 1].flatten()
    # prev_bquat = env.prev_bquat
    

    # rp_dist = np.linalg.norm(tgt_qpos[:3] - act_qpos[:3])
    # rq_dist = multi_quat_norm_v2(multi_quat_diff(tgt_qpos[3:7], act_qpos[3:7])).mean()
    # pose_action_diff = multi_quat_norm_v2(multi_quat_diff(tgt_bquat, act_bquat)).mean()

    # cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # tgt_bangvel = get_angvel_fd(tgt_prev_bquat, tgt_bquat, env.dt)
    # vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    # act_v_reward = math.exp(-k_act_v * (vel_dist ** 2))

    # rq_reward = math.exp(-k_rq * (rq_dist ** 2))
    # rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    # act_p_reward = math.exp(-k_act_p * (pose_action_diff))
    # # rq_reward = 0
    # # rp_reward = 0
    # # act_p_reward = 0

    
    # reward = w_hp * hp_reward + w_hq * hq_reward + w_hv * hv_reward + w_p * p_reward + \
    #     w_jp * jp_reward + w_rp * rp_reward + w_rq * rq_reward  + w_act_v * act_v_reward + w_act_p * act_p_reward
    # print(reward)
    # if flags.debug:
    #     import pdb; pdb.set_trace()
    #     np.set_printoptions(precision=4, suppress=1)
    #     print(np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward, rp_reward, rq_reward, act_v_reward, act_p_reward]))
    
    return reward, np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward, rp_reward, rq_reward, act_v_reward, act_p_reward])

def dynamic_supervision_v3(env, state, action, info):
    # V3 is V2 mutiplicative
    # This is wrong, very wrong. This does not work since you should compare the simulated with the estimated!!!!!!
    cfg = env.cfg
    ws = cfg.policy_specs['reward_weights']
    # w_hp, w_hq, w_p, w_jp, w_rp, w_rq, w_act_p, w_act_v = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
    #     ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0), ws.get('w_act_v', 1.0)
    k_hp, k_hq,  k_p, k_jp, k_rp, k_rq, k_act_p, k_act_v = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0),   \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1), ws.get('k_act_v', 0.1)
    v_ord = ws.get('v_ord', 2)
    
    ind = env.cur_t
    # Head losses
    tgt_hpos = env.ar_context['head_pose'][ind]
    tgt_hvel = env.ar_context['head_vels'][ind]

    cur_hpos = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hp_dist = np.linalg.norm(cur_hpos[:3] - tgt_hpos[:3])
    hp_reward = math.exp(-k_hp * (hp_dist ** 2)) 
    
    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpos[3:], tgt_hpos[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist ** 2))

    cur_bquat = env.get_body_quat()

    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat, tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))

    ## ARNet Action supervision
    act_qpos = env.target['qpos']
    tgt_qpos = env.ar_context['ar_qpos'][ind]

    act_bquat = env.target['bquat'].flatten()
    tgt_bquat = env.ar_context['ar_bquat'][ind].flatten()
    tgt_prev_bquat = env.ar_context['ar_bquat'][ind - 1].flatten()
    prev_bquat = env.prev_bquat

    rp_dist = np.linalg.norm(tgt_qpos[:3] - act_qpos[:3])
    rq_dist = multi_quat_norm_v2(multi_quat_diff(tgt_qpos[3:7], act_qpos[3:7])).mean()
    pose_action_diff = multi_quat_norm_v2(multi_quat_diff(tgt_bquat, act_bquat)).mean()

    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    tgt_bangvel = get_angvel_fd(tgt_prev_bquat, tgt_bquat, env.dt)
    vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    act_v_reward = math.exp(-k_act_v * (vel_dist ** 2))
    # act_v_reward = 0

    rq_reward = math.exp(-k_rq * (rq_dist ** 2))
    rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    act_p_reward = math.exp(-k_act_p * (pose_action_diff))

    # import pdb; pdb.set_trace()

    # reward = hp_reward * hq_reward  *  p_reward * jp_reward * rp_reward  * rq_reward  * act_p_reward * act_v_reward
    reward = hp_reward * hq_reward  *  p_reward * jp_reward * rp_reward  * rq_reward  * act_p_reward 
    # if flags.debug:
        # np.set_printoptions(precision=4, suppress=1)
        # print(reward, np.array([hp_reward, hq_reward, p_reward, jp_reward, rp_reward, rq_reward, act_p_reward, act_v_reward]))
    
    return reward, np.array([hp_reward, hq_reward, p_reward, jp_reward, rp_reward, rq_reward, act_p_reward, act_v_reward])


def dynamic_supervision_v4(env, state, action, info):
    # V4 does not have the action terms (does not regularize the action)
    cfg = env.cfg
    ws = cfg.policy_specs['reward_weights']
    w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_p = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
         ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0)
    k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_p = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1)
    
    ind = env.cur_t
    # Head losses
    tgt_hpose = env.ar_context['head_pose'][ind]
    # tgt_hvel = env.ar_context['head_vels'][ind]

    cur_hpose = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hpvel = (cur_hpose[:3] - prev_hpos[:3]) / env.dt
    # hqvel = get_angvel_fd(prev_hpos[3:], cur_hpose[3:], env.dt)


    hp_dist = np.linalg.norm(cur_hpose[:3] - tgt_hpose[:3])
    hp_reward = math.exp(-k_hp * (hp_dist ** 2)) 
    
    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpose[3:], tgt_hpose[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist ** 2))

    # head velocity reward 
    # hpvel_dist = np.linalg.norm(hpvel - tgt_hvel[:3])
    # hqvel_dist = np.linalg.norm(hqvel - tgt_hvel[3:])
    # hv_reward = math.exp(-hpvel_dist - k_hv * hqvel_dist)
    hv_reward = 0
    
    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))
    
    
    reward = w_hp * hp_reward + w_hq * hq_reward + w_hv * hv_reward + w_p * p_reward + w_jp * jp_reward 

    # if flags.debug:
        # np.set_printoptions(precision=4, suppress=1)
        # print(np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward, rp_reward, rq_reward, act_p_reward]))
    
    return reward, np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward])

def dynamic_supervision_v5(env, state, action, info):
    # V5 is V4 with multiplicative reward
    cfg = env.cfg
    ws = cfg.policy_specs['reward_weights']
    w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_p = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
         ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0)
    k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_p = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1)
    
    ind = env.cur_t
    # Head losses
    tgt_hpose = env.ar_context['head_pose'][ind]
    # tgt_hvel = env.ar_context['head_vels'][ind]

    cur_hpose = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hpvel = (cur_hpose[:3] - prev_hpos[:3]) / env.dt
    # hqvel = get_angvel_fd(prev_hpos[3:], cur_hpose[3:], env.dt)


    hp_dist = np.linalg.norm(cur_hpose[:3] - tgt_hpose[:3])
    hp_reward = math.exp(-k_hp * (hp_dist ** 2)) 
    
    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpose[3:], tgt_hpose[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist ** 2))

    # head velocity reward 
    # hpvel_dist = np.linalg.norm(hpvel - tgt_hvel[:3])
    # hqvel_dist = np.linalg.norm(hqvel - tgt_hvel[3:])
    # hv_reward = math.exp(-hpvel_dist - k_hv * hqvel_dist)
    hv_reward = 0
    
    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))
    
    
    reward =  hp_reward  * hq_reward    * p_reward  * jp_reward 

    # if flags.debug:
        # np.set_printoptions(precision=4, suppress=1)
        # print(np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward, rp_reward, rq_reward, act_p_reward]))
    
    return reward, np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward])


def dynamic_supervision_v6(env, state, action, info):
    # no head reward anymore 
    cfg = env.cfg
    ws = cfg.policy_specs['reward_weights']
    w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_p, w_act_v = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
         ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0), ws.get('w_act_v', 1.0)
    k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_p, k_act_v = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1), ws.get('k_act_v', 0.1)
    v_ord = ws.get('v_ord', 2)

    ind = env.cur_t

    # Head losses
    tgt_hpose = env.ar_context['head_pose'][ind]

    cur_hpose = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hp_dist = np.linalg.norm(cur_hpose[:3] - tgt_hpose[:3])
    hp_reward = math.exp(-k_hp * (hp_dist ** 2)) 
    
    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpose[3:], tgt_hpose[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist ** 2))
    
    
    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))
    

    tgt_bquat = env.ar_context['ar_bquat'][ind].flatten()
    tgt_prev_bquat = env.ar_context['ar_bquat'][ind - 1].flatten()
    prev_bquat = env.prev_bquat

    
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    tgt_bangvel = get_angvel_fd(tgt_prev_bquat, tgt_bquat, env.dt)
    vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    act_v_reward = math.exp(-k_act_v * (vel_dist ** 2))

    reward =   w_hp * hp_reward + w_hq * hq_reward + w_p * p_reward + w_jp * jp_reward + w_act_v * act_v_reward

    if flags.debug:
        import pdb; pdb.set_trace()
        np.set_printoptions(precision=4, suppress=1)
        print(reward, np.array([p_reward, jp_reward, act_v_reward]))
    
    return reward, np.array([hp_reward, hq_reward, p_reward, jp_reward, act_v_reward])


def constant_reward(env, state, action, info):
    reward = 1.0
    if info['end']:
        reward += env.end_reward
    return 1.0, np.zeros(1)

reward_func = {'quat_v2': quat_space_reward_v2,
               'quat_v3': quat_space_reward_v3,
               'deep_mimic': deep_mimic_reward,
               'deep_mimic_v2': deep_mimic_reward_v2,
               'multiplicable_reward': multiplicable_reward,
               'local_world_v1': local_world_reward_v1,
               'local_world_v2': local_world_reward_v2,
               'local_world_v3': local_world_reward_v3,
               'world_quat': world_quat_space_reward,
               'world_quat_v2': world_quat_space_reward_v2,
               'constant': constant_reward,
               'fine_tune_action_reward': fine_tune_action_reward,
               'fine_tune_reward': fine_tune_reward,
               'fine_tune_kin_action_reward':fine_tune_kin_action_reward,
               "deep_mimic_reward_v2_vf": deep_mimic_reward_v2_vf, 
               "deep_mimic_reward_v2_vf_vq": deep_mimic_reward_v2_vf_vq, 
               "dynamic_supervision_v1": dynamic_supervision_v1, 
               "dynamic_supervision_v2": dynamic_supervision_v2, 
               "dynamic_supervision_v3": dynamic_supervision_v3,
               "dynamic_supervision_v4": dynamic_supervision_v4,
               "dynamic_supervision_v5": dynamic_supervision_v5,
               "dynamic_supervision_v6": dynamic_supervision_v6,
               }