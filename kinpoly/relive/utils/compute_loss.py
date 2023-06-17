import torch
import relive.utils.torch_utils as tu
import torch.nn.functional as F

def get_body_rot(pose):
     # exclude root position, quaternion
    quat = pose[:, 7:]
    return quat

def get_root_quat(pose):
    quat = pose[:, 3:7]
    return quat

def get_root_pos(pose):
    pos = pose[:, :3]
    return pos


def quat_diff_batch(q1, q0):
    diff = tu.quaternion_multiply_batch(q1, tu.quaternion_inverse_batch(q0))
    return diff

def get_angvel_fd(prev_bquat, cur_bquat):
    q_diff = quat_diff_batch(cur_bquat, prev_bquat)
    angvel = tu.rotation_from_quaternion_batch(q_diff)
    return angvel

def get_ee(pose, skeleton):
    ee_name = ['LeftFoot', 'RightFoot', 'LeftHand', 'RightHand', 'Head']
    joint_pos, _ = skeleton.get_body_pos_quat(pose, select_joints=ee_name)
    
    return joint_pos


def position_loss(gt_pos, pred_pos):
    return (gt_pos - pred_pos).pow(2).sum(dim=1)

def orientation_loss(gt_quat, pred_quat):
    dist = quat_diff_batch(gt_quat, pred_quat)
    """make the diff quat to be identity"""
    quat_iden = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gt_quat.dtype, device=gt_quat.device).repeat(gt_quat.size()[0], 1)
    loss = torch.abs(dist) - quat_iden
    return loss.pow(2).sum(dim=1)

def pose_rot_loss(gt_pose, pred_pose):
    ### Joint pose is expressed via euler angles
    loss = get_body_rot(gt_pose) - get_body_rot(pred_pose)
    return loss.pow(2).sum(dim=1)

def root_pos_loss(gt_pose, pred_pose):
    loss = get_root_pos(gt_pose) - get_root_pos(pred_pose)
    return loss.pow(2).sum(dim=1)

def root_orientation_loss(gt_pose, pred_pose):
    gt_quat = get_root_quat(gt_pose)
    pred_quat = get_root_quat(pred_pose)
    dist = quat_diff_batch(gt_quat, pred_quat)
    """make the diff quat to be identity"""
    quat_iden = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gt_pose.dtype, device=gt_pose.device).repeat(gt_pose.size()[0], 1)
    loss = torch.abs(dist) - quat_iden
    return loss.pow(2).sum(dim=1)

def end_effector_loss(gt_pose, pred_pose, humanoid_env):
    gt_ee = get_ee(gt_pose, humanoid_env)
    pred_ee = get_ee(pred_pose, humanoid_env)
    
    loss = (gt_ee - pred_ee).view(gt_ee.size()[0], -1)
    return loss.pow(2).sum(dim=1)

def end_effector_pos_loss(gt_pos, pred_pos):
    
    loss = (gt_pos - pred_pos).reshape(gt_pos.shape[0], -1)
    return loss.pow(2).sum(dim=1)

def action_loss(gt_action, pred_action):
    loss = (gt_action - pred_action).view(gt_action.size()[0], -1)
    return loss.pow(2).sum(dim=1)
    
    
def velocity_distance_loss(gt_vel, pred_vel):
    loss = gt_vel - pred_vel
    return loss.pow(2).sum(dim=1)

def linear_velocity_loss(gt_vel, pred_vel):
    loss = gt_vel[:, :3] - pred_vel[:, :3]
    return loss.pow(2).sum(dim=1)

def angular_velocity_loss(gt_vel, pred_vel):
    loss = gt_vel[:, 3:6] - pred_vel[:, 3:6]
    return loss.pow(2).sum(dim=1)

def DeepMimicLoss(gt_pose, pred_pose, gt_vel, pred_vel, cfg, humanoid_env=None):
    assert gt_pose.size() == pred_pose.size()
    w_p = cfg.w_p
    w_vl = cfg.w_vl
    w_va = cfg.w_va
    w_ee = cfg.w_ee
    w_rp = cfg.w_rp
    w_rr = cfg.w_rr
    
    rp_loss = root_pos_loss(gt_pose, pred_pose)
    rr_loss = root_orientation_loss(gt_pose, pred_pose)
    p_loss = pose_rot_loss(gt_pose, pred_pose)
    vl_loss = linear_velocity_loss(gt_vel, pred_vel)
    va_loss = angular_velocity_loss(gt_vel, pred_vel)


    loss = w_rp * rp_loss + w_rr * rr_loss + w_p * p_loss + w_vl * vl_loss + w_va * va_loss
    return loss.mean(), \
        [rp_loss.mean().cpu().item(), rr_loss.mean().cpu().item(), p_loss.mean().cpu().item(), vl_loss.mean().cpu().item(), va_loss.mean().cpu().item(), 0.0]

def TrajLoss(gt_pose, pred_pose, gt_vel, pred_vel, cfg):
    assert gt_pose.size() == pred_pose.size()
    w_vl = cfg.w_vl
    w_va = cfg.w_va
    w_rp = cfg.w_rp
    w_rr = cfg.w_rr
    
    rp_loss = root_pos_loss(gt_pose, pred_pose)
    rr_loss = root_orientation_loss(gt_pose, pred_pose)
    vl_loss = linear_velocity_loss(gt_vel, pred_vel)
    va_loss = angular_velocity_loss(gt_vel, pred_vel)

    loss = w_rp * rp_loss + w_rr * rr_loss + w_vl * vl_loss + w_va * va_loss
    return loss.mean(), \
        [rp_loss.mean().cpu().item(), rr_loss.mean().cpu().item(), 0.0, vl_loss.mean().cpu().item(), va_loss.mean().cpu().item(), 0.0]


def PoseLoss(gt_pose, pred_pose, ofpos=7):
    diff = gt_pose - pred_pose
    mask = torch.zeros(pred_pose.size(), dtype=pred_pose.dtype, device=pred_pose.device)
    mask[:, 0:ofpos] = 0.0 #do not include chair pose to the loss
    loss = (diff * mask).pow(2).sum(dim=1).mean()
    return loss


