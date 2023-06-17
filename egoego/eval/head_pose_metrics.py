import os 
import numpy as np

def get_frobenious_norm(x, y):
    # x/y: T X 4 X 4 
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i]
        y_mat_inv = np.linalg.inv(y[i])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(4)
        error += np.linalg.norm(ident_mat - error_mat, 'fro')
    return error / len(x)

def get_frobenious_norm_rot_only(x, y):
    # x/y: T X 3 X 3 
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i]
        y_mat_inv = np.linalg.inv(y[i])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(3)
        error += np.linalg.norm(ident_mat - error_mat, 'fro')
    return error / len(x)

def compute_head_pose_metrics(head_trans, head_rot, gt_head_trans, gt_head_rot):
    # head_trans: T X 3
    # head_rot: T X 3 X 3  
    T = head_trans.shape[0]
    pred_head_mat = np.zeros((T, 4, 4)) 
    gt_head_mat = np.zeros((T, 4, 4))
    pred_head_mat[:, :3, :3] = head_rot 
    pred_head_mat[:, 3, 3] = 1.0
    gt_head_mat[:, :3, :3] = gt_head_rot 
    gt_head_mat[:, 3, 3] = 1.0 
    pred_head_mat[:, :3, 3] = head_trans 
    gt_head_mat[:, :3, 3] = gt_head_trans 
  
    head_dist = get_frobenious_norm(pred_head_mat, gt_head_mat)

    head_dist_rot_only = get_frobenious_norm_rot_only(head_rot, gt_head_rot)
    head_trans_err = np.linalg.norm(head_trans - gt_head_trans, axis = 1).mean() * 1000
    
    return head_dist, head_dist_rot_only, head_trans_err   