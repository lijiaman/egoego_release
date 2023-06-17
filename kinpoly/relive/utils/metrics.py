from relive.utils import *
from relive.utils.transformation import euler_from_quaternion, quaternion_matrix

def get_joint_angles(poses):
    return poses[:, 7:]

def get_root_angles(poses):
    root_angs = []
    for pose in poses:
        root_euler = np.array(euler_from_quaternion(pose[3:7]))
        root_angs.append(root_euler)

    return np.array(root_angs)

def get_root_matrix(poses):
    matrices = []
    for pose in poses:
        mat = np.identity(4)
        root_pos = pose[:3]
        root_quat = pose[3:7]
        mat = quaternion_matrix(root_quat)
        mat[:3, 3] = root_pos
        matrices.append(mat)
    return matrices
    

'''
def get_joint_angles(poses):
    root_angs = []
    for pose in poses:
        root_euler = np.array(euler_from_quaternion(pose[3:7]))
        root_angs.append(root_euler)
    root_angs = np.vstack(root_angs)
    angles = np.hstack((root_angs, poses[:, 7:]))
    return angles
'''

def get_joint_vels(poses, dt):
    vels = []
    for i in range(poses.shape[0] - 1):
        v = get_qvel_fd(poses[i], poses[i+1], dt, 'heading')
        vels.append(v)
    vels = np.vstack(vels)
    return vels


def get_joint_accels(vels, dt):
    accels = np.diff(vels, axis=0) / dt
    accels = np.vstack(accels)
    return accels

def get_root_pos(poses):
    return poses[:, :3]


def get_mean_dist(x, y):
    return np.linalg.norm(x - y, axis=1).mean()


def get_mean_abs(x):
    return np.abs(x).mean()


def get_frobenious_norm(x, y):
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i]
        y_mat_inv = np.linalg.inv(y[i])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(4)
        error += np.linalg.norm(ident_mat - error_mat, 'fro')
    return error / len(x)

def get_frobenious_norm_rot_only(x, y):
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i][:3, :3]
        y_mat_inv = np.linalg.inv(y[i][:3, :3])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(3)
        error += np.linalg.norm(ident_mat - error_mat, 'fro')
    return error / len(x)