import numpy as np
import torch 
import pytorch3d.transforms as transforms

def length(x, axis=-1, keepdims=True):
    """
    Computes vector norm along a tensor axis(axes)

    :param x: tensor
    :param axis: axis(axes) along which to compute the norm
    :param keepdims: indicates if the dimension(s) on axis should be kept
    :return: The length or vector of lengths.
    """
    lgth = np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims))
    return lgth


def normalize(x, axis=-1, eps=1e-8):
    """
    Normalizes a tensor over some axis (axes)

    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    res = x / (length(x, axis=axis) + eps)
    return res


def quat_normalize(x, eps=1e-8):
    """
    Normalizes a quaternion tensor

    :param x: data tensor
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized quaternions tensor
    """
    res = normalize(x, eps=eps)
    return res


def quat_inv(q):
    """
    Inverts a tensor of quaternions

    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    res = np.asarray([1, -1, -1, -1], dtype=np.float32) * q
    return res


def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate(
        [
            y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
            y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
            y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
            y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
        ],
        axis=-1,
    )

    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * np.cross(q[..., 1:], x)
    res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)

    return res


def quat_between(x, y):
    """
    Quaternion rotations between two 3D-vector arrays

    :param x: tensor of 3D vectors
    :param y: tensor of 3D vetcors
    :return: tensor of quaternions
    """
    res = np.concatenate(
        [
            np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis]
            + np.sum(x * y, axis=-1)[..., np.newaxis],
            np.cross(x, y),
        ],
        axis=-1,
    )
    return res


def rotate_at_frame_smplh(root_trans, root_quat, cano_t_idx=0):
    """
    numpy array 
    root trans: BS X T X 3
    root_quat: BS X T X 4 
    trans2joint: BS X 3 
    cano_t_idx: use which frame for forward direction canonicalization 
    """
    # rest_body_root_joints[0,0]: tensor([ 0.0011, -0.3982,  0.0114])

    global_q = root_quat[:, np.newaxis, :, :] # BS X 1 X T X 4
    global_x = root_trans[:, np.newaxis, :, :] # BS X 1 X T X 3 

    key_glob_Q = global_q[:, :, cano_t_idx:cano_t_idx+1, :]  # (B, 1, 1, 4)
  
    # The floor is on z = xxx. Project the forward direction to xy plane. 
    forward = np.array([1, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
        key_glob_Q, np.array([1, 0, 0])[np.newaxis, np.newaxis, np.newaxis, :]
    ) # In rest pose, x direction is the body left direction, root joint point to left hip joint.  

    forward = normalize(forward)
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
    new_glob_Q = quat_mul(quat_inv(yrot), global_q)
    new_glob_X = quat_mul_vec(quat_inv(yrot), global_x)

    # BS X T X 3, BS X T X 4, BS(1) X 1 X 1 X 4  
    return new_glob_X[:, 0, :, :], new_glob_Q[:, 0, :, :], yrot
    # Need yrot for visualization. yrot deirecly applied to human mesh vertices will recover it to original scene.
