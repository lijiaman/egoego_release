# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import torch
import numpy as np
import os.path as osp
from smplx import SMPL as _SMPL
# from smplx.body_models import ModelOutput
# from smplx.body_models import SMPLOutput 

from smplx.lbs import vertices2joints

SMPL_BONE_NAMES = ['Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 
                    'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']

# SMPL_BONE_NAMES = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', "LeftLeg", "RightLeg", "Spine1", "LeftFoot", \
#         "RightFoot", "Spine2", "LeftToe", "RightToe", "Neck", "LeftChest", "RightChest", "Mouth", "LeftShoulder", \
#          "RightShoulder", "LeftArm", "RightArm", "LeftWrist", "RightWrist", "LeftHand", "RightHand"
#         ]

JOINST_TO_USE = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 37
])

class SMPL_Parser(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        ''' SMPL model constructor
            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_global_orient: bool, optional
                Flag for creating a member variable for the global orientation
                of the body. (default = True)
            global_orient: torch.tensor, optional, Bx3
                The default value for the global orientation variable.
                (default = None)
            create_body_pose: bool, optional
                Flag for creating a member variable for the pose of the body.
                (default = True)
            body_pose: torch.tensor, optional, Bx(Body Joints * 3)
                The default value for the body pose variable.
                (default = None)
            create_betas: bool, optional
                Flag for creating a member variable for the shape space
                (default = True).
            betas: torch.tensor, optional, Bx10
                The default value for the shape member variable.
                (default = None)
            create_transl: bool, optional
                Flag for creating a member variable for the translation
                of the body. (default = True)
            transl: torch.tensor, optional, Bx3
                The default value for the transl variable.
                (default = None)
            dtype: torch.dtype, optional
                The data type for the created variables
            batch_size: int, optional
                The batch size used for creating the member variables
            joint_mapper: object, optional
                An object that re-maps the joints. Useful if one wants to
                re-order the SMPL joints to some other convention (e.g. MSCOCO)
                (default = None)
            gender: str, optional
                Which gender to load
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''
        super(SMPL_Parser, self).__init__(*args, **kwargs)
        self.device = next(self.parameters()).device
        
    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL_Parser, self).forward(*args, **kwargs)
        return smpl_output
    
    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        '''
            Pose should be batch_size x 72
        '''
        if pose.shape[1] == 72:
            pass
        batch_size= pose.shape[0]
        smpl_output = self.forward(betas = th_betas,  transl = None, \
                                   body_pose = pose[:,3:], global_orient = pose[:,:3])
        vertices = smpl_output.vertices
        joints = smpl_output.joints[:,:24]
        # joints = smpl_output.joints[:,JOINST_TO_USE]
        return vertices, joints
    

