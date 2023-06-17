import glob
import os
import sys
import time 
import pdb
import os.path as osp
from tracemalloc import start
sys.path.append(os.getcwd())

from torch import nn
from collections import defaultdict
import joblib

from relive.utils.torch_ext import *
from relive.models.rnn import RNN
from relive.models.resnet import ResNet
from relive.models.mobile_net import MobileNet
from relive.models.mlp import MLP
from relive.utils.torch_utils import (get_heading_q, quaternion_multiply, quaternion_inverse, get_heading_q_batch,transform_vec_batch, quat_from_expmap_batch,
                quat_mul_vec_batch, get_qvel_fd_batch, transform_vec, rotation_from_quaternion, de_heading_batch, quat_mul_vec, quat_from_expmap, quaternion_multiply_batch, quaternion_inverse_batch)
    
from relive.utils.torch_smpl_humanoid import Humanoid
from relive.utils.compute_loss import pose_rot_loss, root_pos_loss, root_orientation_loss, end_effector_pos_loss, linear_velocity_loss, angular_velocity_loss, action_loss, position_loss, orientation_loss

from relive.models.transformer_module import Decoder 
from relive.models.transformer_module import PerStepDecoder 

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.cnn_fdim = 512 
        self.cnn = ResNet(self.cnn_fdim, running_stats=False, pretrained=True)
        # If freeze the CNN params 
        for param in self.cnn.parameters():
            param.requires_grad = False
       
    def to(self, device):
        self.device = device
        super().to(device)
        return self
    
    def forward(self, data):
        # pose: 69 dim body pose
        batch_size, seq_len, _, _, _ = data['of'].shape # 

        of_data = data['of'] # B X T X 224 X 224 X 2 
        of_data = torch.cat((of_data, torch.zeros(of_data.shape[:-1] + (1,), device=of_data.device)), dim=-1)
        h, w = 224, 224 
        c = 3
        of_data = of_data.reshape(-1, h, w, c).permute(0, 3, 1, 2) # B X T X 3 X 224 X 224 
        input_features = self.cnn(of_data).reshape(batch_size, seq_len, self.cnn_fdim) # B X T X D 

        return input_features 
