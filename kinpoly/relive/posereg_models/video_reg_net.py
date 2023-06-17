from relive.utils.torch_posereg import *
from torch import nn
from relive.posereg_models.resnet import ResNet
from relive.posereg_models.tcn import TemporalConvNet
from relive.posereg_models.rnn import RNN
from relive.posereg_models.mlp import MLP
from relive.posereg_models.mobile_net import MobileNet

from collections import defaultdict

class VideoRegNet(nn.Module):

    def __init__(self, out_dim, v_hdim, cnn_fdim, no_cnn=False, frame_shape=(3, 224, 224), mlp_dim=(300, 200),
                 cnn_type='resnet', v_net_type='lstm', v_net_param=None, causal=False):
        super().__init__()
        self.out_dim = out_dim
        self.cnn_fdim = cnn_fdim
        self.v_hdim = v_hdim
        self.no_cnn = no_cnn
        self.frame_shape = frame_shape
        if no_cnn:
            self.cnn = None
        elif cnn_type == 'resnet':
            self.cnn = ResNet(cnn_fdim)
        elif cnn_type == 'mobile':
            self.cnn = MobileNet(cnn_fdim)

        self.v_net_type = v_net_type
        if v_net_type == 'lstm':
            self.v_net = RNN(cnn_fdim, v_hdim, v_net_type, bi_dir=not causal)
        elif v_net_type == 'tcn':
            if v_net_param is None:
                v_net_param = {}
            tcn_size = v_net_param.get('size', [64, 128])
            dropout = v_net_param.get('dropout', 0.2)
            kernel_size = v_net_param.get('kernel_size', 3)
            assert tcn_size[-1] == v_hdim
            self.v_net = TemporalConvNet(cnn_fdim, tcn_size, kernel_size=kernel_size, dropout=dropout, causal=causal)
        self.mlp = MLP(v_hdim, mlp_dim, 'relu')
        self.linear = nn.Linear(self.mlp.out_dim, out_dim)

    def forward_v_net(self, x):
        if self.v_net_type == 'tcn':
            x = x.permute(1, 2, 0).contiguous()
        x = self.v_net(x)
        if self.v_net_type == 'tcn':
            x = x.permute(2, 0, 1).contiguous()
        return x

    def forward(self, data_dict):
        feature_pred = defaultdict(list)

        x = data_dict['of']
        bs, timesteps, _ = x.shape
        # CNN
        if self.cnn is not None:
            x = self.cnn(x.view((-1,) + self.frame_shape)).view(-1, x.size(1), self.cnn_fdim)
        x = self.forward_v_net(x).view(-1, self.v_hdim)
        x = self.mlp(x)
        x = self.linear(x)

        feature_pred['pred_traj'] = x.reshape(bs, timesteps, -1)  

        return feature_pred 

    def compute_loss(self, feature_pred, data):
        w_traj = 1.

        state_pred = feature_pred['pred_traj']
        state_gt = data['posereg_traj_norm']
      
        traj_loss = (state_gt - state_pred).pow(2).sum(dim=1).mean()
     
        loss = w_traj * traj_loss
 
        return loss, [i.item() for i in [traj_loss]]

    def get_cnn_feature(self, x):
        return self.cnn(x.view((-1,) + self.frame_shape))


if __name__ == '__main__':
    net = VideoRegNet(64, 128, 128, v_net_type='tcn', cnn_type='mobile')
    input = ones(32, 1, 3, 224, 224)
    out = net(input)
    print(out.shape)
