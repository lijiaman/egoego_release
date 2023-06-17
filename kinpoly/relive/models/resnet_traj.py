from torch import nn
from torchvision import models
from relive.utils.torch_ext import *
from relive.models.mlp import MLP

class ResNet(nn.Module):
    def __init__(self, out_dim, fix_params=False, running_stats=False, pretrained = False):
        super().__init__()
        self.out_dim = out_dim
        self.resnet = models.resnet18(pretrained  = pretrained )
        if fix_params:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1024)
        self.mlp = MLP(1024, [1024, 512], "relu")
        self.output_lr = nn.Linear(512, out_dim)

        # self.resnet.fc = nn.Linear(512, out_dim)
        
        self.bn_stats(running_stats)
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.mlp(x)
        x = self.output_lr(x)
        return x
    
    def get_embedding(self, x):
        x = self.resnet(x)
        x = self.mlp(x)
        return x


    def bn_stats(self, track_running_stats):
        for m in self.modules():
            if type(m) == nn.BatchNorm2d:
                m.track_running_stats = track_running_stats


if __name__ == '__main__':
    import time
    net = ResNet(128)
    t0 = time.time()
    input = ones(1, 3, 224, 224)
    out = net(input)
    print(time.time() - t0)
    print(out.shape)
