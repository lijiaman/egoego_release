import torch
from torch import nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, out_dim, fix_params=False, running_stats=False, pretrained = False):
        super().__init__()
        self.out_dim = out_dim
        self.resnet = models.resnet18(pretrained  = pretrained )
        if fix_params:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_dim)
        self.bn_stats(running_stats)
        
    def forward(self, x):
        return self.resnet(x)

    def bn_stats(self, track_running_stats):
        for m in self.modules():
            if type(m) == nn.BatchNorm2d:
                m.track_running_stats = track_running_stats

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

if __name__ == '__main__':
    net = ResNet(128)
    input = ones(1, 3, 224, 224)
    out = net(input)
    print(out.shape)
