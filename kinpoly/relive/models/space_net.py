import torch
import torch.nn as nn
from torch.autograd import Variable


class SpaceNet(nn.Module):

    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

    class UnFlatten(nn.Module):
        def forward(self, input, size=256):
            return input.view(input.size(0), size, 2, 2, 2)


    def __init__(self, h_dim=2048, z_dim=128, device=None, dtype=None):
        super(SpaceNet, self).__init__()
        self.device=device
        self.dtype=dtype
        self.z_dim = z_dim

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.conv1_bn = nn.BatchNorm3d(32)
        self.act1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.conv2_bn = nn.BatchNorm3d(64)
        self.act2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.conv3_bn = nn.BatchNorm3d(128)
        self.act3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.conv4_bn = nn.BatchNorm3d(256)
        self.act4 = nn.LeakyReLU()
        self.pool4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.flatten = self.Flatten()
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.unflatten = self.UnFlatten()
        self.unpool1 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.deconv1 = nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.deconv1_bn = nn.BatchNorm3d(128)
        self.deact1 = nn.LeakyReLU()
        self.unpool2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.deconv2 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.deconv2_bn = nn.BatchNorm3d(64)
        self.deact2 = nn.LeakyReLU()
        self.unpool3 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.deconv3 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.deconv3_bn = nn.BatchNorm3d(32)
        self.deact3 = nn.LeakyReLU()
        self.unpool4 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.deconv4 = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.deconv4_bn = nn.BatchNorm3d(1)
        self.deact4 = nn.Sigmoid()
  
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), dtype=self.dtype, device=self.device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def encoder(self, x):
        x = self.pool1(self.act1(self.conv1_bn(self.conv1(x))))
        x = self.pool2(self.act2(self.conv2_bn(self.conv2(x))))
        x = self.pool3(self.act3(self.conv3_bn(self.conv3(x))))
        x = self.pool4(self.act4(self.conv4_bn(self.conv4(x))))
        x = self.flatten(x)
        return x

    def decoder(self, x):
        x = self.unflatten(x)
        x = self.deact1(self.deconv1_bn(self.deconv1(self.unpool1(x))))
        x = self.deact2(self.deconv2_bn(self.deconv2(self.unpool2(x))))
        x = self.deact3(self.deconv3_bn(self.deconv3(self.unpool3(x))))
        x = self.deact4(self.deconv4_bn(self.deconv4(self.unpool4(x))))
        #x = self.sigmoid(x)
        return x

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar