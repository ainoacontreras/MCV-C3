import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeBlock(nn.Module):
    def __init__(self, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(SqueezeBlock, self).__init__()
        self.squeeze = nn.Conv2d(in_channels=squeeze_planes, out_channels=squeeze_planes, kernel_size=1)
        self.expand1x1 = nn.Conv2d(in_channels=squeeze_planes, out_channels=expand1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(in_channels=squeeze_planes, out_channels=expand3x3_planes, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.gelu(self.squeeze(x))
        expand1x1 = F.gelu(self.expand1x1(x))
        expand3x3 = F.gelu(self.expand3x3(x))
        return torch.cat([expand1x1, expand3x3], 1)

class Model(nn.Module):
    def __init__(self, num_classes=8):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        
        self.squeeze_block1 = SqueezeBlock(64, 128, 128)
        self.squeeze_block2 = SqueezeBlock(128, 256, 256)

        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        self.batch_norm2 = nn.BatchNorm2d(num_features=256)
        self.batch_norm3 = nn.BatchNorm2d(num_features=512)

        self.final_conv = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.batch_norm1(x)

        x = self.squeeze_block1(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.batch_norm2(x)

        x = self.squeeze_block2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.batch_norm3(x)

        x = self.final_conv(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)