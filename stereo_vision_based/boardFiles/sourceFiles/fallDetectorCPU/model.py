import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), dilation=(dilation, 1), padding=(dilation, 0))
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        residual = x
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        out += residual
        return out

class FallModel(nn.Module):
    def __init__(self):
        super(FallModel, self).__init__()
        self.first_conv = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=(1, 1), padding=(1, 0), bias=False)
        self.first_bn = nn.BatchNorm2d(num_features=512)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.25)

        self.res1 = ResidualBlock(in_channels=512, dilation=1)
        self.res2 = ResidualBlock(in_channels=512, dilation=1)
        self.res3 = ResidualBlock(in_channels=512, dilation=1)
        self.res4 = ResidualBlock(in_channels=512, dilation=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1)

    def forward(self, x):
        # print(f"x.shape = {x.shape}")
        assert len(x.shape) == 4  # input: batchsize-frame-joints-xyz
        # assert len(x.shape) == 3  # input: batchsize-frame-joints-xyz
        B, F, J, C = x.shape
        # B, J, C = x.shape
        # x = x.reshape((B, F, J * C))
        # x = x.permute(0, 2, 1)
        # x = x.permute(2, 1, 0)
        x = x.permute(0, 3, 1, 2)  
        # print(f"x.shape = {x.shape}")

        x = self.first_conv(x)  # print(f"x.shape = {x.shape}")
        x = self.first_bn(x)  # print(f"x.shape = {x.shape}")
        x = self.relu(x)  # print(f"x.shape = {x.shape}")
        x = self.drop(x)  # print(f"x.shape = {x.shape}")

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avgpool(x)
        x = self.conv5(x)
        x = x.squeeze(dim=3)
        x = x.squeeze(dim=2)

        return x
