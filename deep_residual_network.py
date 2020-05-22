"""
深度残差模型案例
Author:WUH
Time:2020年2月23日18:40:45
"""
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# 将多个transform组合起来使用
transform = transforms.Compose([
    transforms.Resize(40),  # 按照规定的尺寸重新调节PIL.Image
    transforms.RandomHorizontalFlip(),  # 随机水平反转
    transforms.RandomCrop(32),
    transforms.ToTensor()
])
# 训练数据和测试数据
train_dataset = dsets.CIFAR10(root='./data/', train=True,
                              transform=transform, download=True)
test_dataset = dsets.CIFAR10(root='./data/', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)  # 第一个卷积层为3x3二维卷积
        self.bn1 = nn.BatchNorm2d(out_channels)  # ?
        self.relu = nn.ReLU(inplace=True)  # ?
        self.conv2 = conv3x3(out_channels, out_channels)  # ?
        self.bn2 = nn.BatchNorm2d(out_channels)  # ?
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)  # 输入通道为RGB三通道
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])  # ?
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block())







