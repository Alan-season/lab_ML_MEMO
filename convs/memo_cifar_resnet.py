'''
Reference:
https://github.com/khurramjaved96/incremental-learning/blob/autoencoders/model/resnet32.py
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsampleA(nn.Module):
    """下采样模块"""
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        # 确保步幅为2
        assert stride == 2
        # 定义平均池化层
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        # 对输入进行平均池化，经过池化后，图像的边长缩小至原来的一半
        x = self.avg(x) # shape: [128, 16, 32, 32] -> (stage 2)[128, 16, 16, 16] -> (final stage)[128, 32, 8, 8]
        # 拼接零填充的特征图
        return torch.cat((x, x.mul(0)), 1)  # shape: (stage 2)[128, 32, 16, 16], (final stage)[128, 64, 8, 8]

class ResNetBasicblock(nn.Module):
    """基本残差块，一块残差块包含两个卷积层"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        # 第一个卷积层
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        # 第二个卷积层
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        # 下采样模块
        self.downsample = downsample

    def forward(self, x):
        residual = x

        # 通过第一个卷积层、批归一化和ReLU激活
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        # 通过第二个卷积层和批归一化
        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        # 如果存在下采样模块，进行下采样（在stage 2和final stage的第一个残差块进行下采样）
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接并通过ReLU激活
        return F.relu(residual + basicblock, inplace=True)

class GeneralizedResNet_cifar(nn.Module):
    def __init__(self, block, depth, channels=3):
        super(GeneralizedResNet_cifar, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        # 这里的depth指的是ResNet的深度，除去初始卷积层和最后的全连接层，剩下的均为残差块的卷积层
        # 一块基本残差块包含两个卷积层，需要将(depth - 2) // 2块基本残差块分成3大块，前两块（前两个stage）作为通用块
        # 以ResNet32为例，则这里的layer_block=5，表示一个通用块或专用块由5个基本残差块构成
        layer_blocks = (depth - 2) // 6

        # 初始卷积层，将3通道的图像卷积成16通道的数据，其他维不变
        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)

        # 输出维度
        self.out_dim = 64 * block.expansion
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """使用基本残差块构建残差网络（通用块）"""
        downsample = None
        # 当 stride 不等于 1 或者 输入通道数 不等于 输出通道数 * 扩展系数 时，进行下采样
        # 目的：减小图像尺寸同时增加通道数，提取更深层抽象的特征
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 通过初始卷积层和批归一化，再通过ReLU激活
        x = self.conv_1_3x3(x)  # shape: [128, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        # 通过通用块1 (stage 1) 和通用块2 (stage 2)
        x_1 = self.stage_1(x)  # shape: [128, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # shape: [128, 32, 16, 16]
        return x_2
    
class SpecializedResNet_cifar(nn.Module):
    def __init__(self, block, depth, inplanes=32, feature_dim=64):
        """初始化专用块，配置与通用块大致相同"""
        super(SpecializedResNet_cifar, self).__init__()
        self.inplanes = inplanes
        self.feature_dim = feature_dim
        layer_blocks = (depth - 2) // 6
        self.final_stage = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride=2):
        """使用基本残差块构建残差网络（专用块）"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, base_feature_map):
        # 通过专用块
        final_feature_map = self.final_stage(base_feature_map)  # shape: [128, 64, 8, 8]
        # 全局平均池化（步长为8）
        pooled = self.avgpool(final_feature_map)    # shape: [128, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # shape: [128, 64]
        return features

#For cifar & MEMO
def get_resnet8_a2fc():
    basenet = GeneralizedResNet_cifar(ResNetBasicblock,8)
    adaptivenet = SpecializedResNet_cifar(ResNetBasicblock,8)
    return basenet,adaptivenet

def get_resnet14_a2fc():
    basenet = GeneralizedResNet_cifar(ResNetBasicblock,14)
    adaptivenet = SpecializedResNet_cifar(ResNetBasicblock,14)
    return basenet,adaptivenet

def get_resnet20_a2fc():
    basenet = GeneralizedResNet_cifar(ResNetBasicblock,20)
    adaptivenet = SpecializedResNet_cifar(ResNetBasicblock,20)
    return basenet,adaptivenet

def get_resnet26_a2fc():
    basenet = GeneralizedResNet_cifar(ResNetBasicblock,26)
    adaptivenet = SpecializedResNet_cifar(ResNetBasicblock,26)
    return basenet,adaptivenet

def get_resnet32_a2fc():
    # 通用块
    basenet = GeneralizedResNet_cifar(ResNetBasicblock,32)  # -> line 52
    # 专用块
    adaptivenet = SpecializedResNet_cifar(ResNetBasicblock,32)  # -> line 99
    return basenet,adaptivenet