# Change detection head

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from model.cd_modules.psp import _PSPModule
from model.cd_modules.se import ChannelSpatialSELayer

import numpy as np


def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''
    Get the number of input layers to the change detection head.
    '''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3:  # 256 x 256
            in_channels += inner_channel * channel_multiplier[0]
        elif scale < 6:  # 128 x 128
            in_channels += inner_channel * channel_multiplier[1]
        elif scale < 9:  # 64 x 64
            in_channels += inner_channel * channel_multiplier[2]
        elif scale < 12:  # 32 x 32
            in_channels += inner_channel * channel_multiplier[3]
        elif scale < 15:  # 16 x 16
            in_channels += inner_channel * channel_multiplier[4]
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14')
    return in_channels


class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim * len(time_steps), dim, 1)
            if len(time_steps) > 1
            else None,
            nn.ReLU()
            if len(time_steps) > 1
            else None,
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SR(nn.Module):
    '''Spatial reasoning module'''

    # codes from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(SR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        ''' inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = x + self.gamma * out

        return out


class CotSR(nn.Module):
    # codes derived from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(CotSR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        ''' inputs :
                x1 : input feature maps( B X C X H X W)
                x2 : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x1.size()

        q1 = self.query_conv1(x1).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width * height)
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width * height)

        q2 = self.query_conv2(x2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(m_batchsize, -1, width * height)
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width * height)

        energy1 = torch.bmm(q1, k2)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v2, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        energy2 = torch.bmm(q2, k1)
        attention2 = self.softmax(energy2)
        out2 = torch.bmm(v1, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)

        out1 = x1 + self.gamma1 * out1
        out2 = x2 + self.gamma2 * out2

        return out1, out2


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class BiSRNet(nn.Module):
    def __init__(self):
        super(BiSRNet, self).__init__()
        self.SiamSR = SR(128)
        self.CotSR = CotSR(128)
        self.down_conv = nn.Sequential(nn.Conv2d(256, 128, kernel_size = 3, padding = 1), nn.ReLU(), nn.BatchNorm2d(128))
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
                                          nn.Conv2d(64, 2, kernel_size=3, padding=1))
        initialize_weights(self.SiamSR, self.resCD, self.down_conv, self.CotSR, self.classifierCD)
        self.downsample = nn.Sequential(nn.MaxPool2d(2), nn.MaxPool2d(2))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def CD_forward(self, x1, x2):
        b, c, h, w = x1.size()
        x = torch.cat([x1, x2], 1)
        x = self.resCD(x)
        change = self.classifierCD(x)
        return change

    def forward(self, x1, x2):
        x_size = x1.size()
        x1_identity, x2_identity = x1, x2

        x1, x2 = self.downsample(x1), self.downsample(x2)
        x1 = self.SiamSR(x1)
        x2 = self.SiamSR(x2)
        x1, x2 = self.CotSR(x1, x2)
        x1 = F.upsample(x1, x_size[2:], mode='bilinear')
        x2 = F.upsample(x2, x_size[2:], mode='bilinear')
        
        x1, x2 = x1 + x1_identity, x2 + x2_identity
        
        change = self.CD_forward(x1, x2)

        return F.upsample(change, x_size[2:], mode='bilinear')
    
class FDAF(nn.Module):
    """Flow Dual-Alignment Fusion Module.

    Args:
        in_channels (int): Input channels of features.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='IN'),
                 act_cfg=dict(type='GELU')):
        super(FDAF, self).__init__()
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        conv_cfg = None
        norm_cfg = dict(type='IN')
        act_cfg = dict(type='GELU')

        kernel_size = 5
        self.flow_make = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      bias=True, groups=in_channels * 2),
            nn.InstanceNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, 4, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x1, x2, fusion_policy=None):
        """Forward function."""

        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(x1, f1) - x2
        x2_feat = self.warp(x2, f2) - x1

        if fusion_policy == "Attention":
            return x1_feat, x2_feat
        else:
            return torch.abs(x1_feat - x2_feat)

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output

class cd_fdaf_attention(nn.Module):
    '''
    Change detection head (version 2).
    '''

    def __init__(self, feat_scales, out_channels=2, inner_channel=None, channel_multiplier=None, img_size=256,
                 time_steps=None):
        super(cd_fdaf_attention, self).__init__()

        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True)
        self.feat_scales = feat_scales
        self.in_channels = get_in_channels(feat_scales, inner_channel, channel_multiplier)
        self.img_size = img_size
        self.time_steps = time_steps

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        for i in range(0, len(self.feat_scales)):
            dim = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)

            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )

            if i != len(self.feat_scales) - 1:
                dim_out = get_in_channels([self.feat_scales[i + 1]], inner_channel, channel_multiplier)
                self.decoder.append(
                    AttentionBlock(dim=dim, dim_out=dim_out)
                )
            else:
                self.decoder.append(FDAF(in_channels=dim))
        # Final classification head
        # Input channel 'Must' be 128
        self.classifier = BiSRNet()

    def forward(self, feats_A, feats_B):
        # Decoder
        lvl = 0
        for layer in self.decoder:
            if isinstance(layer, Block):
                f_A = feats_A[0][self.feat_scales[lvl]]
                f_B = feats_B[0][self.feat_scales[lvl]]
                for i in range(1, len(self.time_steps)):
                    f_A = torch.cat((f_A, feats_A[i][self.feat_scales[lvl]]), dim=1)
                    f_B = torch.cat((f_B, feats_B[i][self.feat_scales[lvl]]), dim=1)
    
                class_f_A, class_f_B = layer(f_A), layer(f_B)
                if lvl != 0:
                    class_f_A, class_f_B = class_f_A + x_A, class_f_B + x_B
                lvl += 1
            elif isinstance(layer, AttentionBlock):
                class_f_A, class_f_B = layer(class_f_A), layer(class_f_B)
                x_A = F.interpolate(class_f_A, scale_factor=2, mode="bilinear")
                x_B = F.interpolate(class_f_B, scale_factor=2, mode="bilinear")
            else:
                x1, x2 = layer(x_A, x_B, "Attention")

        # Classifier
        cm = self.classifier(x1, x2)

        return cm
