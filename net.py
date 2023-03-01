import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU
import cc_atten
import os

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_relu=False, use_norm= False):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = kernel_size//2)
        # self.dropout = nn.Dropout2d(p=0.5)
        self.use_relu = use_relu
        self.use_norm = use_norm
        self.bn = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        out = self.conv2d(x)
        if self.use_norm:
            x =self.bn(out)
        if self.use_relu:
            x = F.relu(out, inplace=False)

        return out



# Dense Block unit
class resdnet_Block(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=1, kernel_size_neck=3, stride=1, width=4):
        super(resdnet_Block, self).__init__()
        self.width = width

        self.conv1 = ConvLayer(in_channels, self.width * 4, kernel_size, stride, use_relu=True, use_norm=False)
        
        convs1 = []
        for i in range(8):
            convs1.append(ConvLayer(self.width * 4, self.width, kernel_size_neck, stride, use_relu=True, use_norm= False))
        self.convs1 = nn.ModuleList(convs1)

        self.conv2 =  ConvLayer(self.width * 4, out_channels, kernel_size, stride, use_relu=True, use_norm = False)
        
        self.conv3 = ConvLayer(in_channels, out_channels, kernel_size, stride, use_relu=True,  use_norm = False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv3( x)
        x = self.conv1(x)
        
        [x1, x2, x3, x4] = torch.split(x, self.width, 1)
        y1 = self.convs1[0](x)
        y2 = self.convs1[1](torch.cat((x2, x3, x4, y1), dim=1))
        y3 = self.convs1[2](torch.cat((x3, x4, y1, y2), dim=1))
        y4 = self.convs1[3](torch.cat((x4, y1 ,y2 ,y3), dim=1))
        x1 = self.convs1[4](torch.cat((y1 ,y2 ,y3, y4), dim=1))
        x2 = self.convs1[5](torch.cat((y2 ,y3, y4, x1), dim=1))
        x3 = self.convs1[6](torch.cat((y3, y4, x1, x2), dim=1))
        x4 = self.convs1[7](torch.cat((y4 ,x1, x2,x3), dim=1))
        
        out2 =  self.conv2(torch.cat([x1, x2, x3, x4],dim=1))

        return self.relu(out1+out2) 


# DenseFuse network
class ResCCNet_atten_fuse(nn.Module):
    def __init__(self,in_channel=1, out_channel=1):
        super(ResCCNet_atten_fuse, self).__init__()
        resblock = resdnet_Block
               
        width = [4, 8, 16]
        
        channels = [16, 32, 64]
        decoder_channel = [16, 32, 64, 112]
        
        kernel_size_1 = 1
        kernel_size_2 = 3
        stride = 1

        # encoder
        self.encoder_conv = nn.Sequential(nn.Conv2d(in_channel, channels[0], kernel_size=3, padding=1), ReLU(inplace=True))
        self.RDB1 = resblock(channels[0], channels[0], kernel_size_1, kernel_size_2, stride, width[0])
        self.RDB2 = resblock(channels[1], channels[1], kernel_size_1, kernel_size_2, stride, width[1])
        self.RDB3 = resblock(channels[2], channels[2], kernel_size_1, kernel_size_2, stride, width[2])


        # decoder
        self.conv1 = ConvLayer(decoder_channel[3], decoder_channel[2], kernel_size_2, stride,use_relu=True)
        self.conv2 = ConvLayer(decoder_channel[2], decoder_channel[1], kernel_size_2, stride,use_relu=True)
        self.conv3 = ConvLayer(decoder_channel[1], decoder_channel[0], kernel_size_2, stride,use_relu=True)
        self.conv4 = ConvLayer(decoder_channel[0], out_channel, kernel_size_2, stride,use_relu=True)


    def encoder(self, input):
        x0 = self.encoder_conv(input)
        x1 = self.RDB1(x0)
        x2 = self.RDB2(torch.cat((x0, x1), dim=1))
        x3 = self.RDB3(torch.cat((x0, x1, x2), dim=1))

        return torch.cat([x1, x2, x3], dim=1)

    def fusion(self, en1, en2 ,strategy_type='cc_atten', kernel_size=[8,1]):
        if strategy_type == 'cc_atten':
            # attention weight
            fusion_function = cc_atten.attention_fusion_weight
            f_0 = fusion_function(en1, en2,kernel_size)  
        elif strategy_type == 'channel':
            # attention weight
            fusion_function = cc_atten.channel_fusion
            f_0 = fusion_function(en1, en2)
        elif strategy_type == 'spatial':
            # attention weight
            fusion_function = cc_atten.spatial_fusion
            f_0 = fusion_function(en1, en2,kernel_size)
        elif strategy_type =='add':
            # addition
            fusion_function = cc_atten.addition_fusion
            f_0 = fusion_function(en1, en2)


#  fusion_type = ['attention_avg', 'attention_max', 'attention_nuclear']

        return f_0

    def decoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = self.conv4(x)
        return out

