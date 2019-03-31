import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
import pickle

import PyTorch_models.transformer_util as util
import PyTorch_models.Config as conf
import utils.reader as reader


class CNN_3D(nn.Module):
    # 输入:(batch_size,9,50,50,12)
    # 对应着:(batch, in_depth, in_height, in_width, in_channels)

    # 而pytorch 中对应的输入为:(batch_size,12,9,50,50)
    # 对应着:(batch,in_channels,in_depth,in_height,in_width)
    def __init__(self, out_channels_0=32, out_channels_1=16):
        super(CNN_3D, self).__init__()
        self.conv3d = nn.Conv3d(12, out_channels_0, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        init.uniform_(self.conv3d.weight, -0.01, 0.01)
        init.zeros_(self.conv3d.bias)

        self.pool3d = nn.MaxPool3d((3, 3, 3), stride=(3, 3, 3), padding=(0, 1, 1))

        self.conv3d_second = nn.Conv3d(out_channels_0, out_channels_1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        init.uniform_(self.conv3d_second.weight, -0.01, 0.01)
        init.zeros_(self.conv3d_second.bias)

        self.pool3d_second = nn.MaxPool3d((3, 3, 3), stride=(3, 3, 3), padding=(0, 1, 1))

    def forward(self, input):
        # input shape:(batch,in_channels,in_depth,in_height,in_width)
        # (batch_size,12,9,50,50)
        conv_0 = self.conv3d(input)
        conv_0 = F.elu(conv_0)
        pooling_0 = self.pool3d(conv_0)

        conv_1 = self.conv3d_second(pooling_0)
        conv_1 = F.elu(conv_1)
        pooling_1 = self.conv3d_second(conv_1)
        pooling_1 = pooling_1.view(pooling_1.size(0), -1)

        return pooling_1
