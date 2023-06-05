import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self,in_channels:int,kernel_size:int=3,downsample = None,dropout_p:float=0.05):
        super().__init__()
        out_channels = in_channels
        stride = 1
        if downsample is not None:
            out_channels = in_channels*2
            stride = 2
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,padding=1
                      ,stride=stride),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )
        self.downsample = downsample
        self.out = nn.ReLU(inplace=True)
    def forward(self,x):
        residual = x
        out = self.nn(x)
        if self.downsample:
            residual = self.downsample(out)
        out += residual
        out = self.out(out)
        return out


class MarioButtonsModel(nn.Module):
    def __init__(self,group_size:int=1,res_blocks:int=7):
        super().__init__()
        #input dim is 256x256xgroup_size
        out_channels_first = 16
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=group_size,out_channels=16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16,out_channels=out_channels_first,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        #128x128x16

        self.res_layer = self._make_res_layer(in_channels=out_channels_first,blocks=res_blocks)
        
        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_features=2^res_blocks,out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048,out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512,out_features=8),
        )
        self.init_weights()

    def init_weights(self):
        pass

    def _make_res_layer(self,in_channels:int,blocks:int):
        return nn.Sequential(
            *[ResBlock(in_channels=in_channels*2^i,downsample= nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=in_channels*2^(i+1),kernel_size=3,padding=1,stride=2),
                nn.BatchNorm2d(in_channels*2^(i+1))
            )) for i in range(blocks)],
        )
    def forward(self,x):
        first_out = self.first_conv(x)
        res_out = self.res_blocks(first_out)
        final = self.fc_layers(res_out)
        return final
