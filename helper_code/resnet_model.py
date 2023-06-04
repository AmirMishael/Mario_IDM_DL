import torch
import torch.nn as nn

class ResnetModel(nn.Module):
    def __init__(self,group_size:int=3,use_color=False,use_pretrained:bool=False):
        super().__init__()
        self.group_size = group_size
        
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18',num_classes=1000, pretrained=use_pretrained)

        #input_channels = 3*group_size if use_color else 1*group_size

        # if use_pretrained:
        #     assert input_channels <= 3
        #     #self.resnet.conv1 = nn.Conv2d(in_channels=group_size,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)
        #     self.resnet.fc = nn.Linear(in_features=512,out_features=8)
        # else:
        #     self.resnet.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)
        #     self.resnet.fc = nn.Linear(in_features=512,out_features=8)
        #     self.init_weights()
        
        #self.resnet.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)
        input_channels = 3 if use_color else 1
        conv1 = nn.Conv3d(in_channels=input_channels,out_channels=3,kernel_size=(group_size,3,3),stride=1,padding=(1,1,1),bias=True)
        self.resnet.fc = nn.Linear(in_features=512,out_features=8)

        self.nn = nn.Sequential(
            conv1,
            nn.BatchNorm3d(3),
            nn.ReLU(),
            nn.Dropout(0.1),
            self.resnet
        )
        self.init_weights()

    
    def init_weights(self):
        pass
    def forward(self,x):
        return self.nn(x)


