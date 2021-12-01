import torch
import torch.nn as nn
import torch.nn.functional as F
from Agent import Agent

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(11, 2, 3, 1, 1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool3d(2, 1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(2, 6, 3, 1, 2),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2, 1)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(6, 60, 3, 1, 1),
            nn.ReLU()
        )

        self.layer6 = nn.Linear(240, 722)

        self.layer7 = nn.ReLU()

        self.layer8 = nn.Linear(722, 361)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(1, -1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return F.log_softmax(x)
