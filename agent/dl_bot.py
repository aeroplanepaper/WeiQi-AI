import torch
import torch.nn as nn
import torch.nn.functional as F
from Agent import Agent

class ConvNet2D(nn.Module):
    def __init__(self):
        super(ConvNet2D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(4, 4), padding=1, stride=(1, 1)),
            nn.ReLU()
            # nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            nn.ReLU()
            # nn.Sigmoid()
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=60, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            # nn.ReLU()
            nn.Sigmoid()
        )

        self.layer6 = nn.Sequential(
            nn.Linear(in_features=960, out_features=361),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # print(x.shape)
        x = x.view(x.shape[0],-1)
        x = self.layer6(x)
        return x
