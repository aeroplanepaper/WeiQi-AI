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

class ConvNet2DForSimple(nn.Module):
    def __init__(self):
        super(ConvNet2DForSimple, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(8, 8), padding=3, stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout(0.5)
            # nn.Sigmoid()
        )
        # self.layer2 = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=2, stride=2)
        # )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=20, kernel_size=(7, 7), padding=3, stride=(1, 1)),
            nn.ReLU(),
            # nn.Sigmoid()
            nn.Dropout(0.5)
        )

        # self.layer4 = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=2, stride=2)
        # )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(6, 6), padding=2, stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout(0.5)
            # nn.Sigmoid()
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(5, 5), padding=1, stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout(0.5)
            # nn.Sigmoid()
        )

        self.layer6 = nn.Sequential(
            nn.Linear(in_features=4500, out_features=1024),
            # nn.Softmax(dim=1)
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer7 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=391),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        # x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # x = self.layer4(x)
        # print(x.shape)
        x = self.layer5(x)
        x = self.layer8(x)
        x = x.view(x.shape[0],-1)
        x = self.layer6(x)
        x = self.layer7(x)
        return x