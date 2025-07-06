import torch
import torch.nn as nn
import torch.optim as optim

class ClassificationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
                                nn.Conv2d(kernel_size=3, in_channels=1, out_channels=32, stride=1, padding=0),
                                nn.ReLU()
                            )
        self.conv3 = nn.Sequential(
                            nn.Conv2d(kernel_size=3, in_channels=32, out_channels=128, stride=1, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2)
                            )
        self.conv4 = nn.Sequential(
                            nn.Conv2d(kernel_size=3, in_channels=128, out_channels=256, stride=1, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2)
                            )
        self.flat = nn.Flatten()
        self.ll1 = nn.Sequential(nn.LazyLinear(out_features=400),
                                 nn.ReLU()
                                 )
        self.ll2 = nn.Sequential(nn.LazyLinear(out_features=400),
                                 nn.ReLU()
                                 )
        self.ll3 = nn.LazyLinear(out_features=2)

    def forward(self, x):
        outs = None
        x = self.conv1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flat(x)
        x = self.ll1(x)
        x = self.ll2(x) 
        outs = self.ll3(x)
        return outs
