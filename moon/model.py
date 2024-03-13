
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    
    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(84, 256)


    def forward(self, x):
        x = self.fc1(x)
        return x


class MOON(nn.Module):
    def __init__(self):
        super(MOON, self).__init__()
        self.base_encoder = BaseEncoder()
        self.projection_head = ProjectionHead()
        self.output_layer = nn.Linear(256, 10)


    def forward(self, x):
        x = self.base_encoder(x)
        x = self.projection_head(x)
        x = self.output_layer(x)
        return x
