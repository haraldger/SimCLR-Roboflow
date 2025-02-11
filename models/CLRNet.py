import torch
import torch.nn as nn
from torchvision.models import resnet50

from training.utils import get_backbone

class CLRNet(nn.Module):
    def __init__(self, size=50):
        super(CLRNet, self).__init__()

        self.backbone = get_backbone(size)
        hidden_size = self.backbone.fc.in_features
        self.simclr_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.backbone.fc = nn.Identity() # Remove classification head

    def forward(self, x):
        x = self.backbone(x)
        x = self.simclr_head(x)
        return x
