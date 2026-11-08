# Copyright (c) 2026 The GIFSplat Authors.
# SPDX-License-Identifier: CC-BY-NC-4.0
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ObservationFeatures(nn.Module):
    """Frozen ResNet18 pyramid: 64 + 64 + 128 channels."""

    def __init__(self, pretrained=True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.conv1, self.bn1, self.relu = net.conv1, net.bn1, net.relu
        self.maxpool, self.layer1, self.layer2 = net.maxpool, net.layer1, net.layer2
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        x0 = self.conv1((images.float() - self.mean) / self.std)
        x1 = self.layer1(self.maxpool(self.relu(self.bn1(x0))))
        return x0, x1, self.layer2(x1)
