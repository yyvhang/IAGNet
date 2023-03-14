import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from model.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class PointNet_Estimation(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(PointNet_Estimation, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [
                                             32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=134+additional_channel, mlp=[128, 128])

        self.classifier = nn.ModuleList()
        for i in range(num_classes):
            classifier = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.Conv1d(128, 1, 1)
            )
            self.classifier.append(classifier)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, xyz):
        # Set Abstraction layers
        xyz = xyz.contiguous()
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:

            l0_xyz = xyz
            l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat(
            [l0_xyz, l0_points], 1), l1_points)

        score = self.classifier[0](l0_points)
        for index, classifier in enumerate(self.classifier):
            if index == 0:
                continue
            score_ = classifier(l0_points)
            score = torch.cat((score, score_), dim=1)
        score = score.permute(0, 2, 1).contiguous()
        score = self.sigmoid(score)
        return score