# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.quantization


class CPM_condense (nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

        """     Todo: Compute all parameters"""
        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)
        """------Stage 1------"""
        # Padding = (Filter - 1) / 2
        self.conv1_stage1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)
        self.score1 = nn.Conv2d(512, self.k + 1, kernel_size=1)

        """------Stage 2------"""
        self.conv1_stage2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        # Concat layer
        self.concat_stage2 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        self.conv5_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv6_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv7_stage2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.score2 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        """------Stage 3------"""
        self.conv1_stage3 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.concat_stage3 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        self.conv2_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv4_stage3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.score3 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        """------Stage 4------"""
        self.conv1_stage4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.concat_stage4 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        self.conv2_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv4_stage4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.score4 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

    def stage1(self, ori_image):
        x = self.pool1_stage1(
            F.relu(self.conv1_stage1(ori_image), inplace=True))
        x = self.pool2_stage1(F.relu(self.conv2_stage1(x), inplace=True))
        x = self.pool3_stage1(F.relu(self.conv3_stage1(x), inplace=True))
        x = F.relu(self.conv4_stage1(x), inplace=True)
        x = F.relu(self.conv5_stage1(x), inplace=True)
        x = F.relu(self.conv6_stage1(x), inplace=True)
        x = self.score1(x)

        return x

    def stage2(self, feature_image, score_1, pool_center_map):
        x = F.relu(self.conv4_stage2(feature_image), inplace=True)
        x = torch.cat([x, score_1, pool_center_map], dim=1)
        x = F.relu(self.concat_stage2(x), inplace=True)
        x = F.relu(self.conv5_stage2(x), inplace=True)
        x = F.relu(self.conv6_stage2(x), inplace=True)
        x = F.relu(self.conv7_stage2(x), inplace=True)
        x = self.score2(x)

        return x

    def stage3(self, feature_image, score_2, pool_center_map):
        x = F.relu(self.conv1_stage3(feature_image), inplace=True)
        x = torch.cat([x, score_2, pool_center_map], dim=1)
        x = F.relu(self.concat_stage3(x), inplace=True)
        x = F.relu(self.conv2_stage3(x), inplace=True)
        x = F.relu(self.conv3_stage3(x), inplace=True)
        x = F.relu(self.conv4_stage3(x), inplace=True)
        x = self.score3(x)

        return x

    def stage4(self, feature_image, score_3, pool_center_map):
        x = F.relu(self.conv1_stage4(feature_image), inplace=True)
        x = torch.cat([x, score_3, pool_center_map], dim=1)
        x = F.relu(self.concat_stage4(x), inplace=True)
        x = F.relu(self.conv2_stage4(x), inplace=True)
        x = F.relu(self.conv3_stage4(x), inplace=True)
        x = F.relu(self.conv4_stage4(x), inplace=True)
        x = self.score4(x)

        return x

    def middle(self, image):
        x = self.pool1_stage2(F.relu(self.conv1_stage2(image), inplace=True))
        x = self.pool2_stage2(F.relu(self.conv2_stage2(x), inplace=True))
        x = self.pool3_stage2(F.relu(self.conv3_stage2(x), inplace=True))

        return x

    def forward(self, image, center_map):
        pool_center_map = self.pool_center(center_map)

        score_1 = self.stage1(image)

        feature_image = self.middle(image)

        score_2 = self.stage2(feature_image, score_1, pool_center_map)

        score_3 = self.stage3(feature_image, score_2, pool_center_map)

        score_4 = self.stage4(feature_image, score_3, pool_center_map)

        return score_1, score_2, score_3, score_4


class CPM_dpc(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

        """     Todo: Compute all parameters"""
        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)
        """------Stage 1------"""
        # Padding = (Filter - 1) / 2
        #self.conv1_stage1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv1_stage1 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=9, padding=4, groups=3), nn.Conv2d(
            3, 128, kernel_size=1, padding=0, groups=1))
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv2_stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=9, padding=4, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=9, padding=4, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))

        self.pool3_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv4_stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=9, padding=4, groups=128), nn.Conv2d(
            128, 32, kernel_size=1, padding=0, groups=1))
        #self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv5_stage1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=9, padding=4, groups=32), nn.Conv2d(
            32, 512, kernel_size=1, padding=0, groups=1))
        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)

        self.score1 = nn.Conv2d(512, self.k + 1, kernel_size=1)

        """------Stage 2------"""
        #self.conv1_stage2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv1_stage2 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=9, padding=4, groups=3), nn.Conv2d(
            3, 128, kernel_size=1, padding=0, groups=1))
        self.pool1_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv2_stage2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=9, padding=4, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.pool2_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=9, padding=4, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.pool3_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv4_stage2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128), nn.Conv2d(
            128, 32, kernel_size=1, padding=0, groups=1))

        # Concat layer
        self.concat_stage2 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        #self.conv5_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv5_stage2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        #self.conv6_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv6_stage2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))

        self.conv7_stage2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.score2 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        """------Stage 3------"""
        #self.conv1_stage3 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv1_stage3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128), nn.Conv2d(
            128, 32, kernel_size=1, padding=0, groups=1))

        self.concat_stage3 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        #self.conv2_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv2_stage3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        #self.conv3_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3_stage3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.conv4_stage3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)

        self.score3 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        """------Stage 4------"""
        #self.conv1_stage4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv1_stage4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128), nn.Conv2d(
            128, 32, kernel_size=1, padding=0, groups=1))

        self.concat_stage4 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        #self.conv2_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv2_stage4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        #self.conv3_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3_stage4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.conv4_stage4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)

        self.score4 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        """------Stage 5------"""
        #self.conv1_stage5 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv1_stage5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128), nn.Conv2d(
            128, 32, kernel_size=1, padding=0, groups=1))

        self.concat_stage5 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        #self.conv2_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv2_stage5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        #self.conv3_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3_stage5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.conv4_stage5 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.score5 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        """------Stage 6------"""
        #self.conv1_stage6 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv1_stage6 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128), nn.Conv2d(
            128, 32, kernel_size=1, padding=0, groups=1))

        self.concat_stage6 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        #self.conv2_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv2_stage6 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        #self.conv3_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3_stage6 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.conv4_stage6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.score6 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

    def stage1(self, ori_image):
        x = self.pool1_stage1(
            F.relu(self.conv1_stage1(ori_image), inplace=True))
        x = self.pool2_stage1(F.relu(self.conv2_stage1(x), inplace=True))
        x = self.pool3_stage1(F.relu(self.conv3_stage1(x), inplace=True))
        x = F.relu(self.conv4_stage1(x), inplace=True)
        x = F.relu(self.conv5_stage1(x), inplace=True)
        x = F.relu(self.conv6_stage1(x), inplace=True)
        x = self.score1(x)

        return x

    def stage2(self, feature_image, score_1, pool_center_map):
        x = F.relu(self.conv4_stage2(feature_image), inplace=True)
        x = torch.cat([x, score_1, pool_center_map], dim=1)
        x = F.relu(self.concat_stage2(x), inplace=True)
        x = F.relu(self.conv5_stage2(x), inplace=True)
        x = F.relu(self.conv6_stage2(x), inplace=True)
        x = F.relu(self.conv7_stage2(x), inplace=True)
        x = self.score2(x)

        return x

    def stage3(self, feature_image, score_2, pool_center_map):
        x = F.relu(self.conv1_stage3(feature_image), inplace=True)
        x = torch.cat([x, score_2, pool_center_map], dim=1)
        x = F.relu(self.concat_stage3(x), inplace=True)
        x = F.relu(self.conv2_stage3(x), inplace=True)
        x = F.relu(self.conv3_stage3(x), inplace=True)
        x = F.relu(self.conv4_stage3(x), inplace=True)
        x = self.score3(x)

        return x

    def stage4(self, feature_image, score_3, pool_center_map):
        x = F.relu(self.conv1_stage4(feature_image), inplace=True)
        x = torch.cat([x, score_3, pool_center_map], dim=1)
        x = F.relu(self.concat_stage4(x), inplace=True)
        x = F.relu(self.conv2_stage4(x), inplace=True)
        x = F.relu(self.conv3_stage4(x), inplace=True)
        x = F.relu(self.conv4_stage4(x), inplace=True)
        x = self.score4(x)

        return x

    def stage5(self, feature_image, score_4, pool_center_map):
        x = F.relu(self.conv1_stage5(feature_image), inplace=True)
        x = torch.cat([x, score_4, pool_center_map], dim=1)
        x = F.relu(self.concat_stage5(x), inplace=True)
        x = F.relu(self.conv2_stage5(x), inplace=True)
        x = F.relu(self.conv3_stage5(x), inplace=True)
        x = F.relu(self.conv4_stage5(x), inplace=True)
        x = self.score5(x)

        return x

    def stage6(self, feature_image, score_5, pool_center_map):
        x = F.relu(self.conv1_stage6(feature_image), inplace=True)
        x = torch.cat([x, score_5, pool_center_map], dim=1)
        x = F.relu(self.concat_stage6(x), inplace=True)
        x = F.relu(self.conv2_stage6(x), inplace=True)
        x = F.relu(self.conv3_stage6(x), inplace=True)
        x = F.relu(self.conv4_stage6(x), inplace=True)
        x = self.score6(x)

        return x

    def middle(self, image):
        x = self.pool1_stage2(F.relu(self.conv1_stage2(image), inplace=True))
        x = self.pool2_stage2(F.relu(self.conv2_stage2(x), inplace=True))
        x = self.pool3_stage2(F.relu(self.conv3_stage2(x), inplace=True))

        return x

    def forward(self, image, center_map):
        pool_center_map = self.pool_center(center_map)

        score_1 = self.stage1(image)

        feature_image = self.middle(image)

        score_2 = self.stage2(feature_image, score_1, pool_center_map)

        score_3 = self.stage3(feature_image, score_2, pool_center_map)

        score_4 = self.stage4(feature_image, score_3, pool_center_map)

        score_5 = self.stage5(feature_image, score_4, pool_center_map)

        score_6 = self.stage6(feature_image, score_5, pool_center_map)

        return score_1, score_2, score_3, score_4, score_5, score_6

class CPM_dpc_quant(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

        """     Todo: Compute all parameters"""
        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)
        """------Stage 1------"""
        # Padding = (Filter - 1) / 2
        #self.conv1_stage1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv1_stage1 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=9, padding=4, groups=3), nn.Conv2d(
            3, 128, kernel_size=1, padding=0, groups=1))
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv2_stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=9, padding=4, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=9, padding=4, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))

        self.pool3_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv4_stage1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=9, padding=4, groups=128), nn.Conv2d(
            128, 32, kernel_size=1, padding=0, groups=1))
        #self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv5_stage1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=9, padding=4, groups=32), nn.Conv2d(
            32, 512, kernel_size=1, padding=0, groups=1))
        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)

        self.score1 = nn.Conv2d(512, self.k + 1, kernel_size=1)

        """------Stage 2------"""
        #self.conv1_stage2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv1_stage2 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=9, padding=4, groups=3), nn.Conv2d(
            3, 128, kernel_size=1, padding=0, groups=1))
        self.pool1_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv2_stage2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=9, padding=4, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.pool2_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=9, padding=4, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.pool3_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv4_stage2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128), nn.Conv2d(
            128, 32, kernel_size=1, padding=0, groups=1))

        # Concat layer
        self.concat_stage2 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        #self.conv5_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv5_stage2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        #self.conv6_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv6_stage2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))

        self.conv7_stage2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.score2 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        """------Stage 3------"""
        #self.conv1_stage3 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv1_stage3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128), nn.Conv2d(
            128, 32, kernel_size=1, padding=0, groups=1))

        self.concat_stage3 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        #self.conv2_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv2_stage3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        #self.conv3_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3_stage3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.conv4_stage3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)

        self.score3 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        """------Stage 4------"""
        #self.conv1_stage4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv1_stage4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128), nn.Conv2d(
            128, 32, kernel_size=1, padding=0, groups=1))

        self.concat_stage4 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        #self.conv2_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv2_stage4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        #self.conv3_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3_stage4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.conv4_stage4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)

        self.score4 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        """------Stage 5------"""
        #self.conv1_stage5 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv1_stage5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128), nn.Conv2d(
            128, 32, kernel_size=1, padding=0, groups=1))

        self.concat_stage5 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        #self.conv2_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv2_stage5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        #self.conv3_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3_stage5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.conv4_stage5 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.score5 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        """------Stage 6------"""
        #self.conv1_stage6 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv1_stage6 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128), nn.Conv2d(
            128, 32, kernel_size=1, padding=0, groups=1))

        self.concat_stage6 = nn.Conv2d(
            32 + self.k + 2, 128, kernel_size=11, padding=5)

        #self.conv2_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv2_stage6 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        #self.conv3_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3_stage6 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=11, padding=5, groups=128), nn.Conv2d(
            128, 128, kernel_size=1, padding=0, groups=1))
        self.conv4_stage6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.score6 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        #quantization part
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    def stage1(self, ori_image):
        x = self.pool1_stage1(
            F.relu(self.conv1_stage1(ori_image), inplace=True))
        x = self.pool2_stage1(F.relu(self.conv2_stage1(x), inplace=True))
        x = self.pool3_stage1(F.relu(self.conv3_stage1(x), inplace=True))
        x = F.relu(self.conv4_stage1(x), inplace=True)
        x = F.relu(self.conv5_stage1(x), inplace=True)
        x = F.relu(self.conv6_stage1(x), inplace=True)
        x = self.score1(x)

        return x

    def stage2(self, feature_image, score_1, pool_center_map):
        x = F.relu(self.conv4_stage2(feature_image), inplace=True)
        x = torch.cat([x, score_1, pool_center_map], dim=1)
        x = F.relu(self.concat_stage2(x), inplace=True)
        x = F.relu(self.conv5_stage2(x), inplace=True)
        x = F.relu(self.conv6_stage2(x), inplace=True)
        x = F.relu(self.conv7_stage2(x), inplace=True)
        x = self.score2(x)

        return x

    def stage3(self, feature_image, score_2, pool_center_map):
        x = F.relu(self.conv1_stage3(feature_image), inplace=True)
        x = torch.cat([x, score_2, pool_center_map], dim=1)
        x = F.relu(self.concat_stage3(x), inplace=True)
        x = F.relu(self.conv2_stage3(x), inplace=True)
        x = F.relu(self.conv3_stage3(x), inplace=True)
        x = F.relu(self.conv4_stage3(x), inplace=True)
        x = self.score3(x)

        return x

    def stage4(self, feature_image, score_3, pool_center_map):
        x = F.relu(self.conv1_stage4(feature_image), inplace=True)
        x = torch.cat([x, score_3, pool_center_map], dim=1)
        x = F.relu(self.concat_stage4(x), inplace=True)
        x = F.relu(self.conv2_stage4(x), inplace=True)
        x = F.relu(self.conv3_stage4(x), inplace=True)
        x = F.relu(self.conv4_stage4(x), inplace=True)
        x = self.score4(x)

        return x

    def stage5(self, feature_image, score_4, pool_center_map):
        x = F.relu(self.conv1_stage5(feature_image), inplace=True)
        x = torch.cat([x, score_4, pool_center_map], dim=1)
        x = F.relu(self.concat_stage5(x), inplace=True)
        x = F.relu(self.conv2_stage5(x), inplace=True)
        x = F.relu(self.conv3_stage5(x), inplace=True)
        x = F.relu(self.conv4_stage5(x), inplace=True)
        x = self.score5(x)

        return x

    def stage6(self, feature_image, score_5, pool_center_map):
        x = F.relu(self.conv1_stage6(feature_image), inplace=True)
        x = torch.cat([x, score_5, pool_center_map], dim=1)
        x = F.relu(self.concat_stage6(x), inplace=True)
        x = F.relu(self.conv2_stage6(x), inplace=True)
        x = F.relu(self.conv3_stage6(x), inplace=True)
        x = F.relu(self.conv4_stage6(x), inplace=True)
        x = self.score6(x)

        return x

    def middle(self, image):
        x = self.pool1_stage2(F.relu(self.conv1_stage2(image), inplace=True))
        x = self.pool2_stage2(F.relu(self.conv2_stage2(x), inplace=True))
        x = self.pool3_stage2(F.relu(self.conv3_stage2(x), inplace=True))

        return x

    def forward(self, image, center_map):
        image=self.quant(image)
        center_map=self.quant(center_map)

        pool_center_map = self.pool_center(center_map)

        score_1 = self.stage1(image)

        feature_image = self.middle(image)

        score_2 = self.stage2(feature_image, score_1, pool_center_map)

        score_3 = self.stage3(feature_image, score_2, pool_center_map)

        score_4 = self.stage4(feature_image, score_3, pool_center_map)

        score_5 = self.stage5(feature_image, score_4, pool_center_map)

        score_6 = self.stage6(feature_image, score_5, pool_center_map)

        score_1, score_2, score_3, score_4, score_5, score_6=self.dequant(score_1),self.dequant(score_2),self.dequant(score_3),self.dequant(score_4),self.dequant(score_5),self.dequant(score_6)

        return score_1, score_2, score_3, score_4, score_5, score_6