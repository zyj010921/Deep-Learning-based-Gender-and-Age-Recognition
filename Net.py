# -*- coding = utf-8 -*-
# @Time : 2023/5/7 11:50
# @Author : zyj
# @File : Net.py
# @Software : PyCharm
# 网络构建
import torch


class MyMulitpleTaskNet(torch.nn.Module):
    def __init__(self):
        super(MyMulitpleTaskNet, self).__init__()
        self.cnn_layers = torch.nn.Sequential(
            # x => [3, 64, 64]
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2, 2),
            # x => [32, 32, 32]
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2, 2),
            # x => [64, 16, 16]
            torch.nn.Conv2d(64, 96, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(96),
            torch.nn.MaxPool2d(2, 2),
            # x => [96, 8, 8]
            torch.nn.Conv2d(96, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2, 2),

            # x => [128, 4, 4]
            torch.nn.Conv2d(128, 196, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(196),
            torch.nn.MaxPool2d(2, 2)
            # x => [196, 2, 2]
        )

        # 全局最大池化
        self.global_max_pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        # x => [196, 1, 1]

        # 预测age（回归）
        self.age_fc_layers = torch.nn.Sequential(
            torch.nn.Linear(196, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 1),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Sigmoid()
        )

        # 预测gender（分类）
        self.gender_fc_layers = torch.nn.Sequential(
            torch.nn.Linear(196, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 2)
        )

    def forward(self, x):
        # x => [3, 64, 64]
        x = self.cnn_layers(x)

        # x => [196, 2, 2]
        B, C, H, W = x.size()
        out = self.global_max_pooling(x).view(B, -1)  # -1的值由其他层推断出来

        # 全连接层
        out_age = self.age_fc_layers(out)
        out_gender = self.gender_fc_layers(out)
        return out_age, out_gender