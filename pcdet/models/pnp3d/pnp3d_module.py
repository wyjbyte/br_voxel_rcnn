import torch
import torch.nn as nn
import torch.nn.functional as F


class PnP3D(nn.Module):
    def __init__(self, in_channels, k=16, conv_cfg=None):
        """
        PnP3D模块实现
        Args:
            in_channels: 输入特征通道数
            k: KNN搜索的邻居数
            conv_cfg: 卷积配置
        """
        super().__init__()
        self.k = k
        self.conv_cfg = conv_cfg

        # 局部上下文融合网络
        self.local_context = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 全局双线性正则化网络
        self.global_context = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

    def knn(self, x, k):
        """K最近邻搜索"""
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        distance = -xx - inner - xx.transpose(2, 1)

        idx = distance.topk(k=k, dim=-1)[1]
        return idx

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, 3, N) 点云坐标
            features: (B, C, N) 点云特征
        Returns:
            enhanced_features: (B, C, N) 增强后的特征
        """
        batch_size, _, num_points = xyz.shape

        # 1. KNN搜索获取每个点的邻居
        idx = self.knn(xyz, k=self.k)

        # 2. 局部上下文特征增强
        local_features = self.local_context(features)

        # 3. 全局上下文特征增强
        global_features = self.global_context(features)

        # 4. 特征融合 (简化版本)
        enhanced_features = local_features + global_features

        return enhanced_features