#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_distance(x, y, normalized=False, channel_first=False):
    """计算两个点集之间的欧氏距离。

    Args:
        x: torch.Tensor (B, N, C) 或 (B, C, N)
        y: torch.Tensor (B, M, C) 或 (B, C, M)
        normalized (bool=False): 如果点云已归一化，则有"x2 + y2 = 1"，所以"d2 = 2 - 2xy"
        channel_first (bool=False): 如果为True，则点云形状为(B, C, N)

    Returns:
        dist: torch.Tensor (B, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(B, C, N) -> (B, N, C)] x (B, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (B, N, C) x [(B, M, C) -> (B, C, M)]
    
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (B, N, C) 或 (B, C, N) -> (B, N) -> (B, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (B, M, C) 或 (B, C, M) -> (B, M) -> (B, 1, M)
        sq_distances = x2 - 2 * xy + y2
    
    sq_distances = sq_distances.clamp(min=0.0)  # 确保距离非负
    return sq_distances


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        """Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings


class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        """Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        """计算几何结构编码。

        Args:
            points: torch.Tensor (B, N, 3), 输入点云

        Returns:
            embeddings: torch.Tensor (B, N, D), 几何结构编码
        """
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        return embeddings


class FeatureFusionModule(nn.Module):
    def __init__(self, dgcnn_dim, geo_dim, output_dim, fusion_type='concat'):
        super(FeatureFusionModule, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            self.proj = nn.Conv1d(dgcnn_dim + geo_dim, output_dim, 1)
        elif fusion_type == 'add':
            self.proj_dgcnn = nn.Conv1d(dgcnn_dim, output_dim, 1)
            self.proj_geo = nn.Conv1d(geo_dim, output_dim, 1)
        elif fusion_type == 'attention':
            # 实现注意力融合机制
            self.query = nn.Conv1d(dgcnn_dim, output_dim, 1)
            self.key = nn.Conv1d(geo_dim, output_dim, 1)
            self.value = nn.Conv1d(geo_dim, output_dim, 1)
            self.gamma = nn.Parameter(torch.zeros(1))
        
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, dgcnn_feat, geo_feat):
        """融合DGCNN特征和几何特征。

        Args:
            dgcnn_feat: torch.Tensor (B, C1, N), DGCNN特征
            geo_feat: torch.Tensor (B, C2, N), 几何特征

        Returns:
            fused_feat: torch.Tensor (B, C, N), 融合后的特征
        """
        if self.fusion_type == 'concat':
            fused_feat = torch.cat([dgcnn_feat, geo_feat], dim=1)
            fused_feat = self.relu(self.bn(self.proj(fused_feat)))
        elif self.fusion_type == 'add':
            dgcnn_feat = self.proj_dgcnn(dgcnn_feat)
            geo_feat = self.proj_geo(geo_feat)
            fused_feat = self.relu(self.bn(dgcnn_feat + geo_feat))
        elif self.fusion_type == 'attention':
            # 注意力融合机制
            q = self.query(dgcnn_feat)
            k = self.key(geo_feat)
            v = self.value(geo_feat)
            
            # 计算注意力权重
            energy = torch.bmm(q.transpose(1, 2), k)  # [B, N, N]
            attention = F.softmax(energy, dim=2)      # [B, N, N]
            
            # 应用注意力
            geo_feat_attended = torch.bmm(v, attention.transpose(1, 2))
            fused_feat = self.relu(self.bn(dgcnn_feat + self.gamma * geo_feat_attended))
            
        return fused_feat
