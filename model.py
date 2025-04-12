#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from pointnet2 import pointnet2_utils
import torch.nn.functional as F
from torch.autograd import Variable
from util import transform_point_cloud
from chamfer_loss import *
from utils import pairwise_distance_batch, PointNet, Pointer, get_knn_index, Discriminator, feature_extractor, \
    compute_rigid_transformation, get_keypoints


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.num_keypoints = args.n_keypoints
        self.weight_function = Discriminator(args)
        self.fuse = Pointer()
        self.nn_margin = args.nn_margin

    def forward(self, *input):
        """
            Args:
                src: Source point clouds. Size (B, 3, N)
                tgt: target point clouds. Size (B, 3, M)
                src_embedding: Features of source point clouds. Size (B, C, N)
                tgt_embedding: Features of target point clouds. Size (B, C, M)
                src_idx: Nearest neighbor indices. Size [B * N * k]
                k: Number of nearest neighbors.
                src_knn: Coordinates of nearest neighbors. Size [B, N, K, 3]
                i: i-th iteration.
                tgt_knn: Coordinates of nearest neighbors. Size [B, M, K, 3]
                src_idx1: Nearest neighbor indices. Size [B * N * k]
                idx2:  Nearest neighbor indices. Size [B, M, k]
                k1: Number of nearest neighbors.
            Returns:
                R/t: rigid transformation.
                src_keypoints, tgt_keypoints: Selected keypoints of source and target point clouds. Size (B, 3, num_keypoint)
                src_keypoints_knn, tgt_keypoints_knn: KNN of keypoints. Size [b, 3, num_kepoints, k]
                loss_scl: Spatial Consistency loss.
        """
        src = input[0]
        tgt = input[1]

        src_embedding = input[2]
        tgt_embedding = input[3]
        src_idx = input[4]
        k = input[5]
        src_knn = input[6]  # [b, n, k, 3]
        i = input[7]
        tgt_knn = input[8]  # [b, n, k, 3]
        src_idx1 = input[9]  # [b * n * k1]
        idx2 = input[10]  # [b, m, k1]
        k1 = input[11]

        batch_size, num_dims_src, num_points = src.size()
        batch_size, _, num_points_tgt = tgt.size()
        batch_size, _, num_points = src_embedding.size()

        ########################## Matching Map Refinement Module ##########################
        distance_map = pairwise_distance_batch(src_embedding, tgt_embedding)  # [b, n, m]计算两个点云特征之间距离
        print(f"distance_map维度: {distance_map.shape}, 值范围: [{distance_map.min().item():.4f}, {distance_map.max().item():.4f}]")
        # point-wise matching map
        scores = torch.softmax(-distance_map, dim=2)  # [b, n, m]  Eq. (1) 为什么负距离 这样归一化转化为概率分布以后越近分数越高
        print(f"scores维度: {scores.shape}, 值范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
        # neighborhood-wise matching map
        src_knn_scores = scores.view(batch_size * num_points, -1)[src_idx1, :]#n*k,m
        print(f"src_knn_scores初始维度: {src_knn_scores.shape}")
        src_knn_scores = src_knn_scores.view(batch_size, num_points, k1, num_points_tgt)  # [b, n, k, m]
        print(f"src_knn_scores重塑维度: {src_knn_scores.shape}")
        print(f"idx2维度: {idx2.shape}, 索引范围: [{idx2.min().item()}, {idx2.max().item()}]")
        print(f"batch_size: {batch_size}, num_points: {num_points}, k1: {k1}, num_points_tgt: {num_points_tgt}")
        if idx2.max().item() >= num_points_tgt * k1:
            print(f"警告: idx2最大值{idx2.max().item()}超出了num_points_tgt*k1={num_points_tgt*k1}!")

        # src_knn_scores = pointnet2_utils.gather_operation(src_knn_scores.view(batch_size * num_points, k1, num_points_tgt), \
        #                                                   idx2.view(batch_size, 1, num_points_tgt * k1).repeat(1,
        #                                                                                                    num_points,
        #                                                                                                    1).view(
        #                                                       batch_size * num_points, num_points_tgt * k1).int()).view(
        #     batch_size, \
        #     num_points, k1, num_points_tgt, k1)[:, :, 1:, :, 1:].sum(-1).sum(2) / (k1 - 1)  # Eq. (2)
        # src_knn_scores = pointnet2_utils.gather_operation(src_knn_scores.view(batch_size * num_points, k1, num_points),\
        #     idx2.view(batch_size, 1, num_points * k1).repeat(1, num_points, 1).view(batch_size * num_points, num_points * k1).int()).view(batch_size,\
        #         num_points, k1, num_points, k1)[:, :, 1:, :, 1:].sum(-1).sum(2) / (k1-1) # Eq. (2)


        try:
            # 修改gather_operation调用，分步执行
            src_view = src_knn_scores.view(batch_size * num_points, k1, num_points_tgt)
            idx_view = idx2.view(batch_size, 1, num_points_tgt * k1).repeat(1, num_points, 1)
            idx_view_flat = idx_view.view(batch_size * num_points, num_points_tgt * k1).int()
    
            # 检查索引范围
            print(f"src_view: {src_view.shape}")
            print(f"idx_view_flat: {idx_view_flat.shape}, 范围: [{idx_view_flat.min().item()}, {idx_view_flat.max().item()}]")
    
            # 确保索引不超出范围
            idx_view_flat = torch.clamp(idx_view_flat, 0, src_view.shape[2] * src_view.shape[1] - 1)
    
            # 然后调用gather_operation
            result = pointnet2_utils.gather_operation(src_view, idx_view_flat)
            result_view = result.view(batch_size, num_points, k1, num_points_tgt, k1)
            src_knn_scores = result_view[:, :, 1:, :, 1:].sum(-1).sum(2) / (k1 - 1)
    
            # 安全打印
            print(f"gather后src_knn_scores维度: {src_knn_scores.shape}")
            print(f"值范围: [{src_knn_scores.min().item() if not torch.isnan(src_knn_scores).any() else 'NaN'}, {src_knn_scores.max().item() if not torch.isnan(src_knn_scores).any() else 'NaN'}]")
    
        except Exception as e:
            print(f"gather_operation发生错误: {e}") 

        print(f"src_knn_scores是否包含NaN: {torch.isnan(src_knn_scores).any().item()}")
        print(f"src_knn_scores是否包含Inf: {torch.isinf(src_knn_scores).any().item()}")
        print(f"src_knn_scores是否包含负数: {torch.any(src_knn_scores < 0).item()}")
        print(f"src_knn_scores是否包含正数: {torch.any(src_knn_scores > 0).item()}")
        print(f"src_knn_scores是否包含0: {torch.any(src_knn_scores == 0).item()}")
        print(f"gather后src_knn_scores维度: {src_knn_scores.shape}, 值范围: [{src_knn_scores.min().item():.4f}, {src_knn_scores.max().item():.4f}]") #这里出错了！RuntimeError: CUDA error: an illegal memory access was encountered

        src_knn_scores = self.nn_margin - src_knn_scores  # 超参a 0.7 修正分数为什么取负，因为s分数越高 匹配程度越高，所需要修正的越小
        print(f"修正后src_knn_scores: 范围[{src_knn_scores.min().item():.4f}, {src_knn_scores.max().item():.4f}]")
       
       


        refined_distance_map = torch.exp(src_knn_scores) * distance_map
        print(f"refined_distance_map维度: {refined_distance_map.shape}, 范围[{refined_distance_map.min().item():.4f}, {refined_distance_map.max().item():.4f}]")
        refined_matching_map = torch.softmax(-refined_distance_map, dim=2)  # [b, n, m] Eq. (3)
        print(f"refined_matching_map维度: {refined_matching_map.shape}, 范围[{refined_matching_map.min().item():.4f}, {refined_matching_map.max().item():.4f}]")

        # pseudo correspondences of source point clouds (pseudo target point clouds)
        src_corr = torch.matmul(tgt, refined_matching_map.transpose(2, 1).contiguous())  # [b,3,n] Eq. (4)
        print(f"src_corr维度: {src_corr.shape}, 范围[{src_corr.min().item():.4f}, {src_corr.max().item():.4f}]")
        ############################## Inlier Evaluation Module ##############################
        # neighborhoods of pseudo target point clouds
        src_knn_corr = src_corr.transpose(2, 1).contiguous().view(batch_size * num_points, -1)[src_idx, :]
        src_knn_corr = src_knn_corr.view(batch_size, num_points, k, num_dims_src)  # [b, n, k, 3]
        print(f"src_knn_corr维度: {src_knn_corr.shape}, 范围[{src_knn_corr.min().item():.4f}, {src_knn_corr.max().item():.4f}]")

        # edge features of the pseudo target neighborhoods and the source neighborhoods
        knn_distance = src_corr.transpose(2, 1).contiguous().unsqueeze(2) - src_knn_corr  # [b, n, k, 3]#伪目标点云距离
        print(f"knn_distance维度: {knn_distance.shape}, 范围[{knn_distance.min().item():.4f}, {knn_distance.max().item():.4f}]")
        src_knn_distance = src.transpose(2, 1).contiguous().unsqueeze(2) - src_knn  # [b, n, k, 3]#源目标点云距离
        print(f"src_knn_distance维度: {src_knn_distance.shape}, 范围[{src_knn_distance.min().item():.4f}, {src_knn_distance.max().item():.4f}]")
        # inlier confidence
        weight = self.weight_function(knn_distance, src_knn_distance)  # [b, 1, n] # Eq. (7)
        print(f"weight维度: {weight.shape}, 范围[{weight.min().item():.4f}, {weight.max().item():.4f}], 是否有NaN: {torch.isnan(weight).any().item()}")
        #print("src, src_corr",src.shape, src_corr.shape)
        # compute rigid transformation
        print(f"SVD计算前检查: src={src.shape}, src_corr={src_corr.shape}, weight={weight.shape}")
        print(f"src值范围: [{src.min().item():.4f}, {src.max().item():.4f}], 是否有NaN: {torch.isnan(src).any().item()}")
        print(f"src_corr值范围: [{src_corr.min().item():.4f}, {src_corr.max().item():.4f}], 是否有NaN: {torch.isnan(src_corr).any().item()}")
        print(f"weight值范围: [{weight.min().item():.4f}, {weight.max().item():.4f}], 是否有NaN: {torch.isnan(weight).any().item()}")
        R, t = compute_rigid_transformation(src, src_corr, weight)  # weighted SVD

        ########################### Preparation for the Loss Function #########################
        # choose k keypoints with highest weights
        key_num =  num_points//2
        if key_num  <900:
            key_num  = key_num  * 2
        src_topk_idx, src_keypoints, tgt_keypoints = get_keypoints(src, src_corr, weight,
                                                                   key_num)  # 通过置信度 排序 选择关键点

        # spatial consistency loss
        idx_tgt_corr = torch.argmax(refined_matching_map, dim=-1).int()  # [b, n]
        identity = torch.eye(num_points_tgt).cuda().unsqueeze(0).repeat(batch_size, 1, 1)  # [b, m, m]
        one_hot_number = pointnet2_utils.gather_operation(identity, idx_tgt_corr)  # [b, m, n]
        src_keypoints_idx = src_topk_idx.repeat(1, num_points_tgt, 1)  # [b, m, num_keypoints]
        keypoints_one_hot = torch.gather(one_hot_number, dim=2, index=src_keypoints_idx).transpose(2, 1).reshape(
            batch_size * key_num , num_points_tgt)
        # [b, m, num_keypoints] - [b, num_keypoints, m] - [b * num_keypoints, m]
        predicted_keypoints_scores = torch.gather(refined_matching_map.transpose(2, 1), dim=2,
                                                  index=src_keypoints_idx).transpose(2, 1).reshape(
            batch_size * key_num , num_points_tgt)
        loss_scl = (-torch.log(predicted_keypoints_scores + 1e-15) * keypoints_one_hot).sum(1).mean()

        # neighorhood information
        src_keypoints_idx2 = src_topk_idx.unsqueeze(-1).repeat(1, 3, 1, k)  # [b, 3, num_keypoints, k]
        tgt_keypoints_knn = torch.gather(knn_distance.permute(0, 3, 1, 2), dim=2,
                                         index=src_keypoints_idx2)  # [b, 3, num_kepoints, k]

        src_transformed = transform_point_cloud(src, R, t.view(batch_size, 3))
        src_transformed_knn_corr = src_transformed.transpose(2, 1).contiguous().view(batch_size * num_points, -1)[
                                   src_idx, :]
        src_transformed_knn_corr = src_transformed_knn_corr.view(batch_size, num_points, k,
                                                                 num_dims_src)  # [b, n, k, 3]

        knn_distance2 = src_transformed.transpose(2, 1).contiguous().unsqueeze(
            2) - src_transformed_knn_corr  # [b, n, k, 3]
        src_keypoints_knn = torch.gather(knn_distance2.permute(0, 3, 1, 2), dim=2,
                                         index=src_keypoints_idx2)  # [b, 3, num_kepoints, k]
        return R, t.view(batch_size, 3), src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, loss_scl


class LossFunction(nn.Module):
    def __init__(self, args):
        super(LossFunction, self).__init__()
        self.criterion2 = ChamferLoss()
        self.criterion = nn.MSELoss(reduction='sum')
        self.GAL = GlobalAlignLoss()
        self.margin = args.loss_margin

    def forward(self, *input):
        """
            Compute global alignment loss and neighorhood consensus loss
            Args:
                src_keypoints: Keypoints of source point clouds. Size (B, 3, num_keypoint)
                tgt_keypoints: Keypoints of target point clouds. Size (B, 3, num_keypoint)
                rotation_ab: Size (B, 3, 3)
                translation_ab: Size (B, 3)
                src_keypoints_knn: [b, 3, num_kepoints, k]
                tgt_keypoints_knn: [b, 3, num_kepoints, k]
                k: Number of nearest neighbors.
                src_transformed: Transformed source point clouds. Size (B, 3, N)
                tgt: Target point clouds. Size (B, 3, M)
            Returns:
                neighborhood_consensus_loss
                global_alignment_loss
        """
        src_keypoints = input[0]
        tgt_keypoints = input[1]
        rotation_ab = input[2]
        translation_ab = input[3]
        src_keypoints_knn = input[4]
        tgt_keypoints_knn = input[5]
        k = input[6]
        src_transformed = input[7]
        tgt = input[8]

        batch_size = src_keypoints.size()[0]

        global_alignment_loss = self.GAL(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1), self.margin)

        transformed_srckps_forward = transform_point_cloud(src_keypoints, rotation_ab, translation_ab)
        keypoints_loss = self.criterion(transformed_srckps_forward, tgt_keypoints)
        knn_consensus_loss = self.criterion(src_keypoints_knn, tgt_keypoints_knn)
        neighborhood_consensus_loss = knn_consensus_loss / k + keypoints_loss

        return neighborhood_consensus_loss, global_alignment_loss


class LossFunction_kitti(nn.Module):
    def __init__(self, args):
        super(LossFunction_kitti, self).__init__()
        self.criterion2 = ChamferLoss()
        self.criterion = nn.MSELoss(reduction='none')
        self.GAL = GlobalAlignLoss()
        self.margin = args.loss_margin

    def forward(self, *input):
        """
            Compute global alignment loss and neighorhood consensus loss
            Args:
                src_keypoints: Selected keypoints of source point clouds. Size (B, 3, num_keypoint)
                tgt_keypoints: Selected keypoints of target point clouds. Size (B, 3, num_keypoint)
                rotation_ab: Size (B, 3, 3)
                translation_ab: Size (B, 3)
                src_keypoints_knn: [b, 3, num_kepoints, k]
                tgt_keypoints_knn: [b, 3, num_kepoints, k]
                k: Number of nearest neighbors.
                src_transformed: Transformed source point clouds. Size (B, 3, N)
                tgt: Target point clouds. Size (B, 3, M)
            Returns:
                neighborhood_consensus_loss
                global_alignment_loss
        """
        src_keypoints = input[0]
        tgt_keypoints = input[1]
        rotation_ab = input[2]
        translation_ab = input[3]
        src_keypoints_knn = input[4]
        tgt_keypoints_knn = input[5]
        k = input[6]
        src_transformed = input[7]
        tgt = input[8]

        global_alignment_loss = self.GAL(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1), self.margin)

        transformed_srckps_forward = transform_point_cloud(src_keypoints, rotation_ab, translation_ab)
        keypoints_loss = self.criterion(transformed_srckps_forward, tgt_keypoints).sum(1).sum(1).mean()
        knn_consensus_loss = self.criterion(src_keypoints_knn, tgt_keypoints_knn).sum(1).sum(1).mean()
        neighborhood_consensus_loss = knn_consensus_loss + keypoints_loss

        return neighborhood_consensus_loss, global_alignment_loss
def get_index(coord, origin, voxel_size):
    return (coord - origin) / voxel_size
def ca_mask_pdb(grid_shape, x_origin, y_origin, z_origin, x_voxel, y_voxel, z_voxel, src):
    # 确保 grid_shape 是整数元组
    #grid_shape = tuple(grid_shape.cpu().numpy().astype(int))
    device = src.device  # 获取 src 所在的设备
    grid_shape = grid_shape.to(device)
    # 将其他张量移到与 src 相同的设备
    x_origin = x_origin.to(device)
    y_origin = y_origin.to(device)
    z_origin = z_origin.to(device)
    x_voxel = x_voxel.to(device)
    y_voxel = y_voxel.to(device)
    z_voxel = z_voxel.to(device)

    # 创建 data 数组
    data = torch.zeros((grid_shape[0][0],grid_shape[0][1],grid_shape[0][2]), dtype=torch.float32, device=src.device, requires_grad=True)

    resolution = 3.5
    k = (torch.pi / resolution) ** 2
    C = (k / torch.pi) ** 1.5
    bw2 = 6  # bandwidth squared
    bw = torch.sqrt(torch.tensor(bw2, device=src.device))

    # 提前计算源坐标的索引
    coords_indices = torch.stack([
        torch.floor((src[:, 0] - x_origin) / x_voxel).long(),
        torch.floor((src[:, 1] - y_origin) / y_voxel).long(),
        torch.floor((src[:, 2] - z_origin) / z_voxel).long()
    ], dim=1)

    # 预计算 coord_lower 和 coord_upper
    coord_lower = torch.floor(coords_indices - bw).long()
    coord_upper = torch.ceil(coords_indices + bw).long()

    # 使用 torch.meshgrid 代替 np.indices
    for i, coord in enumerate(src):
        iz, jy, kx = coords_indices[i]
        if 0 <= iz < grid_shape[0][0] and 0 <= jy < grid_shape[0][1] and 0 <= kx < grid_shape[0][2]:
            lower = coord_lower[i]
            upper = coord_upper[i]

            # 将 lower 和 upper clamp到合法范围内
            lower_clamp = torch.clamp(lower, min=0, max = min(grid_shape[0][0], grid_shape[0][1], grid_shape[0][2])-1)
            upper_clamp = torch.clamp(upper, min=0, max = min(grid_shape[0][0], grid_shape[0][1], grid_shape[0][2])-1)

            # 生成网格
            z_idx, y_idx, x_idx = torch.meshgrid(
                torch.arange(lower_clamp[0], upper_clamp[0] + 1, device=src.device),
                torch.arange(lower_clamp[1], upper_clamp[1] + 1, device=src.device),
                torch.arange(lower_clamp[2], upper_clamp[2] + 1, device=src.device),
                indexing='ij'
            )

            # 计算距离平方
            dist_sq = (z_idx - iz) ** 2 + (y_idx - jy) ** 2 + (x_idx - kx) ** 2

            # 使用掩码筛选符合距离条件的点
            mask = dist_sq <= bw2
            prob = torch.exp(-k * dist_sq) * mask

            # 将符合条件的概率值写入 data
            new_data = data.clone()
            new_data[z_idx[mask], y_idx[mask], x_idx[mask]] = torch.maximum(
                new_data[z_idx[mask], y_idx[mask], x_idx[mask]],
                prob[mask]
            )
            data = new_data

    return data
class RSCC_Loss(nn.Module):
    def __init__(self, src, tgt, grid_shape, x_origin, y_origin, z_origin, x_voxel, nstart):
        super(RSCC_Loss, self).__init__()
        self.src = src
        self.tgt = tgt
        self.grid_shape = grid_shape
        self.x_origin = x_origin
        self.y_origin = y_origin
        self.z_origin = z_origin
        self.x_voxel = x_voxel
        self.y_voxel = x_voxel
        self.z_voxel = x_voxel

    def forward(self):
        src_mask = ca_mask_pdb(self.grid_shape, self.x_origin, self.y_origin, self.z_origin,
                               self.x_voxel, self.y_voxel, self.z_voxel, self.src.squeeze(0).permute(1, 0))
        tgt_mask = ca_mask_pdb(self.grid_shape, self.x_origin, self.y_origin, self.z_origin,
                               self.x_voxel, self.y_voxel, self.z_voxel, self.tgt.squeeze(0).permute(1, 0))

        # 直接使用 PyTorch 操作，避免重新创建张量
        srcc = src_mask.unsqueeze(0)
        tgtt = tgt_mask.unsqueeze(0)
        print("tgtt",tgtt.shape)
        # Global Alignment Loss (EDCS Loss)
        cc_loss = self.global_alignment_loss(tgtt, srcc)

        return cc_loss

    def global_alignment_loss(self, tgt, src):

        tgt = tgt * src
        src = src * src
        rho_em_mean = torch.sum(tgt, dim=(1, 2, 3), keepdim=True)
        rho_mo_mean = torch.sum(src, dim=(1, 2, 3), keepdim=True)
        '''
        src_mean = torch.mean(src, dim=(1, 2, 3), keepdim=True)
        tgt_mean = torch.mean(tgt, dim=(1, 2, 3), keepdim=True)

        # Center the features
        src_centered = src - src_mean
        tgt_centered = tgt - tgt_mean

        # Compute numerator and denominator for EDCS
        numerator = torch.sum(src_centered * tgt_centered, dim=(1, 2, 3))
        denominator_src = torch.sum(src_centered ** 2, dim=(1, 2, 3))
        denominator_tgt = torch.sum(tgt_centered ** 2, dim=(1, 2, 3))
        denominator = torch.sqrt(denominator_src * denominator_tgt)
        '''


        # EDCS loss
        edcs_loss = rho_mo_mean - rho_em_mean

        return torch.mean(edcs_loss)

class RIENET(nn.Module):
    def __init__(self, args):
        super(RIENET, self).__init__()
        self.emb_nn = feature_extractor(args=args)
        self.single_point_embed = PointNet()
        self.forwards = SVDHead(args=args)
        self.iter = args.n_iters
        if args.dataset == 'kitti':  # or args.dataset == 'icl_nuim':
            self.loss = LossFunction_kitti(args)
        else:
            self.loss = LossFunction(args)
        self.list_k1 = args.list_k1
        self.list_k2 = args.list_k2
        #self.cc_loss = RSCC_Loss(grid_shape, x_origin, y_origin, z_origin, x_voxel, y_voxel, z_voxel,nstart)

    def forward(self, *input):
        """
            feature extraction.
            Args:
                src = input[0]: Source point clouds. Size [B, 3, N]
                tgt = input[1]: Target point clouds. Size [B, 3, N]
            Returns:
                rotation_ab_pred: Size [B, 3, 3]
                translation_ab_pred: Size [B, 3]
                global_alignment_loss
                consensus_loss
                spatial_consistency_loss
        """
        # source, target, rotation, translation,gird_shape,x_origin,y_origin,z_origin,x_voxel,nstart
        src = input[0]
        #print("src", src.shape)
        tgt = input[1]

        gird_shape = input[2]

        x_origin = input[3]
        y_origin = input[4]
        z_origin = input[5]
        x_voxel  = input[6]
        nstart   = input[7]


        #print("tgt", tgt.shape)
        batch_size, _, _ = src.size()
        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1,1)  # 初始化旋转平移矩阵
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        print(f"初始化旋转矩阵(rotation_ab_pred)维度: {rotation_ab_pred.shape}")
        print(f"初始化平移向量(translation_ab_pred)维度: {translation_ab_pred.shape}")
        global_alignment_loss, consensus_loss, spatial_consistency_loss,cc_loss = 0.0, 0.0, 0.0, 0.0

        for i in range(self.iter):
            print(f"\n第{i+1}次迭代:")
            # 打印k1和k2的值
            print(f"当前k1值: {self.list_k1[i]}, k2值: {self.list_k2[i]}")
            src_embedding, src_idx, src_knn, _ = self.emb_nn(src, self.list_k1[i])  # 32个点
            print(f"src_embedding维度: {src_embedding.shape}")  # 特征维度
            print(f"src_idx维度: {src_idx.shape}")  # 邻居索引
            print(f"src_knn维度: {src_knn.shape}")  # 邻居坐标
            tgt_embedding, _, tgt_knn, _ = self.emb_nn(tgt, self.list_k1[i])
            print(f"tgt_embedding维度: {tgt_embedding.shape}")
            print(f"tgt_knn维度: {tgt_knn.shape}")

            print(f"调用get_knn_index前维度: src={src.shape}, tgt={tgt.shape}")
            src_idx1, _ = get_knn_index(src, self.list_k2[i])  # 8个点？？？
            _, tgt_idx = get_knn_index(tgt, self.list_k2[i])
            print(f"src_idx1维度: {src_idx1.shape}")  # 第二组邻居索引
            print(f"tgt_idx维度: {tgt_idx.shape}")  # 目标点云邻居索引

            rotation_ab_pred_i, translation_ab_pred_i, src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, spatial_consistency_loss_i \
                = self.forwards(src, tgt, src_embedding, tgt_embedding, src_idx, self.list_k1[i], src_knn, i, tgt_knn, \
                                src_idx1, tgt_idx, self.list_k2[i])
            print(f"矩阵乘法前 - rotation_ab_pred维度: {rotation_ab_pred.shape}")    
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            print(f"矩阵乘法后 - rotation_ab_pred维度: {rotation_ab_pred.shape}")
            print(f"平移计算前 - translation_ab_pred维度: {translation_ab_pred.shape}")
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i
            print(f"squeeze后加上translation_ab_pred_i - 最终translation_ab_pred维度: {translation_ab_pred.shape}")
            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

            neighborhood_consensus_loss_i, global_alignment_loss_i = self.loss(src_keypoints, tgt_keypoints, \
                                                                               rotation_ab_pred_i,
                                                                               translation_ab_pred_i, src_keypoints_knn,
                                                                               tgt_keypoints_knn, self.list_k2[i], src,
                                                                               tgt)
            cc_loss_fn = RSCC_Loss(src,tgt,gird_shape,x_origin,y_origin,z_origin,x_voxel,nstart)
            # cc = cc_loss_fn.forward()  # 或者简写为 cc = cc_loss_fn()
            #print(f"Loss value: {cc}, requires_grad: {cc.requires_grad}")

            global_alignment_loss += global_alignment_loss_i
            consensus_loss += neighborhood_consensus_loss_i
            spatial_consistency_loss += spatial_consistency_loss_i
            # cc_loss = cc_loss+cc
            cc_loss = torch.tensor(0.0, device=src.device, requires_grad=True)

        return rotation_ab_pred, translation_ab_pred, global_alignment_loss, consensus_loss, spatial_consistency_loss ,cc_loss