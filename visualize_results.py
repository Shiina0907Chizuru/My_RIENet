#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import argparse
import open3d as o3d
from data_modelnet40 import ModelNet40
from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict
from util import npmat2euler
from common.math import se3

def parse_args():
    parser = argparse.ArgumentParser(description='点云配准结果可视化')
    parser.add_argument('--config', type=str, default='configs/modelnet.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/modelnet/models/model.best.t7', help='模型权重路径')
    parser.add_argument('--idx', type=int, default=0, help='要可视化的样本索引')
    parser.add_argument('--show_transformed', type=bool, default=True, help='是否显示预测变换后的点云')
    parser.add_argument('--show_gt_transformed', type=bool, default=True, help='是否显示真实变换后的点云')
    return parser.parse_args()

def parse_args_from_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as fd:
        args = yaml.safe_load(fd)
        args = EasyDict(d=args)
    return args

def create_point_cloud(points, color=[1, 0, 0]):
    """创建Open3D点云对象"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd

def transform_point_cloud(points, rotation, translation):
    """应用旋转和平移变换到点云"""
    return np.matmul(points, rotation.T) + translation.reshape(1, 3)

def visualize_registration_result(source_points, target_points, rotation_pred, translation_pred, 
                                 rotation_gt=None, translation_gt=None, show_transformed=True, 
                                 show_gt_transformed=True):
    """可视化配准结果"""
    # 创建原始点云
    source_pcd = create_point_cloud(source_points, [1, 0, 0])  # 红色：源点云
    target_pcd = create_point_cloud(target_points, [0, 0, 1])  # 蓝色：目标点云
    
    geometries = [source_pcd, target_pcd]
    
    # 添加预测变换后的点云
    if show_transformed:
        transformed_points = transform_point_cloud(source_points, rotation_pred, translation_pred)
        transformed_pcd = create_point_cloud(transformed_points, [0, 1, 0])  # 绿色：预测变换后的点云
        geometries.append(transformed_pcd)
    
    # 添加真实变换后的点云
    if show_gt_transformed and rotation_gt is not None and translation_gt is not None:
        gt_transformed_points = transform_point_cloud(source_points, rotation_gt, translation_gt)
        gt_transformed_pcd = create_point_cloud(gt_transformed_points, [1, 1, 0])  # 黄色：真实变换后的点云
        geometries.append(gt_transformed_pcd)
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coordinate_frame)
    
    # 可视化
    o3d.visualization.draw_geometries(geometries, 
                                     window_name="点云配准结果可视化",
                                     width=1200, 
                                     height=800,
                                     left=50,
                                     top=50)

def calculate_metrics(rotation_gt, translation_gt, rotation_pred, translation_pred):
    """计算配准指标"""
    pred_transform = np.concatenate([rotation_pred, translation_pred.reshape(-1, 3, 1)], axis=-1)
    gt_transform = np.concatenate([rotation_gt, translation_gt.reshape(-1, 3, 1)], axis=-1)
    
    pred_transform_tensor = torch.from_numpy(pred_transform)
    gt_transform_tensor = torch.from_numpy(gt_transform)
    
    # 计算残差变换
    concatenated = se3.concatenate(se3.inverse(gt_transform_tensor), pred_transform_tensor)
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = (torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi).detach().cpu().numpy()
    residual_transmag = concatenated[:, :, 3].norm(dim=-1).detach().cpu().numpy()
    
    # 计算欧拉角误差
    rotation_pred_euler = npmat2euler(rotation_pred)
    rotation_gt_euler = npmat2euler(rotation_gt)
    r_mae = np.mean(np.abs(rotation_pred_euler - rotation_gt_euler))
    t_mae = np.mean(np.abs(translation_gt - translation_pred))
    
    return {
        "旋转误差(度)": residual_rotdeg[0],
        "平移误差(米)": residual_transmag[0],
        "欧拉角MAE": r_mae,
        "平移MAE": t_mae
    }

def main():
    args = parse_args()
    config_args = parse_args_from_yaml(args.config)
    
    # 加载测试数据集
    test_loader = DataLoader(
        ModelNet40(num_points=config_args.n_points,
                  num_subsampled_points=config_args.n_subsampled_points,
                  partition='test', 
                  gaussian_noise=config_args.gaussian_noise,
                  unseen=config_args.unseen, 
                  rot_factor=config_args.rot_factor),
        batch_size=1, shuffle=False, drop_last=False
    )
    
    # 获取指定索引的数据样本
    for i, data in enumerate(test_loader):
        if i == args.idx:
            src, target, rotation_ab, translation_ab = data
            break
    
    # 转换为numpy数组
    src_np = src[0].numpy()
    target_np = target[0].numpy()
    rotation_gt = rotation_ab[0].numpy()
    translation_gt = translation_ab[0].numpy()
    
    # 加载预测结果
    if os.path.exists(args.checkpoint):
        print(f"从 {args.checkpoint} 加载预测结果...")
        # 这里应该加载模型并进行预测，但为简化起见，我们假设已经有了预测结果
        # 实际应用中，您需要加载模型、运行模型并获取预测结果
        checkpoint = torch.load(args.checkpoint)
        print(f"模型训练到第 {checkpoint['epoch']} 轮，最佳结果: {checkpoint['best_result']}")
        
        # 这里我们简单模拟一个预测结果
        # 在实际应用中，您应该加载模型并使用模型预测
        # 如果您有预训练模型，请替换下面的代码
        
        # 模拟的预测结果（应用一些噪声到真实值）
        # 在实际应用中，这部分应该替换为真实的模型预测
        noise_level = 0.05  # 噪声级别
        rotation_pred = rotation_gt + np.random.randn(*rotation_gt.shape) * noise_level
        translation_pred = translation_gt + np.random.randn(*translation_gt.shape) * noise_level
        
        print("注意：当前使用模拟的预测结果。在实际应用中，请使用模型进行预测。")
    else:
        print(f"找不到检查点文件: {args.checkpoint}")
        print("使用真实值作为'预测值'进行可视化...")
        rotation_pred = rotation_gt
        translation_pred = translation_gt
    
    # 计算评估指标
    metrics = calculate_metrics(rotation_gt.reshape(1, 3, 3), 
                              translation_gt.reshape(1, 3), 
                              rotation_pred.reshape(1, 3, 3), 
                              translation_pred.reshape(1, 3))
    
    # 打印评估指标
    print("\n配准评估指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    
    # 可视化结果
    visualize_registration_result(
        src_np, target_np,
        rotation_pred, translation_pred,
        rotation_gt, translation_gt,
        args.show_transformed,
        args.show_gt_transformed
    )

if __name__ == "__main__":
    main()
