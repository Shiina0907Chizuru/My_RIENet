#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def random_rotation_matrix():
    """生成随机旋转矩阵"""
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.astype(np.float32)

def random_translation_vector(scale=10.0):
    """生成随机平移向量"""
    return np.random.uniform(-scale, scale, size=(3,)).astype(np.float32)

def calculate_centroid(point_cloud):
    """计算点云的质心"""
    if point_cloud.shape[0] == 3 and point_cloud.shape[1] != 3:
        # 点云格式为[3, N]
        centroid = np.mean(point_cloud, axis=1)
    else:
        # 点云格式为[N, 3]
        centroid = np.mean(point_cloud, axis=0)
    return centroid

def transform_point_cloud(point_cloud, rotation, translation):
    """应用旋转和平移变换到点云"""
    # 确保点云是[3, N]格式
    is_3N_format = False
    if point_cloud.shape[1] == 3 and point_cloud.shape[0] != 3:
        point_cloud = point_cloud.T
        is_3N_format = True
    
    # 应用变换
    transformed_cloud = np.matmul(rotation, point_cloud) + translation.reshape(3, 1)
    
    # 还原原始格式
    if is_3N_format:
        transformed_cloud = transformed_cloud.T
    
    return transformed_cloud

def calculate_centroid_aligned_transformation(original_rotation, original_translation, centroid_diff):
    """计算质心对齐的变换矩阵"""
    centroid_aligned_rotation = original_rotation.copy()
    centroid_aligned_translation = original_translation + centroid_diff
    return centroid_aligned_rotation, centroid_aligned_translation

def generate_centroid_aligned_cloud(source_cloud, target_cloud, rotation):
    """生成质心对齐的点云"""
    # 保存原始格式
    source_is_3N = source_cloud.shape[0] == 3 and source_cloud.shape[1] != 3
    target_is_3N = target_cloud.shape[0] == 3 and target_cloud.shape[1] != 3
    
    # 转换为一致的格式进行计算
    src_for_calc = source_cloud if not source_is_3N else source_cloud.T
    tgt_for_calc = target_cloud if not target_is_3N else target_cloud.T
    
    # 计算质心
    source_centroid = np.mean(src_for_calc, axis=0)
    target_centroid = np.mean(tgt_for_calc, axis=0)
    
    # 计算质心偏差
    centroid_diff = source_centroid - target_centroid
    
    # 创建对齐的点云
    aligned_cloud = tgt_for_calc.copy()
    aligned_cloud = aligned_cloud + centroid_diff
    
    # 转回原始格式
    if target_is_3N:
        aligned_cloud = aligned_cloud.T
    
    return aligned_cloud, centroid_diff

def visualize_point_clouds(source, target, aligned=None, title="点云比较"):
    """可视化点云"""
    fig = plt.figure(figsize=(15, 5))
    
    # 确保点云是[N, 3]格式用于绘图
    if source.shape[0] == 3 and source.shape[1] != 3:
        source = source.T
    if target.shape[0] == 3 and target.shape[1] != 3:
        target = target.T
    if aligned is not None and aligned.shape[0] == 3 and aligned.shape[1] != 3:
        aligned = aligned.T
    
    # 创建子图
    if aligned is None:
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
    else:
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
    
    # 绘制源点云
    ax1.scatter(source[:, 0], source[:, 1], source[:, 2], c='blue', s=5)
    ax1.set_title("源点云")
    
    # 绘制目标点云
    ax2.scatter(target[:, 0], target[:, 1], target[:, 2], c='red', s=5)
    ax2.set_title("变换后点云")
    
    # 绘制对齐后的点云
    if aligned is not None:
        ax3.scatter(aligned[:, 0], aligned[:, 1], aligned[:, 2], c='green', s=5)
        ax3.set_title("再次变换点云")
    
    plt.suptitle(title)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def compare_clouds(cloud1, cloud2):
    """比较两个点云是否相同"""
    if cloud1.shape != cloud2.shape:
        print(f"点云形状不同: {cloud1.shape} vs {cloud2.shape}")
        return False
    
    # 确保格式一致
    if cloud1.shape[0] == 3 and cloud1.shape[1] != 3:
        cloud1 = cloud1.T
    if cloud2.shape[0] == 3 and cloud2.shape[1] != 3:
        cloud2 = cloud2.T
    
    # 计算欧氏距离
    diff = np.linalg.norm(cloud1 - cloud2, axis=1)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"最大点距离: {max_diff}")
    print(f"平均点距离: {mean_diff}")
    
    # 通常小于1e-5可以认为相同
    return max_diff < 1e-5

def test_centroid_alignment():
    """测试质心对齐的一致性"""
    print("="*80)
    print("测试质心对齐变换的一致性")
    print("="*80)
    
    # 生成随机点云
    n_points = 1000
    scale = 50.0  # 较大的尺度，模拟CIF文件
    
    # 随机点云 [N, 3]格式
    np.random.seed(42)  # 设置随机种子以确保结果可重现
    source_cloud = np.random.rand(n_points, 3).astype(np.float32) * scale
    
    print(f"生成随机点云，点数: {n_points}, 尺度: {scale}")
    print(f"点云形状: {source_cloud.shape}")
    print(f"点云范围: [{np.min(source_cloud)}, {np.max(source_cloud)}]")
    
    # 计算源点云质心
    source_centroid = calculate_centroid(source_cloud)
    print(f"源点云质心: {source_centroid}")
    
    # 生成随机变换矩阵
    rotation = random_rotation_matrix()
    translation = random_translation_vector(scale=5.0)
    
    print("随机旋转矩阵:")
    print(rotation)
    print("随机平移向量:")
    print(translation)
    print(f"平移向量范数: {np.linalg.norm(translation)}")
    
    # 应用变换，得到目标点云
    target_cloud = transform_point_cloud(source_cloud, rotation, translation)
    
    # 计算目标点云质心
    target_centroid = calculate_centroid(target_cloud)
    print(f"目标点云质心: {target_centroid}")
    
    # 理论上的目标质心
    theoretical_target_centroid = np.matmul(rotation, source_centroid) + translation
    print(f"理论目标质心: {theoretical_target_centroid}")
    print(f"质心误差: {np.linalg.norm(target_centroid - theoretical_target_centroid)}")
    
    # 生成质心对齐的点云
    aligned_cloud, centroid_diff = generate_centroid_aligned_cloud(source_cloud, target_cloud, rotation)
    
    # 计算质心对齐的变换矩阵
    aligned_rotation, aligned_translation = calculate_centroid_aligned_transformation(
        rotation, translation, centroid_diff)
    
    print("质心偏差向量:")
    print(centroid_diff)
    print(f"质心偏差范数: {np.linalg.norm(centroid_diff)}")
    
    print("质心对齐旋转矩阵:")
    print(aligned_rotation)
    print("质心对齐平移向量:")
    print(aligned_translation)
    
    # 使用质心对齐的变换矩阵再次变换源点云
    realigned_cloud = transform_point_cloud(source_cloud, aligned_rotation, aligned_translation)
    
    # 计算再次变换后的点云质心
    realigned_centroid = calculate_centroid(realigned_cloud)
    print(f"再次变换后点云质心: {realigned_centroid}")
    
    # 比较质心
    print(f"源点云质心: {source_centroid}")
    print(f"目标点云质心: {target_centroid}")
    print(f"对齐点云质心: {calculate_centroid(aligned_cloud)}")
    print(f"再次变换点云质心: {realigned_centroid}")
    
    # 比较点云
    print("\n比较对齐点云和再次变换点云:")
    is_same = compare_clouds(aligned_cloud, realigned_cloud)
    if is_same:
        print("结论: 两个点云实质上相同，质心对齐变换是一致的。")
    else:
        print("结论: 两个点云不同，质心对齐变换可能存在问题。")
    
    # 可视化结果
    visualize_point_clouds(source_cloud, target_cloud, realigned_cloud, "质心对齐测试")

if __name__ == "__main__":
    test_centroid_alignment()
