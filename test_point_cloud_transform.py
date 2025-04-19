#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def print_separator(title):
    """打印分隔符，使输出更易读"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def random_rotation_matrix():
    """生成随机旋转矩阵"""
    print_separator("生成随机旋转矩阵")
    
    # 使用QR分解生成随机旋转矩阵
    A = np.random.randn(3, 3)
    print("随机矩阵 A:")
    print(A)
    
    Q, R = np.linalg.qr(A)
    print("\nQR分解后的Q矩阵:")
    print(Q)
    
    # 确保是特殊正交矩阵(行列式为1)
    det_Q = np.linalg.det(Q)
    print(f"\nQ的行列式: {det_Q}")
    
    if np.linalg.det(Q) < 0:
        print("行列式为负，调整Q矩阵")
        Q[:, 0] = -Q[:, 0]
        print("调整后的Q矩阵:")
        print(Q)
        print(f"调整后Q的行列式: {np.linalg.det(Q)}")
    
    rotation = Q.astype(np.float32)
    print("\n最终旋转矩阵:")
    print(rotation)
    
    # 验证是否为正交矩阵
    identity_test = np.matmul(rotation, rotation.T)
    print("\n检验R*R^T是否为单位矩阵:")
    print(identity_test)
    print(f"误差: {np.max(np.abs(identity_test - np.eye(3)))}")
    
    return rotation

def random_translation_vector(scale=0.1):
    """生成随机平移向量"""
    print_separator("生成随机平移向量")
    
    translation = np.random.uniform(-scale, scale, size=(3,)).astype(np.float32)
    print(f"随机平移向量 (scale={scale}):")
    print(translation)
    print(f"平移向量范数: {np.linalg.norm(translation)}")
    
    return translation

def calculate_centroid(point_cloud):
    """计算点云的质心"""
    print_separator("计算点云质心")
    
    print(f"点云形状: {point_cloud.shape}")
    
    # 确保点云格式正确
    if point_cloud.shape[0] == 3 and point_cloud.shape[1] != 3:
        print("点云格式为[3, N]")
        # 如果点云是[3, N]格式
        centroid = np.mean(point_cloud, axis=1)
    else:
        print("点云格式为[N, 3]")
        # 如果点云是[N, 3]格式
        centroid = np.mean(point_cloud, axis=0)
    
    print("质心坐标:")
    print(centroid)
    
    return centroid

def transform_point_cloud(point_cloud, rotation, translation, rotation_only=False):
    """应用旋转和平移变换到点云"""
    print_separator("应用点云变换")
    
    # 检查点云范围
    cloud_min = np.min(point_cloud)
    cloud_max = np.max(point_cloud)
    cloud_range = cloud_max - cloud_min
    print(f"原始点云范围: 最小值={cloud_min}, 最大值={cloud_max}, 范围={cloud_range}")
    
    # 确保点云是[3, N]格式
    original_shape = point_cloud.shape
    if point_cloud.shape[1] == 3 and point_cloud.shape[0] != 3:
        print(f"转置点云: 从{point_cloud.shape}到", end="")
        point_cloud = point_cloud.T
        print(f"{point_cloud.shape}")
    
    # 获取点云数据范围
    print(f"点云数据范围: {np.min(point_cloud)} 到 {np.max(point_cloud)}")
    
    if rotation_only:
        print("仅应用旋转变换")
        transformed_point_cloud = np.matmul(rotation, point_cloud)
    else:
        print("应用旋转和平移变换")
        transformed_point_cloud = np.matmul(rotation, point_cloud) + translation.reshape(3, 1)
    
    print(f"变换后点云形状: {transformed_point_cloud.shape}")
    
    # 检查变换后的点云范围
    transformed_min = np.min(transformed_point_cloud)
    transformed_max = np.max(transformed_point_cloud)
    transformed_range = transformed_max - transformed_min
    print(f"变换后点云范围: 最小值={transformed_min}, 最大值={transformed_max}, 范围={transformed_range}")
    
    # 如果需要，恢复原始形状
    if original_shape[1] == 3 and original_shape[0] != 3:
        transformed_point_cloud = transformed_point_cloud.T
        print(f"恢复形状: {transformed_point_cloud.shape}")
    
    return transformed_point_cloud

def generate_centroid_aligned_cloud(source_cloud, target_cloud, rotation):
    """生成质心对齐的目标点云"""
    print_separator("生成质心对齐的点云")
    
    # 获取输入点云的格式
    source_format_is_3N = source_cloud.shape[0] == 3 and source_cloud.shape[1] != 3
    target_format_is_3N = target_cloud.shape[0] == 3 and target_cloud.shape[1] != 3
    
    print(f"源点云形状: {source_cloud.shape}, 格式为[3, N]: {source_format_is_3N}")
    print(f"目标点云形状: {target_cloud.shape}, 格式为[3, N]: {target_format_is_3N}")
    
    # 确保源点云和目标点云格式一致，便于计算质心和对齐
    if source_format_is_3N != target_format_is_3N:
        print("警告：源点云和目标点云格式不一致，将进行调整")
        if not source_format_is_3N:  # 如果源点云不是[3, N]格式，转换它
            source_cloud = source_cloud.T
            print(f"调整后源点云形状: {source_cloud.shape}")
        if not target_format_is_3N:  # 如果目标点云不是[3, N]格式，转换它
            target_cloud = target_cloud.T
            print(f"调整后目标点云形状: {target_cloud.shape}")
    
    # 计算质心
    if source_format_is_3N:
        source_centroid = np.mean(source_cloud, axis=1)
        target_centroid = np.mean(target_cloud, axis=1)
    else:
        source_centroid = np.mean(source_cloud, axis=0)
        target_centroid = np.mean(target_cloud, axis=0)
    
    print("源点云质心:")
    print(source_centroid)
    print("目标点云质心:")
    print(target_centroid)
    
    # 计算质心偏差
    centroid_diff = source_centroid - target_centroid
    print("质心偏差向量:")
    print(centroid_diff)
    print(f"质心偏差向量范数: {np.linalg.norm(centroid_diff)}")
    
    # 创建质心对齐的点云
    if target_format_is_3N:
        # 如果目标点云是[3, N]格式
        aligned_cloud = target_cloud.copy()
        # 添加平移来对齐质心，对于[3, N]格式，需要广播平移向量
        aligned_cloud = aligned_cloud + centroid_diff.reshape(3, 1)
    else:
        # 如果目标点云是[N, 3]格式
        aligned_cloud = target_cloud.copy()
        # 添加平移来对齐质心
        aligned_cloud = aligned_cloud + centroid_diff
    
    # 检查质心对齐是否成功
    if target_format_is_3N:
        aligned_centroid = np.mean(aligned_cloud, axis=1)
    else:
        aligned_centroid = np.mean(aligned_cloud, axis=0)
    
    print("质心对齐后的点云质心:")
    print(aligned_centroid)
    
    # 计算源点云质心和对齐后点云质心的差异
    alignment_error = np.linalg.norm(source_centroid - aligned_centroid)
    print(f"质心对齐误差: {alignment_error}")
    
    return aligned_cloud, centroid_diff

def calculate_centroid_aligned_transformation(original_rotation, original_translation, centroid_diff):
    """计算质心对齐的变换矩阵"""
    print_separator("计算质心对齐的变换矩阵")
    
    print("原始旋转矩阵:")
    print(original_rotation)
    print("原始平移向量:")
    print(original_translation)
    print("质心偏差向量:")
    print(centroid_diff)
    
    # 质心对齐的变换保持旋转不变，只调整平移
    centroid_aligned_rotation = original_rotation.copy()
    
    # 计算新的平移向量
    # 新的平移 = 原始平移 + 质心偏差
    centroid_aligned_translation = original_translation + centroid_diff
    
    print("质心对齐旋转矩阵:")
    print(centroid_aligned_rotation)
    print("质心对齐平移向量:")
    print(centroid_aligned_translation)
    
    return centroid_aligned_rotation, centroid_aligned_translation

def visualize_point_clouds(source, target, aligned=None, title="点云可视化"):
    """可视化源点云、目标点云和对齐后的点云"""
    print_separator("点云可视化")
    
    if aligned is None:
        fig = plt.figure(figsize=(12, 6))
        n_plots = 2
    else:
        fig = plt.figure(figsize=(18, 6))
        n_plots = 3
    
    # 确保点云是[N, 3]格式
    if source.shape[0] == 3 and source.shape[1] != 3:
        source = source.T
    if target.shape[0] == 3 and target.shape[1] != 3:
        target = target.T
    if aligned is not None and aligned.shape[0] == 3 and aligned.shape[1] != 3:
        aligned = aligned.T
    
    # 绘制源点云
    ax1 = fig.add_subplot(1, n_plots, 1, projection='3d')
    ax1.scatter(source[:, 0], source[:, 1], source[:, 2], c='blue', s=1, label='Source')
    ax1.set_title("源点云")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 绘制目标点云
    ax2 = fig.add_subplot(1, n_plots, 2, projection='3d')
    ax2.scatter(target[:, 0], target[:, 1], target[:, 2], c='red', s=1, label='Target')
    ax2.set_title("目标点云")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 如果有对齐后的点云，也绘制它
    if aligned is not None:
        ax3 = fig.add_subplot(1, n_plots, 3, projection='3d')
        ax3.scatter(aligned[:, 0], aligned[:, 1], aligned[:, 2], c='green', s=1, label='Aligned')
        ax3.set_title("质心对齐后的点云")
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
    
    plt.suptitle(title)
    
    # 保存图像
    output_dir = "point_cloud_test_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"))
    print(f"图像已保存到 {output_dir}/{title.replace(' ', '_')}.png")
    plt.close()

def test_with_generated_data(n_points=1000, scale=1.0, translation_scale=0.1, format_3N=False):
    """使用生成的数据测试点云变换和质心对齐流程"""
    print_separator(f"用{n_points}个点测试点云变换和质心对齐 (尺度 = {scale})")
    
    # 生成随机点云
    np.random.seed(42)  # 设置随机种子以确保结果可重现
    
    # 原始点云
    if format_3N:
        source_cloud = np.random.rand(3, n_points).astype(np.float32) * scale
        print(f"生成[3, N]格式的源点云，形状: {source_cloud.shape}")
    else:
        source_cloud = np.random.rand(n_points, 3).astype(np.float32) * scale
        print(f"生成[N, 3]格式的源点云，形状: {source_cloud.shape}")
    
    # 获取点云范围
    cloud_min = np.min(source_cloud)
    cloud_max = np.max(source_cloud)
    print(f"源点云范围: [{cloud_min}, {cloud_max}]")
    
    # 生成旋转矩阵和平移向量
    rotation_matrix = random_rotation_matrix()
    translation_vector = random_translation_vector(scale=translation_scale)
    
    # 应用变换
    target_cloud = transform_point_cloud(source_cloud, rotation_matrix, translation_vector)
    
    # 计算源点云和目标点云的质心
    source_centroid = calculate_centroid(source_cloud)
    target_centroid = calculate_centroid(target_cloud)
    
    # 测试不同尺度因子的质心对齐效果
    scale_factors = [1.0, 10.0, 0.1]
    
    for sf in scale_factors:
        print_separator(f"测试尺度因子 {sf}")
        # 生成质心对齐的点云
        aligned_cloud, centroid_diff = generate_centroid_aligned_cloud(source_cloud, target_cloud, rotation_matrix)
        
        # 计算质心对齐的变换矩阵
        aligned_rotation, aligned_translation = calculate_centroid_aligned_transformation(
            rotation_matrix, translation_vector, centroid_diff)
        
        # 使用质心对齐的变换矩阵重新变换源点云
        realigned_cloud = transform_point_cloud(source_cloud, aligned_rotation, aligned_translation)
        
        # 计算质心
        realigned_centroid = calculate_centroid(realigned_cloud)
        
        print(f"\n使用尺度因子 {sf} 的质心对齐结果:")
        print(f"源点云质心: {source_centroid}")
        print(f"目标点云质心: {target_centroid}")
        print(f"对齐后点云质心: {realigned_centroid}")
        
        # 可视化结果
        visualize_point_clouds(source_cloud, target_cloud, aligned_cloud, 
                               title=f"点云比较 (尺度 = {scale}, 尺度因子 = {sf})")
    
    print("\n测试完成！")

def test_specific_scales():
    """测试不同尺度下的点云变换和质心对齐"""
    scales = [1.0, 10.0, 100.0]
    
    for scale in scales:
        test_with_generated_data(n_points=1000, scale=scale, translation_scale=0.1*scale)
    
    # 测试一个非常大的尺度，类似于埃米单位的CIF文件
    test_with_generated_data(n_points=1000, scale=100.0, translation_scale=0.1)

def main():
    print_separator("开始点云变换和质心对齐测试")
    
    # 测试不同尺度
    test_specific_scales()
    
    # 额外测试[3, N]格式
    test_with_generated_data(n_points=1000, scale=10.0, format_3N=True)
    
    print("\n所有测试都已完成。结果已保存在 'point_cloud_test_results' 目录中。")

if __name__ == "__main__":
    main()
