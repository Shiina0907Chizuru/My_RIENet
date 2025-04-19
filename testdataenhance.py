#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入cif_to_point_cloud.py中的函数
from cif_to_point_cloud import (
    parse_cif, 
    random_rotation_matrix, 
    transform_point_cloud_numpy, 
    calculate_centroid, 
    save_as_cif
)

def load_pkl(pkl_path):
    """加载PKL文件，提取源点云、目标点云、旋转矩阵和平移向量"""
    print(f"加载PKL文件: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 提取源点云、目标点云、旋转矩阵和平移向量
    source_points = data.get('source', None)
    target_points = data.get('target', None)
    rotation_matrix = data.get('rotation', None)
    translation_vector = data.get('translation', None)
    
    # 检查是否提取成功
    if source_points is None or target_points is None or rotation_matrix is None or translation_vector is None:
        keys = list(data.keys())
        # 尝试其他可能的键名
        if 'points_src' in keys and 'points_tgt' in keys:
            source_points = data.get('points_src')
            target_points = data.get('points_tgt')
        if 'rotation_ab' in keys:
            rotation_matrix = data.get('rotation_ab')
        if 'translation_ab' in keys:
            translation_vector = data.get('translation_ab')
    
    # 检查是否是PyTorch张量，如果是则转换为NumPy数组
    import torch
    if isinstance(source_points, torch.Tensor):
        print("将源点云从PyTorch张量转换为NumPy数组")
        source_points = source_points.detach().cpu().numpy()
    if isinstance(target_points, torch.Tensor):
        print("将目标点云从PyTorch张量转换为NumPy数组")
        target_points = target_points.detach().cpu().numpy()
    if isinstance(rotation_matrix, torch.Tensor):
        print("将旋转矩阵从PyTorch张量转换为NumPy数组")
        rotation_matrix = rotation_matrix.detach().cpu().numpy()
    if isinstance(translation_vector, torch.Tensor):
        print("将平移向量从PyTorch张量转换为NumPy数组")
        translation_vector = translation_vector.detach().cpu().numpy()
    
    # 如果点云是批处理形式（例如[1, 3, N]），则去掉批处理维度
    if len(source_points.shape) > 2 and source_points.shape[0] == 1:
        source_points = source_points.squeeze(0)
        print(f"移除源点云批处理维度，新形状: {source_points.shape}")
    
    if len(target_points.shape) > 2 and target_points.shape[0] == 1:
        target_points = target_points.squeeze(0)
        print(f"移除目标点云批处理维度，新形状: {target_points.shape}")
    
    print(f"源点云形状: {source_points.shape}")
    print(f"目标点云形状: {target_points.shape}")
    print(f"旋转矩阵:\n{rotation_matrix}")
    print(f"平移向量: {translation_vector}")
    
    return source_points, target_points, rotation_matrix, translation_vector

def compute_composite_transformation(R1, t1, R2, t2):
    """计算两个变换的组合: (R1,t1) ∘ (R2,t2)
    结果是先应用(R2,t2)，再应用(R1,t1)的变换
    """
    # R_new = R1 * R2
    # t_new = R1 * t2 + t1
    R_new = np.matmul(R1, R2)
    t_new = np.matmul(R1, t2) + t1
    
    return R_new, t_new

def apply_data_augmentation(source_cloud, target_cloud, rotation, translation, output_dir, base_name, cif_content=None, atom_types=None, atom_labels=None, atom_lines=None):
    """应用数据增强：生成随机旋转并更新变换矩阵
    保持源点云不变，只对目标点云应用随机旋转
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保点云是[3,N]格式，便于后续处理
    if source_cloud.shape[1] == 3 and source_cloud.shape[0] != 3:
        source_cloud = source_cloud.T
        print("转换源点云格式为[3,N]")
    
    if target_cloud.shape[1] == 3 and target_cloud.shape[0] != 3:
        target_cloud = target_cloud.T
        print("转换目标点云格式为[3,N]")
    
    # 保存原始点云的CIF文件（如果提供了CIF内容）
    if cif_content is not None:
        # 源点云
        source_cloud_for_cif = source_cloud.T if source_cloud.shape[0] == 3 else source_cloud
        source_cif_path = os.path.join(output_dir, f"{base_name}_source.cif")
        save_as_cif(source_cloud_for_cif, atom_types, atom_labels, cif_content, atom_lines, source_cif_path)
        print(f"保存原始源点云CIF文件: {source_cif_path}")
        
        # 目标点云
        target_cloud_for_cif = target_cloud.T if target_cloud.shape[0] == 3 else target_cloud
        target_cif_path = os.path.join(output_dir, f"{base_name}_target.cif")
        save_as_cif(target_cloud_for_cif, atom_types, atom_labels, cif_content, atom_lines, target_cif_path)
        print(f"保存原始目标点云CIF文件: {target_cif_path}")
    
    # 打印原始变换信息
    print(f"\n原始旋转矩阵:\n{rotation}")
    print(f"原始平移向量: {translation}")
    
    # 计算原始质心
    source_centroid = calculate_centroid(source_cloud)
    target_centroid = calculate_centroid(target_cloud)
    print(f"原始源点云质心: {source_centroid}")
    print(f"原始目标点云质心: {target_centroid}")
    
    # 生成随机旋转矩阵用于数据增强
    augment_rotation = random_rotation_matrix()
    print(f"\n数据增强随机旋转矩阵:\n{augment_rotation}")
    
    # 1. 源点云保持不变
    augmented_source = source_cloud.copy()  # 直接复制，不做任何变换
    print(f"保持源点云不变，形状: {augmented_source.shape}")
    
    # 2. 对目标点云应用随机旋转
    augmented_target = transform_point_cloud_numpy(target_cloud, augment_rotation, np.zeros(3))
    print(f"应用随机旋转到目标点云，形状: {augmented_target.shape}")
    
    # 3. 更新变换矩阵以反映数据增强
    # 原始变换: target = R * source + t
    # 随机旋转目标点云: augmented_target = R_aug * target = R_aug * (R * source + t)
    # 新的变换关系: augmented_target = R_new * source + t_new
    # 因此新的旋转矩阵为R_new = R_aug * R，新的平移向量为t_new = R_aug * t
    augmented_rotation = np.matmul(augment_rotation, rotation)
    augmented_translation = np.matmul(augment_rotation, translation)
    
    print(f"\n增强后的旋转矩阵:\n{augmented_rotation}")
    print(f"增强后的平移向量: {augmented_translation}")
    
    # 计算增强后的质心
    augmented_source_centroid = calculate_centroid(augmented_source)
    augmented_target_centroid = calculate_centroid(augmented_target)
    print(f"增强后源点云质心（不变）: {augmented_source_centroid}")
    print(f"增强后目标点云质心: {augmented_target_centroid}")
    
    # 4. 添加质心对齐步骤
    print("\n=== 进行质心对齐 ===")
    
    # 计算偏移向量（目标点云质心 - 源点云质心）
    centroid_offset = augmented_target_centroid - augmented_source_centroid
    print(f"质心偏移向量: {centroid_offset}")
    
    # 调整平移向量以保证质心对齐
    # 新的平移向量 = 原平移向量 - 质心偏移
    # 因为我们希望变换后 R*S + t_aligned 的质心 = 源点云质心
    centroid_aligned_translation = augmented_translation - centroid_offset
    print(f"质心对齐后的平移向量: {centroid_aligned_translation}")
    
    # 使用质心对齐的平移向量计算新的目标点云
    centroid_aligned_target = transform_point_cloud_numpy(augmented_source, augmented_rotation, centroid_aligned_translation)
    
    # 验证质心对齐是否有效
    aligned_target_centroid = calculate_centroid(centroid_aligned_target)
    print(f"质心对齐后的目标点云质心: {aligned_target_centroid}")
    print(f"源点云质心: {augmented_source_centroid}")
    print(f"质心差异: {np.linalg.norm(aligned_target_centroid - augmented_source_centroid)}")
    
    # 保存增强后的CIF文件（如果提供了CIF内容）
    if cif_content is not None:
        # 增强后的源点云 (与原始源点云相同)
        augmented_source_for_cif = augmented_source.T if augmented_source.shape[0] == 3 else augmented_source
        aug_source_cif_path = os.path.join(output_dir, f"{base_name}_aug_source.cif")
        save_as_cif(augmented_source_for_cif, atom_types, atom_labels, cif_content, atom_lines, aug_source_cif_path)
        print(f"保存增强后的源点云CIF文件 (不变): {aug_source_cif_path}")
        
        # 增强后的目标点云（应用随机旋转，但不进行质心对齐）
        augmented_target_for_cif = augmented_target.T if augmented_target.shape[0] == 3 else augmented_target
        aug_target_cif_path = os.path.join(output_dir, f"{base_name}_aug_target.cif")
        save_as_cif(augmented_target_for_cif, atom_types, atom_labels, cif_content, atom_lines, aug_target_cif_path)
        print(f"保存增强后的目标点云CIF文件 (未质心对齐): {aug_target_cif_path}")
        
        # 增强后的目标点云（应用随机旋转，并进行质心对齐）
        aligned_target_for_cif = centroid_aligned_target.T if centroid_aligned_target.shape[0] == 3 else centroid_aligned_target
        aligned_target_cif_path = os.path.join(output_dir, f"{base_name}_aug_target_aligned.cif")
        save_as_cif(aligned_target_for_cif, atom_types, atom_labels, cif_content, atom_lines, aligned_target_cif_path)
        print(f"保存增强后的目标点云CIF文件 (已质心对齐): {aligned_target_cif_path}")
        
        # 直接变换得到的目标点云（使用原增强旋转矩阵和平移向量）
        direct_target = transform_point_cloud_numpy(source_cloud, augmented_rotation, augmented_translation)
        direct_target_for_cif = direct_target.T if direct_target.shape[0] == 3 else direct_target
        direct_target_cif_path = os.path.join(output_dir, f"{base_name}_direct_target.cif")
        save_as_cif(direct_target_for_cif, atom_types, atom_labels, cif_content, atom_lines, direct_target_cif_path)
        print(f"保存直接变换得到的目标点云CIF文件 (未质心对齐): {direct_target_cif_path}")
        
        # 直接变换得到的目标点云（使用质心对齐的平移向量）
        aligned_direct_target = transform_point_cloud_numpy(source_cloud, augmented_rotation, centroid_aligned_translation)
        aligned_direct_target_for_cif = aligned_direct_target.T if aligned_direct_target.shape[0] == 3 else aligned_direct_target
        aligned_direct_target_cif_path = os.path.join(output_dir, f"{base_name}_direct_target_aligned.cif")
        save_as_cif(aligned_direct_target_for_cif, atom_types, atom_labels, cif_content, atom_lines, aligned_direct_target_cif_path)
        print(f"保存直接变换得到的目标点云CIF文件 (已质心对齐): {aligned_direct_target_cif_path}")
    
    # 保存更新的PKL文件（使用质心对齐的平移向量）
    augmented_data = {
        'source': augmented_source if augmented_source.shape[0] == 3 else augmented_source.T,
        'target': centroid_aligned_target if centroid_aligned_target.shape[0] == 3 else centroid_aligned_target.T,
        'rotation': augmented_rotation,
        'translation': centroid_aligned_translation
    }
    
    augmented_pkl_path = os.path.join(output_dir, f"{base_name}_aug_aligned.pkl")
    with open(augmented_pkl_path, 'wb') as f:
        pickle.dump(augmented_data, f)
    print(f"保存质心对齐后的PKL文件: {augmented_pkl_path}")
    
    # 返回增强后且质心对齐的数据
    return augmented_source, centroid_aligned_target, augmented_rotation, centroid_aligned_translation

def visualize_point_clouds(source, target, aug_source=None, aug_target=None, direct_target=None, output_path=None):
    """可视化点云"""
    n_plots = 2
    if aug_source is not None and aug_target is not None:
        n_plots = 4
    if direct_target is not None:
        n_plots = 5
    
    fig = plt.figure(figsize=(n_plots*5, 5))
    
    # 确保点云是[N, 3]格式用于可视化
    if source.shape[0] == 3:
        source = source.T
    if target.shape[0] == 3:
        target = target.T
    
    # 原始源点云
    ax1 = fig.add_subplot(1, n_plots, 1, projection='3d')
    ax1.scatter(source[:, 0], source[:, 1], source[:, 2], c='blue', s=5)
    ax1.set_title("原始源点云")
    
    # 原始目标点云
    ax2 = fig.add_subplot(1, n_plots, 2, projection='3d')
    ax2.scatter(target[:, 0], target[:, 1], target[:, 2], c='red', s=5)
    ax2.set_title("原始目标点云")
    
    # 增强后的点云
    if aug_source is not None and aug_target is not None:
        if aug_source.shape[0] == 3:
            aug_source = aug_source.T
        if aug_target.shape[0] == 3:
            aug_target = aug_target.T
        
        # 增强后的源点云
        ax3 = fig.add_subplot(1, n_plots, 3, projection='3d')
        ax3.scatter(aug_source[:, 0], aug_source[:, 1], aug_source[:, 2], c='green', s=5)
        ax3.set_title("增强后源点云")
        
        # 增强后的目标点云
        ax4 = fig.add_subplot(1, n_plots, 4, projection='3d')
        ax4.scatter(aug_target[:, 0], aug_target[:, 1], aug_target[:, 2], c='purple', s=5)
        ax4.set_title("增强后目标点云")
    
    # 直接变换得到的目标点云
    if direct_target is not None:
        if direct_target.shape[0] == 3:
            direct_target = direct_target.T
        
        ax5 = fig.add_subplot(1, n_plots, 5, projection='3d')
        ax5.scatter(direct_target[:, 0], direct_target[:, 1], direct_target[:, 2], c='orange', s=5)
        ax5.set_title("直接变换目标点云")
    
    plt.suptitle("点云数据增强比较")
    plt.tight_layout()
    
    # 保存图像
    if output_path is not None:
        plt.savefig(output_path)
        print(f"已保存可视化结果到: {output_path}")
    
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='测试点云数据增强')
    parser.add_argument('--pkl_path', type=str, required=True, help='PKL文件路径，包含源点云、目标点云、旋转矩阵和平移向量')
    parser.add_argument('--cif_path', type=str, help='可选：CIF文件路径，用于生成可视化用的CIF文件')
    parser.add_argument('--output_dir', type=str, default='./data_augmentation_test', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载PKL数据
    source_cloud, target_cloud, rotation, translation = load_pkl(args.pkl_path)
    
    # 如果提供了CIF文件，解析它
    cif_content = None
    atom_types = None
    atom_labels = None
    atom_lines = None
    if args.cif_path and os.path.exists(args.cif_path):
        point_cloud, atom_types, atom_labels, cif_content, atom_lines = parse_cif(args.cif_path)
        print(f"成功解析CIF文件: {args.cif_path}")
    
    # 生成基本文件名
    base_name = os.path.splitext(os.path.basename(args.pkl_path))[0]
    
    # 应用数据增强
    aug_source, aug_target, aug_rotation, aug_translation = apply_data_augmentation(
        source_cloud, target_cloud, rotation, translation, args.output_dir, base_name,
        cif_content, atom_types, atom_labels, atom_lines
    )
    
    # 使用增强后的变换矩阵直接变换源点云，用于验证
    if source_cloud.shape[1] == 3 and source_cloud.shape[0] != 3:
        source_for_transform = source_cloud.T
    else:
        source_for_transform = source_cloud
    
    direct_target = transform_point_cloud_numpy(source_for_transform, aug_rotation, aug_translation)
    
    # # 可视化结果
    # vis_output_path = os.path.join(args.output_dir, f"{base_name}_visualization.png")
    # visualize_point_clouds(source_cloud, target_cloud, aug_source, aug_target, direct_target, vis_output_path)
    
    print("\n数据增强测试完成！")

if __name__ == "__main__":
    main()

# 原始文件：
# 6coz_point_train_source.cif - 原始源点云
# 6coz_point_train_target.cif - 原始目标点云
# 增强后文件：
# 6coz_point_train_aug_source.cif - 增强后源点云（与原始相同）
# 6coz_point_train_aug_target.cif - 增强后目标点云（已应用随机旋转）
# 6coz_point_train_direct_target.cif - 使用新变换矩阵直接变换得到的目标点云
# 更新的PKL文件：
# 6coz_point_train_aug.pkl - 包含增强后的点云数据和更新后的变换矩阵