#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import glob
import concurrent.futures
import multiprocessing
from functools import partial
import random
import time
import shutil  # 添加shutil模块导入，用于文件复制操作

def parse_cif(cif_path, verbose=True):
    """解析CIF文件，提取原子坐标和完整CIF内容
    
    参数:
        cif_path: CIF文件路径
        verbose: 是否打印详细信息，默认为True
        
    返回:
        point_cloud: 形状为[N, 3]的点云数据
        atom_types: 原子类型列表
        atom_labels: 原子标签列表
        cif_content: CIF文件完整内容
        atom_lines: 原子行号列表
    """
    if verbose:
        print(f"正在解析CIF文件: {cif_path}")
    
    atoms = []
    atom_lines = []
    atom_types = []
    atom_labels = []
    cif_content = []
    
    try:
        with open(cif_path, 'r') as f:
            lines = f.readlines()
        
        # 保存CIF文件的完整内容
        cif_content = lines.copy()
        
        # 解析PDB格式的CIF文件
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 检测原子行（以ATOM开头）
            if line.startswith('ATOM'):
                parts = line.split()
                if len(parts) >= 13:  # 确保有足够的列
                    try:
                        atom_number = parts[1]
                        atom_type = parts[3]  # 原子类型（如 CA）
                        atom_label = parts[5]  # 残基名称（如 ALA）
                        
                        # 获取坐标（第10、11、12列，空格分隔）
                        x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                        
                        atoms.append([x, y, z])
                        atom_types.append(atom_type)
                        atom_labels.append(atom_label)
                        atom_lines.append(i)
                    except (ValueError, IndexError) as e:
                        if verbose:
                            print(f"警告：无法解析第 {i+1} 行: {e}")
                        continue
        
        # 检查是否成功提取到原子坐标
        if not atoms:
            raise ValueError("未能从CIF文件中提取到原子坐标")
        
        if verbose:
            print(f"成功解析 {len(atoms)} 个原子")
        
        # 转换为numpy数组
        return np.array(atoms, dtype=np.float32), atom_types, atom_labels, cif_content, atom_lines
    
    except Exception as e:
        print(f"解析CIF文件时出错: {str(e)}")
        raise

def normalize_point_cloud(points, verbose=True):
    """
    将点云归一化到[-1, 1]范围，并确保质心在原点
    
    参数:
        points: 形状为[N, 3]或[3, N]的点云数据
        verbose: 是否打印详细信息，默认为True
    
    返回:
        normalized_points: 归一化后的点云
        scale_factor: 归一化使用的缩放因子
        centroid: 原始点云的质心
    """
    # 确保点云形状为[N, 3]
    transpose_needed = False
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T
        transpose_needed = True
    
    # 计算质心并平移到原点
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # 计算在各个维度上的最大绝对值，用于缩放
    max_abs = np.max(np.abs(centered_points), axis=0)  # 对每个维度单独计算
    overall_max = np.max(max_abs)  # 使用整体最大值确保比例一致
    scale_factor = overall_max
    
    # 归一化到[-1, 1]范围
    normalized_points = centered_points / scale_factor
    
    # 打印详细信息，帮助调试
    if verbose:
        print(f"原始点云维度: {points.shape}")
        print(f"各维度最大绝对值: {max_abs}")
        print(f"使用的缩放因子: {scale_factor}")
        print(f"归一化后点云范围: X[{np.min(normalized_points[:,0]):.4f}, {np.max(normalized_points[:,0]):.4f}], "
              f"Y[{np.min(normalized_points[:,1]):.4f}, {np.max(normalized_points[:,1]):.4f}], "
              f"Z[{np.min(normalized_points[:,2]):.4f}, {np.max(normalized_points[:,2]):.4f}]")
    
    # 如果需要，转置回原始形状
    if transpose_needed:
        normalized_points = normalized_points.T
    
    return normalized_points, scale_factor, centroid

def normalize_point_clouds_with_shared_params(source_points, target_points, verbose=True):
    """
    使用共享参数对源点云和目标点云进行归一化
    
    参数:
        source_points: 源点云，形状为[N, 3]或[3, N]
        target_points: 目标点云，形状为[M, 3]或[3, M]
        verbose: 是否打印详细信息，默认为True
    
    返回:
        normalized_source: 归一化后的源点云
        normalized_target: 归一化后的目标点云
        scale_factor: 使用的缩放因子（两个点云中较大的）
        centroid: 使用的质心（目标点云的质心）
    """
    # 确保两个点云都是[N, 3]格式
    source_transpose_needed = False
    target_transpose_needed = False
    
    if source_points.shape[0] == 3 and source_points.shape[1] != 3:
        source_points = source_points.T
        source_transpose_needed = True
    
    if target_points.shape[0] == 3 and target_points.shape[1] != 3:
        target_points = target_points.T
        target_transpose_needed = True
    
    # 首先分别计算源点云和目标点云的归一化参数
    _, source_scale, source_centroid = normalize_point_cloud(source_points, verbose=False)
    _, target_scale, target_centroid = normalize_point_cloud(target_points, verbose=False)
    
    # 按照要求，使用目标点云的质心和两者中较大的缩放因子
    centroid = target_centroid
    scale_factor = max(source_scale, target_scale)
    
    if verbose:
        print(f"源点云维度: {source_points.shape}, 目标点云维度: {target_points.shape}")
        print(f"源点云质心: {source_centroid}, 目标点云质心: {target_centroid}")
        print(f"源点云缩放因子: {source_scale}, 目标点云缩放因子: {target_scale}")
        print(f"使用的质心(目标点云): {centroid}")
        print(f"使用的缩放因子(较大值): {scale_factor}")
    
    # 对两个点云使用相同的参数进行归一化
    source_centered = source_points - centroid
    target_centered = target_points - centroid
    
    normalized_source = source_centered / scale_factor
    normalized_target = target_centered / scale_factor
    
    # 显示归一化后的范围
    if verbose:
        print(f"归一化后源点云范围: X[{np.min(normalized_source[:,0]):.4f}, {np.max(normalized_source[:,0]):.4f}], "
              f"Y[{np.min(normalized_source[:,1]):.4f}, {np.max(normalized_source[:,1]):.4f}], "
              f"Z[{np.min(normalized_source[:,2]):.4f}, {np.max(normalized_source[:,2]):.4f}]")
        print(f"归一化后目标点云范围: X[{np.min(normalized_target[:,0]):.4f}, {np.max(normalized_target[:,0]):.4f}], "
              f"Y[{np.min(normalized_target[:,1]):.4f}, {np.max(normalized_target[:,1]):.4f}], "
              f"Z[{np.min(normalized_target[:,2]):.4f}, {np.max(normalized_target[:,2]):.4f}]")
    
    # 如果需要，转置回原始形状
    if source_transpose_needed:
        normalized_source = normalized_source.T
    if target_transpose_needed:
        normalized_target = normalized_target.T
    
    return normalized_source, normalized_target, scale_factor, centroid

def create_grid_info(normalize=False, grid_shape=np.array([64, 64, 64], dtype=np.int32), voxel_size=None):
    """
    创建体素网格信息
    
    参数:
        normalize: 是否归一化，默认为False
        grid_shape: 网格形状，默认为[64, 64, 64]
        voxel_size: 体素大小，默认为None（会根据normalize自动确定）
        
    返回:
        grid_info: 网格信息字典
    """
    # 确保grid_shape是numpy数组
    if not isinstance(grid_shape, np.ndarray):
        grid_shape = np.array(grid_shape, dtype=np.int32)
    
    # 将标量值改为一维数组，避免索引越界问题
    x_origin = np.array([0.0], dtype=np.float32)
    y_origin = np.array([0.0], dtype=np.float32)
    z_origin = np.array([0.0], dtype=np.float32)
    
    # 计算体素大小，根据是否归一化调整
    if voxel_size is None:
        if normalize:
            voxel_size = 2.0 / grid_shape  # 归一化点云范围[-1, 1]，总长度为2
        else:
            voxel_size = np.array([0.05, 0.05, 0.05], dtype=np.float32)  # 默认体素大小
    elif isinstance(voxel_size, (int, float)):
        voxel_size = np.array([voxel_size, voxel_size, voxel_size], dtype=np.float32)
    
    # 生成网格信息字典
    grid_info = {
        'grid_shape': grid_shape,
        'x_origin': x_origin,
        'y_origin': y_origin,
        'z_origin': z_origin,
        'x_voxel': voxel_size[0],
        'y_voxel': voxel_size[1],
        'z_voxel': voxel_size[2],
        'normalized': normalize
    }
    
    return grid_info

def random_rotation_matrix():
    """生成随机旋转矩阵"""
    # 使用QR分解生成随机旋转矩阵
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    
    # 确保是特殊正交矩阵(行列式为1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    
    return Q.astype(np.float32)

def random_translation_vector(scale=0.1):
    """生成随机平移向量"""
    return np.random.uniform(-scale, scale, size=(3,)).astype(np.float32)

def transform_point_cloud(point_cloud, rotation, translation=None, rotation_only=False):
    """使用旋转矩阵和平移向量变换点云
    
    参数:
        point_cloud: 输入点云，形状为[N, 3]或[3, N]
        rotation: 旋转矩阵，形状为[3, 3]
        translation: 平移向量，形状为[3]，默认为None
        rotation_only: 是否仅执行旋转变换，默认为False
        
    返回:
        transformed_point_cloud: 变换后的点云，与输入点云形状相同
    """
    # 确保点云是[3, N]格式
    transpose_needed = False
    if point_cloud.shape[1] == 3 and point_cloud.shape[0] != 3:
        point_cloud = point_cloud.T
        transpose_needed = True
    
    # 应用旋转和平移
    if rotation_only or translation is None:
        # 只应用旋转，不应用平移
        transformed_point_cloud = np.matmul(rotation, point_cloud)
    else:
        # 应用旋转和平移
        transformed_point_cloud = np.matmul(rotation, point_cloud) + translation.reshape(3, 1)
    
    # 如果需要，转置回原始形状
    if transpose_needed:
        transformed_point_cloud = transformed_point_cloud.T
    
    return transformed_point_cloud

def calculate_centroid(point_cloud):
    """计算点云的质心"""
    # 检查点云的形状
    if point_cloud.shape[0] == 3 and point_cloud.shape[1] != 3:
        # 点云是[3, N]格式
        centroid = np.mean(point_cloud, axis=1)
    else:
        # 点云是[N, 3]格式
        centroid = np.mean(point_cloud, axis=0)
    
    return centroid

def calculate_centroid_aligned_transformation(original_rotation, original_translation, centroid_diff):
    """
    根据原始变换和质心偏差计算质心对齐的变换矩阵
    
    参数:
        original_rotation: 原始旋转矩阵 [3, 3]
        original_translation: 原始平移向量 [3]
        centroid_diff: 质心偏差向量 [3]
        
    返回:
        新的旋转矩阵和平移向量，使得变换后的点云质心与源点云质心对齐
    """
    # 旋转矩阵保持不变
    aligned_rotation = original_rotation.copy()
    
    # 计算新的平移向量，使得变换后的点云质心与源点云质心对齐
    # t_new = t_old - centroid_diff
    aligned_translation = original_translation - centroid_diff
    
    return aligned_rotation, aligned_translation

def create_pkl_file(source_points, target_points, output_path, grid_info, rotation=None, translation=None, normalize=False, verbose=True):
    """
    创建 pkl 文件，保存为 PointCloud_ca_Dataset 格式
    
    参数:
        source_points: 源点云，形状为[N, 3]或[3, N]
        target_points: 目标点云，形状为[N, 3]或[3, N]
        output_path: 输出 PKL 文件路径
        grid_info: 网格信息字典
        rotation: 旋转矩阵，默认为 None（使用单位矩阵）
        translation: 平移向量，默认为 None（使用零向量）
        normalize: 是否对点云进行归一化，默认为 False
        verbose: 是否打印详细信息，默认为 True
    """
    # 检查点云形状并打印信息
    if verbose:
        print(f"保存前源点云形状: {source_points.shape}")
        if target_points is not None:
            print(f"保存前目标点云形状: {target_points.shape}")
    
    # 在特定情况下对源点云和目标点云进行归一化
    # 检查grid_info中是否已经有归一化信息
    already_normalized = grid_info.get('normalized', False)
    
    if normalize and not already_normalized:
        if verbose:
            print("在create_pkl_file中进行归一化...")
        source_points, target_points, scale_factor, centroid = normalize_point_clouds_with_shared_params(source_points, target_points, verbose=verbose)
        # 当归一化时，将这些信息添加到grid_info
        grid_info['normalized'] = True
        grid_info['scale_factor'] = float(scale_factor)
        grid_info['original_centroid'] = centroid.tolist()
    elif already_normalized and verbose:
        print(f"grid_info中已存在归一化信息: scale_factor={grid_info.get('scale_factor', 'N/A')}, 跳过重复归一化")
    
    # 转换为 torch 张量
    source_points_tensor = torch.from_numpy(source_points).float()
    target_points_tensor = torch.from_numpy(target_points).float() if target_points is not None else None
    
    # 如果没有提供旋转矩阵和平移向量，使用默认值
    if rotation is None:
        rotation_tensor = torch.eye(3, dtype=torch.float32)
    else:
        # 注意：在pkldataprocess.py中，rotation矩阵的语义是正确的
        # 它表示从源点云到目标点云的变换，与cif_to_point_cloud.py中的语义一致
        # 与test_rotation_translation.py不同，这里不需要进行转置操作
        rotation_tensor = torch.from_numpy(rotation).float()
    
    if translation is None:
        translation_tensor = torch.zeros(3, dtype=torch.float32)
    else:
        translation_tensor = torch.from_numpy(translation).float()
    
    if verbose:
        print(f"保存为 torch 张量后源点云形状: {source_points_tensor.shape}")
        if target_points is not None:
            print(f"保存为 torch 张量后目标点云形状: {target_points_tensor.shape}")
    
    # 从 grid_info 字典中提取所需的字段
    grid_shape = grid_info['grid_shape']
    x_origin = grid_info['x_origin']
    y_origin = grid_info['y_origin']
    z_origin = grid_info['z_origin']
    x_voxel = grid_info['x_voxel']
    nstart = np.array([32], dtype=np.int32)  # 起始索引，默认为 32
    
    # 保存为字典格式
    data_dict = {
        'source': source_points_tensor,  # 源点云
        'rotation': rotation_tensor,   # 旋转矩阵
        'translation': translation_tensor,  # 平移向量
        'grid_info': grid_info  # 将整个 grid_info 作为一个字段，而不是展开
    }
    
    # 只有当目标点云不为空时才添加
    if target_points is not None:
        data_dict['target'] = target_points_tensor
    
    # 如果点云已经归一化，也保存归一化信息
    if 'scale_factor' in grid_info:
        data_dict['scale_factor'] = grid_info['scale_factor']
    if 'original_centroid' in grid_info:
        data_dict['original_centroid'] = grid_info['original_centroid']
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    if verbose:
        print(f"已保存到: {output_path}")
        print(f"保存的源点云形状: {source_points_tensor.shape}")
        if target_points is not None:
            print(f"保存的目标点云形状: {target_points_tensor.shape}")
        print(f"保存的旋转矩阵: {'单位矩阵' if rotation is None else '随机旋转矩阵'}")
        if rotation is not None:
            print("该旋转矩阵表示从源点云到目标点云的变换，与cif_to_point_cloud.py中的语义一致")
        print(f"保存的平移向量: {'零向量' if translation is None else '随机平移向量'}")
        if translation is not None:
            print("该平移向量表示从源点云到目标点云的变换，与cif_to_point_cloud.py中的语义一致")
        print(f"保存的数据字典结构与字段: {', '.join(data_dict.keys())}")
        print("注意：pkl文件中存储的旋转矩阵和平移向量与cif_to_point_cloud.py中的语义一致，即表示从源点云到目标点云的变换")
    
    return data_dict

def save_as_cif(coordinates, atom_types, atom_labels, original_cif, atom_lines, output_path):
    """将点云坐标保存为 CIF 格式"""
    new_cif = original_cif.copy()
    
    # 更新原子坐标
    for i, (coord, line_idx) in enumerate(zip(coordinates, atom_lines)):
        x, y, z = coord
        line = original_cif[line_idx].strip().split()
        
        # 保持原始行的格式，只更新坐标部分
        line[10] = f"{x:.6f}"
        line[11] = f"{y:.6f}"
        line[12] = f"{z:.6f}"
        
        # 重新构建行
        new_line = " ".join(line)
        new_cif[line_idx] = new_line + "\n"
    
    # 写入到新文件
    with open(output_path, 'w') as f:
        f.writelines(new_cif)
    
    print(f"已保存CIF文件: {output_path}")

def visualize_point_clouds(source, target, title="点云可视化"):
    """可视化源点云和目标点云"""
    # 确保点云是[N, 3]格式
    if source.shape[0] == 3 and source.shape[1] != 3:
        source = source.T
    if target.shape[0] == 3 and target.shape[1] != 3:
        target = target.T
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制源点云（蓝色）
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], c='blue', label='源点云', s=5, alpha=0.7)
    
    # 绘制目标点云（红色）
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='red', label='目标点云', s=5, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def process_pair(source_cif, target_cif, output_dir, visualize=False, normalize=False, verbose=True, random_transform=False, rotation_only=False):
    """
    处理一对CIF文件，创建点云数据
    
    参数:
        source_cif: 源CIF文件路径
        target_cif: 目标CIF文件路径或None（如果使用随机变换）
        output_dir: 输出目录
        visualize: 是否可视化点云，默认为False
        normalize: 是否归一化点云，默认为False
        verbose: 是否打印详细信息，默认为True
        random_transform: 是否应用随机旋转平移变换，默认为False
        rotation_only: 是否只应用旋转变换不应用平移，默认为False
        
    返回:
        output_pkl: 生成的PKL文件路径
        output_source_cif: 生成的源CIF文件路径
        output_target_cif: 生成的目标CIF文件路径
    """
    if verbose:
        print(f"处理源文件: {source_cif}")
        print(f"处理目标文件: {target_cif}")
    
    # 创建输出目录结构
    # 1. 为PKL文件创建主目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 为CIF文件创建 cifdata 目录
    cif_data_dir = os.path.join(output_dir, "cifdata")
    os.makedirs(cif_data_dir, exist_ok=True)
    
    if verbose:
        print(f"创建输出目录结构: {output_dir} (为PKL文件) 和 {cif_data_dir} (为CIF文件)")
    
    # 获取源文件名和蛋白质ID
    source_base_name = os.path.splitext(os.path.basename(source_cif))[0]
    
    # 从源文件名提取蛋白质ID（如从"8h1p_point.cif"提取"8h1p"）
    protein_id = source_base_name.split("_")[0] if "_" in source_base_name else source_base_name
    
    # 创建输出文件路径
    # 处理目标文件为None的情况
    if target_cif is None:
        # 如果没有目标文件（随机变换模式）
        output_base_name = f"{source_base_name}_transformed"
        suffix = "_rot" if rotation_only else ""
        output_pkl = os.path.join(output_dir, f"{output_base_name}{suffix}.pkl")
        output_source_cif = os.path.join(output_dir, f"{output_base_name}_source{suffix}.cif")
        output_target_cif = os.path.join(output_dir, f"{output_base_name}_target{suffix}.cif")
        output_aligned_pkl = os.path.join(output_dir, f"{output_base_name}_aligned{suffix}.pkl")
    else:
        # 正常模式，有目标文件
        target_base_name = os.path.splitext(os.path.basename(target_cif))[0]
        output_base_name = f"{source_base_name}_{target_base_name}"
        
        # 创建输出文件路径
        output_pkl = os.path.join(output_dir, f"{output_base_name}.pkl")
        output_source_cif = os.path.join(output_dir, f"{output_base_name}_source.cif")
        output_target_cif = os.path.join(output_dir, f"{output_base_name}_target.cif")
    
    # 检查文件是否已存在（防止重复处理）
    if os.path.exists(output_pkl) and os.path.exists(output_source_cif) and os.path.exists(output_target_cif):
        if verbose:
            print(f"文件 {output_pkl} 已存在，跳过处理")
        return output_pkl, output_source_cif, output_target_cif
    
    # 解析CIF文件
    try:
        # 解析源CIF文件
        source_points, source_atom_types, source_atom_labels, source_cif_content, source_atom_lines = parse_cif(source_cif, verbose=verbose)
        
        # 解析源点云
        # 确保源点云为[3, N]格式用于变换
        point_cloud_for_transform = source_points.T if source_points.shape[1] == 3 else source_points
        
        # 处理目标点云 - 必须有目标CIF文件
        if target_cif is not None:
            # 使用提供的目标CIF文件
            target_points, target_atom_types, target_atom_labels, target_cif_content, target_atom_lines = parse_cif(target_cif, verbose=verbose)
            
            # 记录点云大小信息
            if source_points.shape[0] != target_points.shape[0] and verbose:
                print(f"注意：源点云有 {source_points.shape[0]} 个点，目标点云有 {target_points.shape[0]} 个点")
            
            # 确保目标点云为[3, N]格式用于处理
            target_point_cloud = target_points.T if target_points.shape[1] == 3 else target_points
        else:
            # 必须提供目标CIF文件
            raise ValueError("必须提供目标CIF文件，源点云和目标点云必须从各自的路径中获取")
            
        # 归一化处理 (先归一化，再应用变换)
        normalized_source_points = source_points.copy()
        normalized_target_points = target_points.copy()
        normalized_source_cloud = point_cloud_for_transform.copy()
        normalized_target_cloud = target_point_cloud.copy()
        
        if normalize:
            if verbose:
                print("对源点云和目标点云进行归一化...")
            
            # 使用共享参数对源点云和目标点云进行归一化 (质心取目标点云的，缩放因子取较大的)
            normalized_source_points, normalized_target_points, scale_factor, centroid = normalize_point_clouds_with_shared_params(
                source_points, target_points, verbose=verbose
            )
            
            # 同样对[3, N]格式的点云进行归一化
            normalized_source_cloud = normalized_source_points.T if normalized_source_points.shape[1] == 3 else normalized_source_points
            normalized_target_cloud = normalized_target_points.T if normalized_target_points.shape[1] == 3 else normalized_target_points
            
            # 更新源点云为归一化后的点云
            source_points = normalized_source_points
            target_points = normalized_target_points
            point_cloud_for_transform = normalized_source_cloud
            target_point_cloud = normalized_target_cloud
        
        # 如果指定了random_transform，对归一化后的源点云进行随机旋转平移
        if random_transform:
            # 生成随机旋转矩阵和平移向量
            rotation_matrix = random_rotation_matrix()
            
            if rotation_only:
                translation_vector = np.zeros(3, dtype=np.float32)  # 全零平移向量
                if verbose:
                    print("只应用旋转，不应用平移")
            else:
                translation_vector = random_translation_vector()
            
            if verbose:
                print("变换矩阵信息:")
                print("旋转矩阵:")
                print(rotation_matrix)
                print("平移向量:")
                print(translation_vector)
                print("注意：只对源点云应用随机旋转，目标点云保持原样")
            
            # 应用变换到源点云 (变换归一化后的点云)
            transformed_source_cloud = transform_point_cloud(
                point_cloud_for_transform, 
                rotation_matrix, 
                translation_vector,
                rotation_only
            )
            
            # 更新源点云和用于处理的数据
            if source_points.shape[1] == 3:
                # 如果原始点云是[N, 3]格式，将变换后的点云转换回该格式
                transformed_source_points = transformed_source_cloud.T
            else:
                transformed_source_points = transformed_source_cloud
                
            # 创建蛋白质所属的子目录
            protein_cif_dir = os.path.join(cif_data_dir, source_base_name)
            if verbose:
                print(f"创建蛋白质目录: {protein_cif_dir}")
            os.makedirs(protein_cif_dir, exist_ok=True)
            
            # 保存原始源点云为CIF文件，放在子目录中
            try:
                if verbose:
                    print(f"准备保存原始源点云到: {source_base_name}_original_source.cif")
                    print(f"点云形状: {normalized_source_points.shape}, 原子类型数: {len(source_atom_types)}, 原子行数: {len(source_atom_lines)}")
                
                output_original_source_cif = os.path.join(protein_cif_dir, f"{source_base_name}_original_source.cif")
                save_as_cif(normalized_source_points, source_atom_types, source_atom_labels, source_cif_content, source_atom_lines, output_original_source_cif)
            except Exception as e:
                print(f"保存原始源点云时出错: {str(e)}")
                print(f"出错的变量和值: \n"
                      f"  source_base_name = {source_base_name}\n"
                      f"  protein_cif_dir = {protein_cif_dir}")
                raise
            
            # 更新源点云为变换后的点云
            source_points = transformed_source_points
            point_cloud_for_transform = transformed_source_cloud
        else:
            # 不使用随机变换，设置为单位矩阵和零向量
            rotation_matrix = np.eye(3, dtype=np.float32)
            translation_vector = np.zeros(3, dtype=np.float32)
        
        # 为返回值初始化输出路径变量
        # 注意：实际文件在下面重新指定的地方保存
        output_source_cif = os.path.join(cif_data_dir, source_base_name, f"{source_base_name}_source.cif")
        output_target_cif = os.path.join(cif_data_dir, source_base_name, f"{source_base_name}_target.cif")
        
        # 创建网格信息
        grid_info = create_grid_info(normalize=False)  # 初始创建时不设置归一化标记
        
        # 手动添加归一化信息到grid_info
        if normalize:
            grid_info['normalized'] = True
            grid_info['scale_factor'] = float(scale_factor)
            grid_info['original_centroid'] = centroid.tolist()
            if verbose:
                print(f"已将归一化信息添加到grid_info: 缩放因子={scale_factor}, 原始质心={centroid}")
        else:
            grid_info['normalized'] = False
        
        # 创建并保存PKL文件（在主目录中）
        if random_transform:
            # 使用随机生成的旋转矩阵和平移向量（只应用于源点云）
            create_pkl_file(
                point_cloud_for_transform, 
                target_point_cloud, 
                output_pkl, 
                grid_info, 
                rotation=rotation_matrix,    # 使用应用于源点云的旋转矩阵
                translation=translation_vector,  # 使用应用于源点云的平移向量
                normalize=False,  # 设置为False避免重复归一化
                verbose=verbose
            )
            
            # 不为原始源点云生成PKL文件
            if verbose:
                print(f"只保存原始源点云的CIF文件，不生成PKL文件")
        else:
            # 不使用随机变换
            if verbose:
                print("不应用随机旋转，使用单位矩阵和零向量")
                
            create_pkl_file(
                point_cloud_for_transform, 
                target_point_cloud, 
                output_pkl, 
                grid_info, 
                rotation=rotation_matrix,  # 这里是单位矩阵
                translation=translation_vector,  # 这里是零向量
                normalize=False,  # 设置为False避免重复归一化
                verbose=verbose
            )
            
        # 保存目标和源点云为CIF文件（在蛋白质属于的子目录中）
        # 使用蛋白质ID作为子目录名称，而不是源文件的完整名称
        protein_cif_dir = os.path.join(cif_data_dir, protein_id)
        os.makedirs(protein_cif_dir, exist_ok=True)
        
        if verbose:
            print(f"使用蛋白质ID: {protein_id} 构建目录和文件名")
        
        # 保存原始源点云(如果存在)
        if random_transform and os.path.exists(os.path.join(output_dir, f"{output_base_name}_original_source.cif")):
            original_source_cif = os.path.join(output_dir, f"{output_base_name}_original_source.cif")
            output_original_source_cif = os.path.join(protein_cif_dir, f"{protein_id}_original_source.cif")
            # 将原始文件复制到新位置
            shutil.copy2(original_source_cif, output_original_source_cif)
            
        # 保存变换后的源点云
        output_source_cif = os.path.join(protein_cif_dir, f"{protein_id}_source.cif")
        save_as_cif(source_points, source_atom_types, source_atom_labels, source_cif_content, source_atom_lines, output_source_cif)
        
        # 保存目标点云
        try:
            if verbose:
                print(f"准备保存目标点云到: {protein_id}_target.cif")
                print(f"点云形状: {target_points.shape}, 原子类型数: {len(target_atom_types)}, 原子行数: {len(target_atom_lines)}")
            
            protein_cif_dir = os.path.join(cif_data_dir, protein_id)
            os.makedirs(protein_cif_dir, exist_ok=True)
            
            output_target_cif = os.path.join(protein_cif_dir, f"{protein_id}_target.cif")
            save_as_cif(target_points, target_atom_types, target_atom_labels, target_cif_content, target_atom_lines, output_target_cif)
        except Exception as e:
            print(f"保存目标点云时出错: {str(e)}")
            print(f"出错的变量和值: \n"
                  f"  protein_id = {protein_id}\n"
                  f"  cif_data_dir = {cif_data_dir}\n"
                  f"  protein_cif_dir = {os.path.join(cif_data_dir, protein_id)}")
            raise
        
        # 另外，如果目标文件存在，也保存原始的目标文件
        if target_cif is not None:
            target_original_cif = os.path.join(protein_cif_dir, f"{protein_id}_original_target.cif")
            # 直接复制原始目标文件
            shutil.copy2(target_cif, target_original_cif)
        
        # 如果需要可视化，则可视化点云
        if visualize:
            if verbose:
                print("执行点云可视化...")
            visualize_point_clouds(point_cloud_for_transform, target_point_cloud, title=f"源点云和目标点云: {output_base_name}")
        
        if verbose:
            print(f"文件对{source_cif}和{target_cif}处理完成!")
        return output_pkl, output_source_cif, output_target_cif
    
    except Exception as e:
        print(f"处理文件对 {source_cif} 和 {target_cif} 时出错: {str(e)}")
        return None, None, None

def batch_process(source_dir, target_dir, output_dir, visualize=False, normalize=False, parallel=True, max_workers=None, random_transform=False, rotation_only=False):
    """
    批量处理两个目录中的CIF文件对
    
    参数:
        source_dir: 源CIF文件目录
        target_dir: 目标CIF文件目录（如果random_transform为True，则此参数可以为None）
        output_dir: 输出目录
        visualize: 是否可视化点云，默认为False
        normalize: 是否归一化点云，默认为False
        parallel: 是否并行处理，默认为True
        max_workers: 最大线程数，默认为None（使用CPU核心数）
        random_transform: 是否应用随机旋转平移变换，默认为False
        rotation_only: 是否只应用旋转变换不应用平移，默认为False
    
    返回:
        processed_files: 已处理文件对的列表，每项为(output_pkl, output_source_cif, output_target_cif)
    """
    # 获取所有源CIF文件
    source_files = glob.glob(os.path.join(source_dir, "*.cif"))
    
    # 打印源目录和目标目录中的文件列表，以便调试
    print(f"源目录: {source_dir}")
    print(f"源目录中找到 {len(source_files)} 个CIF文件")
    if len(source_files) > 0:
        print("前5个源文件:")
        for i, f in enumerate(source_files[:5]):
            print(f"  {i+1}. {os.path.basename(f)}")
    
    if target_dir:
        target_files = glob.glob(os.path.join(target_dir, "*.cif"))
        print(f"目标目录: {target_dir}")
        print(f"目标目录中找到 {len(target_files)} 个CIF文件")
        if len(target_files) > 0:
            print("前5个目标文件:")
            for i, f in enumerate(target_files[:5]):
                print(f"  {i+1}. {os.path.basename(f)}")
    
    if not source_files:
        print(f"错误：源目录 {source_dir} 中未找到CIF文件")
        return []
    
    # 生成需要处理的文件对
    file_pairs = []
    
    # 绝对不要将只有源文件添加到品就列表中
    if target_dir is None:
        print(f"错误：必须提供目标目录，源点云和目标点云必须分别从各自的目录中获取")
        return []
    
    # 有目标目录，尝试匹配源文件和目标文件
    # 从目标目录中获取目标文件
    print(f"使用目标目录: {target_dir}")
    
    # 通过提取蛋白质ID查找目标文件
    for source_file in source_files:
        source_basename = os.path.basename(source_file)
        
        # 从源文件名提取蛋白质ID（如从"8h1p_point.cif"提取"8h1p"）
        protein_id = source_basename.split("_")[0] if "_" in source_basename else os.path.splitext(source_basename)[0]
        
        # 尝试不同的目标文件命名模式
        target_basename_patterns = [
            f"{protein_id}_segment.cif",  # 如"8h1p_segment.cif"
            f"{protein_id}_mrc2cif.cif",  # 如"8h1p_mrc2cif.cif"
            f"{protein_id}.cif",          # 如"8h1p.cif"
            source_basename               # 相同的文件名
        ]
        
        found_match = False
        for pattern in target_basename_patterns:
            target_file = os.path.join(target_dir, pattern)
            if os.path.exists(target_file):
                file_pairs.append((source_file, target_file))
                print(f"匹配到文件对: {source_basename} -> {pattern}")
                found_match = True
                break
        
        if not found_match:
            print(f"警告：未找到与 {protein_id} 对应的目标文件，尝试了以下模式: {', '.join(target_basename_patterns)}")
    
    print(f"找到 {len(file_pairs)} 对需要处理的文件")
    if random_transform:
        print(f"将应用随机旋转平移到源点云")
        if rotation_only:
            print("只应用旋转变换，不应用平移变换")
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        print(f"创建输出目录: {output_dir}")
    
    processed_files = []
    
    # 根据是否并行处理选择不同的处理方式
    if parallel:
        # 自动设置并行处理的线程数量
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        print(f"使用并行处理，线程数: {max_workers}")
        
        # 创建并行实例化的process_pair函数
        process_func = partial(process_pair_wrapper, 
                             output_dir=output_dir, 
                             visualize=visualize, 
                             normalize=normalize, 
                             verbose=False,
                             random_transform=random_transform,
                             rotation_only=rotation_only)
        
        # 使用并行执行器进行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_func, source_file, target_file) for source_file, target_file in file_pairs]
            
            # 显示进度条
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理文件对")):
                try:
                    result = future.result()
                    if result[0] is not None:
                        processed_files.append(result)
                except Exception as e:
                    print(f"并行处理时出错: {str(e)}")
    else:
        # 使用顺序处理
        print("使用顺序处理")
        print(f"随机变换: {random_transform}, 只旋转: {rotation_only}")
        for i, (source_file, target_file) in enumerate(tqdm(file_pairs, desc="处理文件对")):
            result = process_pair(source_file, target_file, output_dir, visualize, normalize, 
                              verbose=True, random_transform=random_transform, rotation_only=rotation_only)
            if result[0] is not None:
                processed_files.append(result)
    
    print(f"批量处理完成! 共处理了 {len(processed_files)} 对文件")
    return processed_files


def process_pair_wrapper(source_file, target_file, output_dir, visualize=False, normalize=False, verbose=False, random_transform=False, rotation_only=False):
    """用于并行处理的封装函数，捕获异常并返回结果"""
    try:
        return process_pair(source_file, target_file, output_dir, visualize, normalize, verbose, random_transform, rotation_only)
    except Exception as e:
        print(f"处理文件对 {source_file} 和 {target_file} 时出错: {str(e)}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='从两个CIF文件创建点云数据，支持随机旋转平移或使用固定变换')
    parser.add_argument('--source_cif', type=str, help='源CIF文件路径')
    parser.add_argument('--target_cif', type=str, help='目标CIF文件路径')
    parser.add_argument('--source_dir', type=str, help='源CIF文件目录，用于批处理')
    parser.add_argument('--target_dir', type=str, help='目标CIF文件目录，用于批处理')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='是否可视化点云')
    parser.add_argument('--normalize', action='store_true', help='是否归一化点云')
    parser.add_argument('--no_parallel', action='store_true', help='禁用并行处理')
    parser.add_argument('--max_workers', type=int, default=None, help='并行处理的最大线程数，默认为CPU核心数')
    parser.add_argument('--random_transform', action='store_true', help='是否应用随机旋转平移变换')
    parser.add_argument('--rotation_only', action='store_true', help='是否只应用旋转变换不应用平移')
    
    args = parser.parse_args()
    
    # 设置是否并行处理
    parallel = not args.no_parallel
    
    # 检查参数组合
    if args.random_transform:
        print("启用随机旋转平移变换模式")
        if args.rotation_only:
            print("只应用旋转变换，不应用平移")
    
    if (args.source_cif):
        # 单文件处理模式
        if args.random_transform:
            # 随机变换模式，不需要目标文件
            print(f"开始处理单个文件，生成随机变换: {args.source_cif}")
            result = process_pair(args.source_cif, None, args.output_dir, args.visualize, args.normalize, 
                              random_transform=True, rotation_only=args.rotation_only)
        else:
            # 需要目标文件
            if not args.target_cif:
                parser.print_help()
                print("\n错误：当不使用随机变换时，需要提供目标CIF文件")
                return
            
            print(f"开始处理单对文件: {args.source_cif} 和 {args.target_cif}")
            result = process_pair(args.source_cif, args.target_cif, args.output_dir, args.visualize, args.normalize)
        
        if result[0] is not None:
            print(f"成功创建PKL文件: {result[0]}")
        else:
            print("处理失败")
    elif (args.source_dir):
        # 批处理模式
        if args.random_transform:
            # 随机变换模式
            print(f"开始批量处理，生成随机变换: 从 {args.source_dir} 到 {args.output_dir}")
            print(f"目标目录: {args.target_dir if args.target_dir else '未提供'}")
            print(f"并行处理: {parallel}, 可视化: {args.visualize}, 归一化: {args.normalize}, 只旋转: {args.rotation_only}")
            results = batch_process(
                args.source_dir, 
                args.target_dir,  # 使用用户提供的目标目录
                args.output_dir, 
                args.visualize, 
                args.normalize, 
                parallel=parallel, 
                max_workers=args.max_workers,
                random_transform=True,
                rotation_only=args.rotation_only
            )
        else:
            # 需要目标目录
            if not args.target_dir:
                parser.print_help()
                print("\n错误：当不使用随机变换时，需要提供目标目录")
                return
                
            print(f"开始批量处理: 从 {args.source_dir} 和 {args.target_dir} 到 {args.output_dir}")
            print(f"并行处理: {parallel}, 可视化: {args.visualize}, 归一化: {args.normalize}")
            results = batch_process(
                args.source_dir, 
                args.target_dir, 
                args.output_dir, 
                args.visualize, 
                args.normalize, 
                parallel=parallel, 
                max_workers=args.max_workers
            )
        
        print(f"批处理完成，共生成了 {len(results)} 个PKL文件")
    else:
        parser.print_help()
        print("\n错误：请提供有效的参数组合：")
        print("1. --source_cif [和 --target_cif] 用于处理单文件或单对文件")
        print("2. --source_dir [和 --target_dir] 用于批处理")
        print("3. 添加 --random_transform 可以使用随机旋转平移变换，此时不需要target目标文件或目录")

if __name__ == '__main__':
    main()
# python pkldataprocess.py --source_dir /zhaoxuanj/Point_dataset/5000downcif --target_dir /zhaoxuanj/Point_dataset/381mrc2cif --random_transform --output_dir /zhaoxuanj/Point_dataset/381_mrc_pdb_pkldataset --normalize