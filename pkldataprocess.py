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

def parse_cif(cif_path):
    """解析CIF文件，提取原子坐标和完整CIF内容"""
    print(f"正在解析CIF文件: {cif_path}")
    
    atoms = []
    atom_lines = []
    atom_types = []
    atom_labels = []
    cif_content = []
    
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
                    atom_type = parts[3]  # CA 原子类型
                    atom_label = parts[5]  # 残基名称（如 ALA）
                    
                    # 获取坐标（第10、11、12列，空格分隔）
                    x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                    
                    atoms.append([x, y, z])
                    atom_types.append(atom_type)
                    atom_labels.append(atom_label)
                    atom_lines.append(i)
                except (ValueError, IndexError) as e:
                    print(f"警告：无法解析第 {i+1} 行: {e}")
                    continue
    
    # 检查是否成功提取到原子坐标
    if not atoms:
        raise ValueError("未能从CIF文件中提取到原子坐标")
    
    print(f"成功解析 {len(atoms)} 个原子")
    
    # 转换为numpy数组
    return np.array(atoms, dtype=np.float32), atom_types, atom_labels, cif_content, atom_lines

def create_grid_info():
    """创建体素网格信息"""
    # 这里生成一些默认值，实际应用中需要根据点云范围来确定
    grid_shape = np.array([64, 64, 64], dtype=np.int32)
    
    # 将标量值改为一维数组，避免索引越界问题
    x_origin = np.array([0.0], dtype=np.float32)
    y_origin = np.array([0.0], dtype=np.float32)
    z_origin = np.array([0.0], dtype=np.float32)
    x_voxel = np.array([0.05], dtype=np.float32)  # 体素大小
    nstart = np.array([32], dtype=np.int32)  # 起始索引
    
    return grid_shape, x_origin, y_origin, z_origin, x_voxel, nstart

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

def create_pkl_file(source_points, target_points, output_path, grid_info):
    """创建pkl文件，保存为PointCloud_ca_Dataset格式，旋转平移矩阵置零"""
    grid_shape, x_origin, y_origin, z_origin, x_voxel, nstart = grid_info
    
    # 检查点云形状并打印信息
    print(f"保存前源点云形状: {source_points.shape}")
    print(f"保存前目标点云形状: {target_points.shape}")
    
    # 转换为torch张量
    source_points_tensor = torch.from_numpy(source_points).float()
    target_points_tensor = torch.from_numpy(target_points).float()
    
    # 创建零旋转矩阵（单位矩阵）和零平移向量
    rotation_tensor = torch.eye(3, dtype=torch.float32)
    translation_tensor = torch.zeros(3, dtype=torch.float32)
    
    print(f"保存为torch张量后源点云形状: {source_points_tensor.shape}")
    print(f"保存为torch张量后目标点云形状: {target_points_tensor.shape}")
    
    # 保存为字典格式
    data_dict = {
        'source': source_points_tensor,  # 源点云
        'target': target_points_tensor,  # 目标点云
        'rotation': rotation_tensor,   # 旋转矩阵（单位矩阵）
        'translation': translation_tensor,  # 平移向量（零向量）
        'grid_shape': grid_shape,
        'x_origin': x_origin,
        'y_origin': y_origin,
        'z_origin': z_origin,
        'x_voxel': x_voxel,
        'nstart': nstart
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"已保存到: {output_path}")
    print(f"保存的源点云形状: {source_points_tensor.shape}")
    print(f"保存的目标点云形状: {target_points_tensor.shape}")
    print(f"保存的旋转矩阵: 单位矩阵")
    print(f"保存的平移向量: 零向量")

def save_as_cif(coordinates, atom_types, atom_labels, original_cif, atom_lines, output_path):
    """将点云坐标保存为CIF格式"""
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

def process_pair(source_cif, target_cif, output_dir, visualize=False):
    """处理一对CIF文件，创建点云数据"""
    print(f"处理源文件: {source_cif}")
    print(f"处理目标文件: {target_cif}")
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 获取源文件名（不带路径和扩展名）
    source_base_name = os.path.splitext(os.path.basename(source_cif))[0]
    target_base_name = os.path.splitext(os.path.basename(target_cif))[0]
    output_base_name = f"{source_base_name}_{target_base_name}"
    
    # 解析CIF文件
    source_points, source_atom_types, source_atom_labels, source_cif_content, source_atom_lines = parse_cif(source_cif)
    target_points, target_atom_types, target_atom_labels, target_cif_content, target_atom_lines = parse_cif(target_cif)
    
    # 创建输出文件路径
    output_pkl = os.path.join(output_dir, f"{output_base_name}.pkl")
    output_source_cif = os.path.join(output_dir, f"{output_base_name}_source.cif")
    output_target_cif = os.path.join(output_dir, f"{output_base_name}_target.cif")
    
    # 记录点云大小信息，但不再强制匹配大小
    if source_points.shape[0] != target_points.shape[0]:
        print(f"注意：源点云有 {source_points.shape[0]} 个点，目标点云有 {target_points.shape[0]} 个点")
    
    # 确保点云为[3, N]格式用于处理
    point_cloud_for_transform = source_points.T if source_points.shape[1] == 3 else source_points
    target_point_cloud = target_points.T if target_points.shape[1] == 3 else target_points
    
    # 创建网格信息
    grid_info = create_grid_info()
    
    # 保存源点云和目标点云为CIF文件
    save_as_cif(source_points, source_atom_types, source_atom_labels, source_cif_content, source_atom_lines, output_source_cif)
    save_as_cif(target_points, target_atom_types, target_atom_labels, target_cif_content, target_atom_lines, output_target_cif)
    
    # 创建并保存PKL文件（旋转平移矩阵置零）
    create_pkl_file(point_cloud_for_transform, target_point_cloud, output_pkl, grid_info)
    
    # 如果需要可视化，则可视化点云
    if visualize:
        print("执行点云可视化...")
        visualize_point_clouds(point_cloud_for_transform, target_point_cloud, title=f"源点云和目标点云: {output_base_name}")
    
    print(f"文件对{source_cif}和{target_cif}处理完成!")
    return output_pkl, output_source_cif, output_target_cif

def batch_process(source_dir, target_dir, output_dir, visualize=False):
    """批量处理两个目录中的CIF文件对"""
    # 获取所有源CIF文件
    source_files = glob.glob(os.path.join(source_dir, "*.cif"))
    
    # 获取所有目标CIF文件
    target_files = glob.glob(os.path.join(target_dir, "*.cif"))
    
    if not source_files:
        print(f"错误：源目录 {source_dir} 中未找到CIF文件")
        return
    
    if not target_files:
        print(f"错误：目标目录 {target_dir} 中未找到CIF文件")
        return
    
    print(f"找到 {len(source_files)} 个源CIF文件和 {len(target_files)} 个目标CIF文件")
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 处理每对文件
    for i, source_file in enumerate(tqdm(source_files, desc="处理源文件")):
        for target_file in tqdm(target_files, desc=f"处理源文件 {i+1}/{len(source_files)} 的目标文件", leave=False):
            process_pair(source_file, target_file, output_dir, visualize)
    
    print("批量处理完成!")

def main():
    parser = argparse.ArgumentParser(description='从两个CIF文件创建点云数据（旋转平移矩阵置零）')
    parser.add_argument('--source_cif', type=str, help='源CIF文件路径')
    parser.add_argument('--target_cif', type=str, help='目标CIF文件路径')
    parser.add_argument('--source_dir', type=str, help='源CIF文件目录，用于批处理')
    parser.add_argument('--target_dir', type=str, help='目标CIF文件目录，用于批处理')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='是否可视化点云')
    
    args = parser.parse_args()
    
    # 检查参数组合
    if (args.source_cif and args.target_cif):
        # 单对文件处理模式
        process_pair(args.source_cif, args.target_cif, args.output_dir, args.visualize)
    elif (args.source_dir and args.target_dir):
        # 批处理模式
        batch_process(args.source_dir, args.target_dir, args.output_dir, args.visualize)
    else:
        parser.print_help()
        print("\n错误：请提供有效的参数组合：")
        print("1. --source_cif 和 --target_cif 用于处理单对文件")
        print("2. --source_dir 和 --target_dir 用于批处理")

if __name__ == '__main__':
    main()
