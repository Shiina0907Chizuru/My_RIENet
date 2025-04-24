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
    """解析CIF文件，仅提取CA原子坐标"""
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
    
    # 解析CIF文件
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 检测原子行（以ATOM开头）
        if line.startswith('ATOM'):
            parts = line.split()
            if len(parts) >= 13:  # 确保有足够的列
                try:
                    atom_type = parts[3]  # 原子类型
                    
                    # 仅处理CA原子
                    if atom_type == "CA":
                        atom_number = parts[1]
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
        raise ValueError("未能从CIF文件中提取到CA原子坐标")
    
    print(f"成功从CIF文件解析 {len(atoms)} 个CA原子")
    
    # 转换为numpy数组
    return np.array(atoms, dtype=np.float32), atom_types, atom_labels, cif_content, atom_lines

def parse_pdb(pdb_path):
    """解析PDB文件，仅提取CA原子坐标"""
    print(f"正在解析PDB文件: {pdb_path}")
    
    atoms = []
    atom_types = []
    atom_labels = []
    
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    
    # 解析PDB文件
    for line in lines:
        if line.startswith('ATOM  '):
            # PDB标准格式中，CA原子信息位于特定位置
            atom_name = line[12:16].strip()
            
            # 只提取CA原子
            if atom_name == "CA":
                try:
                    resname = line[17:20].strip()  # 残基名
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    atoms.append([x, y, z])
                    atom_types.append(atom_name)
                    atom_labels.append(resname)
                except (ValueError, IndexError) as e:
                    print(f"警告：解析PDB行时出错: {e}\n{line}")
                    continue
    
    # 检查是否成功提取到原子坐标
    if not atoms:
        raise ValueError("未能从PDB文件中提取到CA原子坐标")
    
    print(f"成功从PDB文件解析 {len(atoms)} 个CA原子")
    
    # 转换为numpy数组
    return np.array(atoms, dtype=np.float32), atom_types, atom_labels

def load_structure_file(file_path):
    """根据文件类型加载结构文件，返回点云数据"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.cif':
        atoms, atom_types, atom_labels, _, _ = parse_cif(file_path)
        return atoms, atom_types, atom_labels
    elif file_ext == '.pdb':
        atoms, atom_types, atom_labels = parse_pdb(file_path)
        return atoms, atom_types, atom_labels
    else:
        raise ValueError(f"不支持的文件类型: {file_ext}，仅支持.cif和.pdb格式")

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

def calculate_rotation_translation(source_points, target_points):
    """
    计算从源点云到目标点云的最优旋转和平移
    使用SVD方法实现点云配准
    """
    # 确保点云是[N, 3]格式
    if source_points.shape[0] == 3 and source_points.shape[1] != 3:
        source_points = source_points.T
    if target_points.shape[0] == 3 and target_points.shape[1] != 3:
        target_points = target_points.T
    
    # 计算质心
    source_centroid = calculate_centroid(source_points)
    target_centroid = calculate_centroid(target_points)
    
    # 将点云中心化
    centered_source = source_points - source_centroid
    centered_target = target_points - target_centroid
    
    # 计算协方差矩阵
    H = np.dot(centered_source.T, centered_target)
    
    # 对协方差矩阵进行SVD分解
    U, _, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = np.dot(Vt.T, U.T)
    
    # 确保旋转矩阵具有正确的行列式（特殊正交矩阵）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # 计算平移向量
    t = target_centroid - np.dot(R, source_centroid)
    
    return R.astype(np.float32), t.astype(np.float32)

def create_pkl_file(source_points, target_points, rotation, translation, output_path, grid_info):
    """创建pkl文件，保存为PointCloud_ca_Dataset格式"""
    # 确保点云为[N, 3]格式
    if source_points.shape[0] == 3 and source_points.shape[1] != 3:
        source_points_pkl = source_points.T
    else:
        source_points_pkl = source_points.copy()
    
    if target_points.shape[0] == 3 and target_points.shape[1] != 3:
        target_points_pkl = target_points.T
    else:
        target_points_pkl = target_points.copy()
    
    # 解包grid_info
    grid_shape, x_origin, y_origin, z_origin, x_voxel, nstart = grid_info
    
    # 创建数据字典
    data_dict = {
        'source_point_cloud': torch.from_numpy(source_points_pkl),
        'target_point_cloud': torch.from_numpy(target_points_pkl),
        'rot': torch.from_numpy(rotation),
        't': torch.from_numpy(translation),
        'grid_shape': torch.from_numpy(grid_shape),
        'x_origin': torch.from_numpy(x_origin),
        'y_origin': torch.from_numpy(y_origin),
        'z_origin': torch.from_numpy(z_origin),
        'x_voxel': torch.from_numpy(x_voxel),
        'nstart': torch.from_numpy(nstart)
    }
    
    # 保存为pkl文件
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"已保存点云数据到: {output_path}")
    print(f"源点云形状: {source_points_pkl.shape}")
    print(f"目标点云形状: {target_points_pkl.shape}")
    print(f"旋转矩阵: \n{rotation}")
    print(f"平移向量: {translation}")

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

def process_data_files(source_file, target_file, output_path, visualize=False):
    """处理源文件和目标文件，生成点云配准数据"""
    # 加载源结构和目标结构
    source_points, source_atom_types, source_atom_labels = load_structure_file(source_file)
    target_points, target_atom_types, target_atom_labels = load_structure_file(target_file)
    
    print(f"源点云大小: {source_points.shape}")
    print(f"目标点云大小: {target_points.shape}")
    
    # 计算旋转和平移
    rotation, translation = calculate_rotation_translation(source_points, target_points)
    
    # 创建网格信息
    grid_info = create_grid_info()
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建并保存pkl文件
    create_pkl_file(source_points, target_points, rotation, translation, output_path, grid_info)
    
    # 如果需要，可视化点云
    if visualize:
        visualize_point_clouds(source_points, target_points, title="源点云和目标点云")
    
    print(f"处理完成，数据已保存到 {output_path}")

def main():
    parser = argparse.ArgumentParser(description='从两个结构文件创建点云配准数据')
    parser.add_argument('--source', type=str, required=True, help='源结构文件路径（.cif或.pdb）')
    parser.add_argument('--target', type=str, required=True, help='目标结构文件路径（.cif或.pdb）')
    parser.add_argument('--output', type=str, default='output.pkl', help='输出pkl文件路径')
    parser.add_argument('--visualize', action='store_true', help='是否可视化点云')
    
    args = parser.parse_args()
    
    # 检查源文件和目标文件是否存在
    if not os.path.exists(args.source):
        print(f"错误：源文件 {args.source} 不存在")
        return
    
    if not os.path.exists(args.target):
        print(f"错误：目标文件 {args.target} 不存在")
        return
    
    # 处理文件
    process_data_files(args.source, args.target, args.output, args.visualize)

if __name__ == '__main__':
    main()
