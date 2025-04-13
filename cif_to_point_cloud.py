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

def transform_point_cloud_numpy(point_cloud, rotation, translation):
    """使用numpy将点云进行变换 - 使用右乘方式：点云 = 旋转矩阵 * 点云 + 平移向量"""
    # 确保点云是[3, N]格式
    if point_cloud.shape[1] == 3 and point_cloud.shape[0] != 3:
        print(f"转置点云: 从{point_cloud.shape}到", end="")
        point_cloud = point_cloud.T
        print(f"{point_cloud.shape}")
    
    # 现在point_cloud已经是[3, N]格式，可以直接应用旋转和平移
    # R[3,3] * point_cloud[3,N] + t[3,1] = transformed_point_cloud[3,N]
    transformed_point_cloud = np.matmul(rotation, point_cloud) + translation.reshape(3, 1)
    
    return transformed_point_cloud

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

def create_pkl_file(source_points, target_points, rotation, translation, output_path, grid_info):
    """创建pkl文件，保存为PointCloud_ca_Dataset格式"""
    grid_shape, x_origin, y_origin, z_origin, x_voxel, nstart = grid_info
    
    # 检查点云形状并打印信息
    print(f"保存前源点云形状: {source_points.shape}")
    print(f"保存前目标点云形状: {target_points.shape}")
    
    # 确保点云是[3, N]格式
    if source_points.shape[1] == 3 and source_points.shape[0] != 3:
        print(f"转置点云: 从{source_points.shape}到", end="")
        source_points = source_points.T
        print(f"{source_points.shape}")
    
    if target_points.shape[1] == 3 and target_points.shape[0] != 3:
        print(f"转置点云: 从{target_points.shape}到", end="")
        target_points = target_points.T
        print(f"{target_points.shape}")
    
    data_dict = {
        'source': source_points,  
        'target': target_points,
        'rotation': rotation,
        'translation': translation,
        'grid': grid_shape,
        'x_origin': x_origin,
        'y_origin': y_origin,
        'z_origin': z_origin,
        'x_voxel': x_voxel,
        'nstart': nstart
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"已保存到: {output_path}")
    print(f"保存的源点云形状: {source_points.shape}")
    print(f"保存的目标点云形状: {target_points.shape}")

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
        
        new_cif[line_idx] = "  ".join(line) + "\n"
    
    # 添加注释说明这是变换后的结构
    header_comment = f"# This is a transformed structure\n"
    new_cif.insert(0, header_comment)
    
    # 写入新的CIF文件
    with open(output_path, 'w') as f:
        f.writelines(new_cif)
    
    print(f"已保存目标点云为CIF格式: {output_path}")

def visualize_point_clouds(source, target, title="点云可视化"):
    """可视化源点云和目标点云"""
    fig = plt.figure(figsize=(10, 5))
    
    # 绘制源点云
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(source[:, 0], source[:, 1], source[:, 2], c='blue', s=1)
    ax1.set_title("源点云")
    
    # 绘制目标点云
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(target[:, 0], target[:, 1], target[:, 2], c='red', s=1)
    ax2.set_title("目标点云")
    
    plt.suptitle(title)
    plt.savefig("point_cloud_visualization.png")
    print("点云可视化已保存到 point_cloud_visualization.png")
    # 默认不显示图像，只保存
    # plt.show()

def process_batch(batch_dir, output_dir, visualize=False):
    """批量处理指定目录下所有子文件夹中的xxxx_point.cif文件"""
    print(f"开始批处理目录: {batch_dir}")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 获取所有子文件夹
    subdirectories = [os.path.join(batch_dir, d) for d in os.listdir(batch_dir) 
                     if os.path.isdir(os.path.join(batch_dir, d))]
    
    if not subdirectories:
        print(f"警告: 在{batch_dir}中未找到子文件夹")
        return
    
    total_processed = 0
    
    # 遍历每个子文件夹
    for subdir in tqdm(subdirectories, desc="处理子文件夹"):
        # 在子文件夹中查找符合条件的cif文件 (xxxx_point.cif)
        cif_files = [f for f in os.listdir(subdir) 
                    if os.path.isfile(os.path.join(subdir, f)) and f.endswith('_point.cif')]
        
        if not cif_files:
            print(f"在{subdir}中未找到xxxx_point.cif文件，跳过")
            continue
            
        # 对每个找到的文件进行处理
        for cif_file in cif_files:
            cif_path = os.path.join(subdir, cif_file)
            
            # 创建对应的输出子目录（保持原有目录结构）
            subdir_name = os.path.basename(subdir)
            sub_output_dir = os.path.join(output_dir, subdir_name)
            if not os.path.exists(sub_output_dir):
                os.makedirs(sub_output_dir)
            
            try:
                # 处理单个cif文件（复用现有函数）
                process_single_file(cif_path, sub_output_dir, visualize)
                total_processed += 1
            except Exception as e:
                print(f"处理文件{cif_path}时出错: {e}")
    
    print(f"批处理完成! 共处理了{total_processed}个文件")

def process_single_file(cif_path, output_dir, visualize=False):
    """处理单个cif文件"""
    print(f"处理文件: {cif_path}")
    
    # 获取输入cif文件的基本名称（不含路径和扩展名）
    base_name = os.path.splitext(os.path.basename(cif_path))[0]
    
    # 生成输出文件路径
    output_pkl = os.path.join(output_dir, f"{base_name}.pkl")
    output_cif = os.path.join(output_dir, f"{base_name}_target.cif")
    
    print(f"输入CIF文件: {cif_path}")
    print(f"输出PKL文件: {output_pkl}")
    print(f"输出目标CIF文件: {output_cif}")
    
    # 解析CIF文件
    try:
        point_cloud, atom_types, atom_labels, cif_content, atom_lines = parse_cif(cif_path)
    except Exception as e:
        print(f"解析CIF文件时出错：{e}")
        raise e
    
    print(f"成功提取点云数据，共 {len(point_cloud)} 个点")
    
    # 生成随机旋转矩阵和平移向量
    rotation_matrix = random_rotation_matrix()
    translation_vector = random_translation_vector()
    
    print("旋转矩阵:")
    print(rotation_matrix)
    print("平移向量:")
    print(translation_vector)
    
    # 变换点云生成目标点云 - 应用变换：R * point_cloud + t
    target_point_cloud = transform_point_cloud_numpy(point_cloud, rotation_matrix, translation_vector)
    
    # 创建网格信息
    grid_info = create_grid_info()
    
    # 创建并保存pkl文件
    create_pkl_file(point_cloud, target_point_cloud, rotation_matrix, translation_vector, output_pkl, grid_info)
    
    # 将目标点云转置回[N, 3]格式，用于保存为CIF文件
    target_point_cloud_for_cif = target_point_cloud.T
    
    # 保存目标结构为CIF格式
    save_as_cif(target_point_cloud_for_cif, atom_types, atom_labels, cif_content, atom_lines, output_cif)
    
    # 只有当明确指定--visualize参数时才执行可视化
    if visualize:
        print("执行点云可视化...")
        visualize_point_clouds(point_cloud, target_point_cloud)
    
    print(f"文件{cif_path}处理完成!")

def main():
    parser = argparse.ArgumentParser(description='CIF文件转换为点云数据并进行变换')
    parser.add_argument('cif_path', type=str, nargs='?', help='CIF文件路径')
    parser.add_argument('--batch_dir', type=str, help='批处理目录路径，指定该参数将进行批处理')
    parser.add_argument('--output_dir', type=str, default='.', help='输出文件保存目录')
    parser.add_argument('--visualize', action='store_true', help='是否可视化点云')
    
    args = parser.parse_args()
    
    # 检查是否进行批处理
    if args.batch_dir:
        # 批处理模式
        if not os.path.exists(args.batch_dir):
            print(f"错误：批处理目录 {args.batch_dir} 不存在")
            return
        
        process_batch(args.batch_dir, args.output_dir, args.visualize)
    else:
        # 单文件处理模式
        if not args.cif_path:
            parser.print_help()
            print("\n错误：请提供CIF文件路径或使用--batch_dir参数指定批处理目录")
            return
            
        if not os.path.exists(args.cif_path):
            print(f"错误：文件 {args.cif_path} 不存在")
            return
        
        # 确保输出目录存在
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print(f"创建输出目录: {args.output_dir}")
        
        # 处理单个文件
        process_single_file(args.cif_path, args.output_dir, args.visualize)
        
        print("处理完成!")

if __name__ == '__main__':
    main()
