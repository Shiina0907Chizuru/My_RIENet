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

def transform_point_cloud_numpy(point_cloud, rotation, translation, rotation_only=False):
    """使用numpy将点云进行变换 - 使用右乘方式：点云 = 旋转矩阵 * 点云 + 平移向量"""
    # 确保点云是[3, N]格式
    if point_cloud.shape[1] == 3 and point_cloud.shape[0] != 3:
        print(f"转置点云: 从{point_cloud.shape}到", end="")
        point_cloud = point_cloud.T
        print(f"{point_cloud.shape}")
    
    # 现在point_cloud已经是[3, N]格式，可以直接应用旋转和平移
    # R[3,3] * point_cloud[3,N] + t[3,1] = transformed_point_cloud[3,N]
    if rotation_only:
        # 只应用旋转，不应用平移
        transformed_point_cloud = np.matmul(rotation, point_cloud)
        print("应用旋转变换（无平移）")
    else:
        # 应用旋转和平移
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
    
    # 转换为torch张量
    source_points_tensor = torch.from_numpy(source_points).float()
    target_points_tensor = torch.from_numpy(target_points).float()
    rotation_tensor = torch.from_numpy(rotation).float()
    translation_tensor = torch.from_numpy(translation).float()
    
    # # Pytorch tensor 形状为 [Batch, Channel, Num]
    # if source_points_tensor.shape[0] != 1:
    #     source_points_tensor = source_points_tensor.unsqueeze(0)
    # if target_points_tensor.shape[0] != 1:
    #     target_points_tensor = target_points_tensor.unsqueeze(0)
    
    print(f"保存为torch张量后源点云形状: {source_points_tensor.shape}")
    print(f"保存为torch张量后目标点云形状: {target_points_tensor.shape}")
    
    # 保存为字典格式
    data_dict = {
        'source': source_points_tensor,  # 源点云
        'target': target_points_tensor,  # 目标点云
        'rotation': rotation_tensor,   # 旋转矩阵
        'translation': translation_tensor,  # 平移向量
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

def calculate_centroid_aligned_transformation(original_rotation, original_translation, centroid_diff):
    """根据原始变换和质心偏差计算质心对齐的变换矩阵
    
    Args:
        original_rotation: 原始旋转矩阵 [3, 3]
        original_translation: 原始平移向量 [3]
        centroid_diff: 质心偏差向量 [3]
        
    Returns:
        新的平移向量，使得变换后的点云质心与源点云质心对齐
    """
    # 旋转矩阵保持不变
    new_rotation = original_rotation.copy()
    
    # 计算新的平移向量：原平移向量 + 质心偏差
    new_translation = original_translation + centroid_diff
    
    return new_rotation, new_translation

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

def save_as_cif_with_suffix(coordinates, atom_types, atom_labels, original_cif, atom_lines, output_path, suffix="_c"):
    """将点云坐标保存为CIF格式，并添加自定义后缀"""
    # 添加后缀到输出文件名
    base_name, ext = os.path.splitext(output_path)
    new_output_path = f"{base_name}{suffix}{ext}"
    
    # 复用现有函数逻辑
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
    header_comment = f"# This is a centroid-aligned transformed structure\n"
    new_cif.insert(0, header_comment)
    
    # 写入新的CIF文件
    with open(new_output_path, 'w') as f:
        f.writelines(new_cif)
    
    print(f"已保存质心对齐点云为CIF格式: {new_output_path}")
    
    return new_output_path

def calculate_centroid(point_cloud):
    """计算点云的质心"""
    # 确保点云是[N, 3]格式用于计算质心
    if point_cloud.shape[0] == 3 and point_cloud.shape[1] != 3:
        point_cloud = point_cloud.T
    
    # 计算质心
    centroid = np.mean(point_cloud, axis=0)
    return centroid

def generate_centroid_aligned_cloud(source_cloud, target_cloud, rotation):
    """生成质心对齐的目标点云
    Args:
        source_cloud: 源点云，形状为[N, 3]或[3, N]
        target_cloud: 目标点云，形状为[N, 3]或[3, N]
        rotation: 旋转矩阵，形状为[3, 3]
    Returns:
        aligned_cloud: 质心对齐的目标点云，与输入格式相同
    """
    # 确保点云是[N, 3]格式用于计算质心
    source_format_is_3N = False
    target_format_is_3N = False
    
    if source_cloud.shape[0] == 3 and source_cloud.shape[1] != 3:
        source_format_is_3N = True
        source_cloud_for_centroid = source_cloud.T
    else:
        source_cloud_for_centroid = source_cloud
    
    if target_cloud.shape[0] == 3 and target_cloud.shape[1] != 3:
        target_format_is_3N = True
        target_cloud_for_centroid = target_cloud.T
    else:
        target_cloud_for_centroid = target_cloud
    
    # 计算源点云和目标点云的质心
    source_centroid = np.mean(source_cloud_for_centroid, axis=0)
    target_centroid = np.mean(target_cloud_for_centroid, axis=0)
    
    # 计算质心偏差
    centroid_diff = source_centroid - target_centroid
    
    # 根据质心偏差计算新的平移向量
    # 注意：这里需要考虑旋转对质心的影响
    # 我们需要使旋转后的点云质心与源点云质心对齐
    
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
    
    return aligned_cloud, centroid_diff

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

def process_batch(batch_dir, output_dir, visualize=False, rotation_only=False):
    """批量处理指定目录下所有子文件夹中的xxxx_point.cif文件"""
    print(f"开始批处理目录: {batch_dir}")
    print(f"只旋转不平移: {rotation_only}")
    
    # 遍历所有子文件夹
    subdirs = [d for d in os.listdir(batch_dir) if os.path.isdir(os.path.join(batch_dir, d))]
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    total_files = 0
    processed_files = 0
    
    if not subdirs:
        print(f"在 {batch_dir} 中未找到子文件夹，将直接在该目录中搜索cif文件")
        
        # 直接在当前目录中查找所有 *_point.cif 文件
        cif_files = glob.glob(os.path.join(batch_dir, '*_point.cif'))
        total_files = len(cif_files)
        
        for cif_path in cif_files:
            try:
                # 为每个CIF文件创建单独的子目录
                base_name = os.path.splitext(os.path.basename(cif_path))[0]
                file_output_dir = os.path.join(output_dir, base_name)
                if not os.path.exists(file_output_dir):
                    os.makedirs(file_output_dir)
                    print(f"创建CIF文件对应的输出子目录: {file_output_dir}")
                
                process_single_file(cif_path, file_output_dir, visualize, rotation_only)
                processed_files += 1
            except Exception as e:
                print(f"处理文件 {cif_path} 时出错: {str(e)}")
    else:
        print(f"找到 {len(subdirs)} 个子文件夹")
        
        # 处理每个子文件夹中的 xxxx_point.cif 文件
        for subdir in subdirs:
            subdir_path = os.path.join(batch_dir, subdir)
            subdir_output = os.path.join(output_dir, subdir)
            
            # 确保子文件夹对应的输出目录存在
            if not os.path.exists(subdir_output):
                os.makedirs(subdir_output)
            
            # 查找所有 *_point.cif 文件
            cif_files = glob.glob(os.path.join(subdir_path, '*_point.cif'))
            total_files += len(cif_files)
            
            for cif_path in cif_files:
                try:
                    # 为每个CIF文件创建单独的子目录
                    base_name = os.path.splitext(os.path.basename(cif_path))[0]
                    file_output_dir = os.path.join(subdir_output, base_name)
                    if not os.path.exists(file_output_dir):
                        os.makedirs(file_output_dir)
                        print(f"创建CIF文件对应的输出子目录: {file_output_dir}")
                    
                    process_single_file(cif_path, file_output_dir, visualize, rotation_only)
                    processed_files += 1
                except Exception as e:
                    print(f"处理文件 {cif_path} 时出错: {str(e)}")
    
    print(f"批处理完成. 成功处理 {processed_files}/{total_files} 个文件")

def process_single_file(cif_path, output_dir, visualize=False, rotation_only=False):
    """处理单个cif文件"""
    print(f"处理文件: {cif_path}")
    print(f"只旋转不平移: {rotation_only}")
    
    # 解析CIF文件
    point_cloud, atom_types, atom_labels, cif_content, atom_lines = parse_cif(cif_path)
    
    # 保证输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建输出文件名
    base_name = os.path.splitext(os.path.basename(cif_path))[0]
    output_cif = os.path.join(output_dir, base_name + "_target.cif")
    output_pkl = os.path.join(output_dir, base_name + "_train.pkl")
    output_centroid_pkl = os.path.join(output_dir, base_name + "_train_c.pkl")
    
    # 生成随机旋转矩阵和平移向量
    rotation_matrix = random_rotation_matrix()
    
    # 是否应用平移
    if rotation_only:
        print("只应用旋转，不应用平移")
        translation_vector = np.zeros(3, dtype=np.float32)  # 全零平移向量
    else:
        translation_vector = random_translation_vector()
    
    # 确保点云是[3, N]格式用于变换
    if point_cloud.shape[1] == 3 and point_cloud.shape[0] != 3:
        # 如果点云是[N, 3]格式，转换为[3, N]格式
        point_cloud_for_transform = point_cloud.T
    else:
        point_cloud_for_transform = point_cloud.copy()
    
    # 应用变换
    target_point_cloud = transform_point_cloud_numpy(
        point_cloud_for_transform, 
        rotation_matrix, 
        translation_vector,
        rotation_only
    )
    
    print("变换矩阵信息:")
    print("旋转矩阵:")
    print(rotation_matrix)
    print("平移向量:")
    print(translation_vector)
    
    # 为了符合PKL文件格式，创建网格信息
    grid_info = create_grid_info()
    
    # 创建并保存PKL文件
    create_pkl_file(
        point_cloud_for_transform, 
        target_point_cloud, 
        rotation_matrix, 
        translation_vector, 
        output_pkl, 
        grid_info
    )
    
    # 将目标点云转换回与源点云相同的格式以保存到CIF文件
    if point_cloud.shape[1] == 3:
        # 如果原始点云是[N, 3]格式，那么需要将目标点云从[3, N]转为[N, 3]
        target_point_cloud_for_cif = target_point_cloud.T
    else:
        target_point_cloud_for_cif = target_point_cloud
    
    # 保存目标结构为CIF格式
    save_as_cif(target_point_cloud_for_cif, atom_types, atom_labels, cif_content, atom_lines, output_cif)
    
    # 计算质心对齐的点云并保存
    source_centroid = calculate_centroid(point_cloud)
    target_centroid = calculate_centroid(target_point_cloud)
    
    print("源点云质心:")
    print(source_centroid)
    print("目标点云质心:")
    print(target_centroid)
    
    # 生成质心对齐的目标点云
    centroid_aligned_cloud, centroid_diff = generate_centroid_aligned_cloud(point_cloud, target_point_cloud, rotation_matrix)
    
    print("质心偏差向量:")
    print(centroid_diff)
    
    # 计算质心对齐的变换矩阵
    centroid_aligned_rotation, centroid_aligned_translation = calculate_centroid_aligned_transformation(
        rotation_matrix, translation_vector, centroid_diff
    )
    
    print("质心对齐旋转矩阵:")
    print(centroid_aligned_rotation)
    print("质心对齐平移向量:")
    print(centroid_aligned_translation)
    
    # 保存质心对齐的目标点云为CIF格式
    if centroid_aligned_cloud.shape[0] == 3:
        # 确保转换为[N, 3]格式
        centroid_aligned_cloud_for_cif = centroid_aligned_cloud.T
    else:
        centroid_aligned_cloud_for_cif = centroid_aligned_cloud
    
    # 保存质心对齐的CIF文件
    save_as_cif_with_suffix(
        centroid_aligned_cloud_for_cif, 
        atom_types, 
        atom_labels, 
        cif_content, 
        atom_lines, 
        output_cif,
        suffix="_c"
    )
    
    # 创建并保存质心对齐的pkl文件
    create_pkl_file(
        point_cloud_for_transform, 
        centroid_aligned_cloud, 
        centroid_aligned_rotation, 
        centroid_aligned_translation, 
        output_centroid_pkl, 
        grid_info
    )
    
    # 使用质心对齐的变换矩阵再次变换源点云，用于验证
    print("使用质心对齐的变换矩阵再次变换源点云，生成验证文件...")
    
    # 应用质心对齐的变换矩阵到源点云
    fixed_point_cloud = transform_point_cloud_numpy(
        point_cloud_for_transform,
        centroid_aligned_rotation,
        centroid_aligned_translation
    )
    
    # 将固定点云转换为与源点云相同的格式
    if point_cloud.shape[1] == 3:
        fixed_point_cloud_for_cif = fixed_point_cloud.T
    else:
        fixed_point_cloud_for_cif = fixed_point_cloud
    
    # 保存为fix.cif文件
    output_fixed_cif = os.path.join(output_dir, base_name + "_fix.cif")
    save_as_cif(
        fixed_point_cloud_for_cif,
        atom_types,
        atom_labels,
        cif_content,
        atom_lines,
        output_fixed_cif
    )
    
    print(f"已保存验证用的固定点云文件: {output_fixed_cif}")
    
    # 计算验证点云的质心
    fixed_centroid = calculate_centroid(fixed_point_cloud)
    print("验证点云质心:")
    print(fixed_centroid)
    print("源点云质心:")
    print(source_centroid)
    print(f"质心差异: {np.linalg.norm(fixed_centroid - source_centroid)}")
    
    # 如果需要可视化，则同时可视化原始和质心对齐的点云
    if visualize:
        print("执行点云可视化...")
        visualize_point_clouds(point_cloud, target_point_cloud)
        
        # 额外可视化质心对齐的点云
        if centroid_aligned_cloud.shape[0] == 3:
            visualize_point_clouds(point_cloud.T, centroid_aligned_cloud, title="质心对齐点云可视化")
        else:
            visualize_point_clouds(point_cloud, centroid_aligned_cloud, title="质心对齐点云可视化")
    
    print(f"文件{cif_path}处理完成!")

def main():
    parser = argparse.ArgumentParser(description='CIF文件转换为点云数据并进行变换')
    parser.add_argument('cif_path', type=str, nargs='?', help='CIF文件路径')
    parser.add_argument('--batch_dir', type=str, help='批处理目录路径，指定该参数将进行批处理')
    parser.add_argument('--output_dir', type=str, default='.', help='输出文件保存目录')
    parser.add_argument('--visualize', action='store_true', help='是否可视化点云')
    parser.add_argument('--rotation_only', action='store_true', help='是否只应用旋转变换，不应用平移')
    
    args = parser.parse_args()
    
    # 检查是否进行批处理
    if args.batch_dir:
        # 批处理模式
        if not os.path.exists(args.batch_dir):
            print(f"错误：批处理目录 {args.batch_dir} 不存在")
            return
        
        process_batch(args.batch_dir, args.output_dir, args.visualize, args.rotation_only)
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
        process_single_file(args.cif_path, args.output_dir, args.visualize, args.rotation_only)
        
        print("处理完成!")

if __name__ == '__main__':
    main()
