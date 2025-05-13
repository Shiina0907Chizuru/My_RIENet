#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle
import open3d as o3d

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
    point_cloud = np.array(atoms, dtype=np.float32)
    
    return point_cloud, atom_types, atom_labels, cif_content, atom_lines

def compute_overlap(src, tgt, search_voxel_size):
    """
    计算两个点云之间的重叠率，使用互为最近邻的策略
    
    参数:
        src: 源点云，形状为[N, 3]
        tgt: 目标点云，形状为[M, 3]
        search_voxel_size: 搜索半径，用于确定重叠
        
    返回:
        has_corr_src: 源点云中有对应点的布尔掩码
        has_corr_tgt: 目标点云中有对应点的布尔掩码
        src_tgt_corr: 源点云和目标点云之间的对应关系
        overlap_ratio_src: 源点云的重叠比例
        overlap_ratio_tgt: 目标点云的重叠比例
    """
    # 将numpy数组转换为Open3D点云对象
    if isinstance(src, torch.Tensor):
        src_np = src.detach().cpu().numpy()
    else:
        src_np = src
    
    if isinstance(tgt, torch.Tensor):
        tgt_np = tgt.detach().cpu().numpy()
    else:
        tgt_np = tgt
    
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_np)
    
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_np)
    
    # 第一步：检查目标点云中的点在源点云中是否有对应点
    tgt_corr = np.full(tgt_np.shape[0], -1, dtype=int)
    src_tree = o3d.geometry.KDTreeFlann(src_pcd)
    for i, t in enumerate(tgt_np):
        # 在源点云中搜索指定半径内的点
        [k, idx, _] = src_tree.search_radius_vector_3d(t, search_voxel_size)
        if k > 0:
            # 取第一个最近的点作为对应点
            tgt_corr[i] = idx[0]
    
    # 第二步：检查源点云中的点在目标点云中是否有对应点
    src_corr = np.full(src_np.shape[0], -1, dtype=int)
    tgt_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    for i, s in enumerate(src_np):
        # 在目标点云中搜索指定半径内的点
        [k, idx, _] = tgt_tree.search_radius_vector_3d(s, search_voxel_size)
        if k > 0:
            # 取第一个最近的点作为对应点
            src_corr[i] = idx[0]
    
    # 第三步：计算互为最近邻的对应点
    src_indices = np.arange(len(src_corr))
    valid_src = src_corr >= 0
    # 互为最近邻条件：tgt_corr[src_corr[i]] == i
    mutual = np.zeros(len(src_corr), dtype=bool)
    mutual[valid_src] = (tgt_corr[src_corr[valid_src]] == src_indices[valid_src])
    
    # 根据互为最近邻关系生成对应关系矩阵
    src_tgt_corr = np.stack([np.nonzero(mutual)[0], src_corr[mutual]])
    
    # 确定重叠区域内的点
    has_corr_src = src_corr >= 0  # 源点云中有对应点的标记
    has_corr_tgt = tgt_corr >= 0  # 目标点云中有对应点的标记
    
    # 计算重叠比例
    overlap_ratio_src = np.sum(has_corr_src) / len(src_np)
    overlap_ratio_tgt = np.sum(has_corr_tgt) / len(tgt_np)
    
    return has_corr_src, has_corr_tgt, src_tgt_corr, overlap_ratio_src, overlap_ratio_tgt

def visualize_overlap(src_points, tgt_points, has_corr_src, has_corr_tgt):
    """
    可视化两个点云之间的重叠部分
    
    参数:
        src_points: 源点云，形状为[N, 3]
        tgt_points: 目标点云，形状为[M, 3]
        has_corr_src: 源点云中有对应点的布尔掩码
        has_corr_tgt: 目标点云中有对应点的布尔掩码
    """
    # 将PyTorch张量转换为numpy数组
    if isinstance(src_points, torch.Tensor):
        src_points = src_points.detach().cpu().numpy()
    if isinstance(tgt_points, torch.Tensor):
        tgt_points = tgt_points.detach().cpu().numpy()
    
    # 确保点云是[N, 3]格式
    if src_points.shape[1] != 3 and src_points.shape[0] == 3:
        src_points = src_points.T
    if tgt_points.shape[1] != 3 and tgt_points.shape[0] == 3:
        tgt_points = tgt_points.T
    
    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制源点云重叠部分
    ax.scatter(src_points[has_corr_src, 0], src_points[has_corr_src, 1], src_points[has_corr_src, 2], 
              c='r', marker='o', s=5, label='Source Cloud Overlap')
    
    # 绘制源点云非重叠部分
    ax.scatter(src_points[~has_corr_src, 0], src_points[~has_corr_src, 1], src_points[~has_corr_src, 2], 
              c='darkred', marker='o', s=5, label='Source Cloud Non-overlap')
    
    # 绘制目标点云重叠部分
    ax.scatter(tgt_points[has_corr_tgt, 0], tgt_points[has_corr_tgt, 1], tgt_points[has_corr_tgt, 2], 
              c='g', marker='o', s=5, label='Target Cloud Overlap')
    
    # 绘制目标点云非重叠部分
    ax.scatter(tgt_points[~has_corr_tgt, 0], tgt_points[~has_corr_tgt, 1], tgt_points[~has_corr_tgt, 2], 
              c='darkgreen', marker='o', s=5, label='Target Cloud Non-overlap')
    
    # 设置图例、标题和轴标签
    ax.set_title('Point Cloud Overlap Visualization')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.legend()
    
    # 设置坐标轴比例相等
    max_range = np.array([
        src_points[:,0].max() - src_points[:,0].min(), 
        src_points[:,1].max() - src_points[:,1].min(), 
        src_points[:,2].max() - src_points[:,2].min(),
        tgt_points[:,0].max() - tgt_points[:,0].min(), 
        tgt_points[:,1].max() - tgt_points[:,1].min(), 
        tgt_points[:,2].max() - tgt_points[:,2].min()
    ]).max() / 2.0
    
    mid_x = (src_points[:,0].mean() + tgt_points[:,0].mean()) / 2
    mid_y = (src_points[:,1].mean() + tgt_points[:,1].mean()) / 2
    mid_z = (src_points[:,2].mean() + tgt_points[:,2].mean()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 显示图形
    plt.tight_layout()
    plt.show()

def process_cif_pair(cif_path1, cif_path2, search_radius=0.1, visualize=False):
    """
    处理一对CIF文件，计算它们之间的重叠率
    
    参数:
        cif_path1: 第一个CIF文件路径
        cif_path2: 第二个CIF文件路径
        search_radius: 搜索半径，默认为0.1
        visualize: 是否可视化重叠区域，默认为False
    
    返回:
        overlap_ratio_1: 第一个点云的重叠比例
        overlap_ratio_2: 第二个点云的重叠比例
    """
    try:
        # 解析CIF文件
        point_cloud1, atom_types1, atom_labels1, _, _ = parse_cif(cif_path1)
        point_cloud2, atom_types2, atom_labels2, _, _ = parse_cif(cif_path2)
        
        print(f"第一个点云形状: {point_cloud1.shape}")
        print(f"第二个点云形状: {point_cloud2.shape}")
        
        # 计算重叠率
        has_corr_1, has_corr_2, corr_1_2, overlap_ratio_1, overlap_ratio_2 = compute_overlap(
            point_cloud1, point_cloud2, search_radius
        )
        
        print("\n点云重叠率:")
        print(f"第一个点云重叠率: {overlap_ratio_1:.4f}")
        print(f"第二个点云重叠率: {overlap_ratio_2:.4f}")
        print(f"第一个点云重叠点数: {np.sum(has_corr_1)} / {len(point_cloud1)}")
        print(f"第二个点云重叠点数: {np.sum(has_corr_2)} / {len(point_cloud2)}")
        print(f"互相匹配的点对数: {corr_1_2.shape[1]}")
        
        # 可视化重叠区域
        if visualize:
            visualize_overlap(point_cloud1, point_cloud2, has_corr_1, has_corr_2)
        
        return overlap_ratio_1, overlap_ratio_2
        
    except Exception as e:
        print(f"处理CIF文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def get_prefix(filename):
    """从文件名中提取前缀，如 '7boz_mrc2cif.cif' 返回 '7boz'、'7wug_point.cif' 返回 '7wug'"""
    # 去除路径，只保留文件名
    basename = os.path.basename(filename)
    # 对于 mrc2cif.cif 类型
    if '_mrc2cif.cif' in basename:
        return basename.split('_mrc2cif.cif')[0]
    # 对于 point.cif 类型
    elif '_point.cif' in basename:
        return basename.split('_point.cif')[0]
    # 其他情况，取第一个下划线前的部分
    else:
        parts = basename.split('_')
        if len(parts) > 0:
            return parts[0]
        else:
            # 缺省取去掉扩展名的部分
            return os.path.splitext(basename)[0]

def process_batch(cif1_dir, cif2_dir, search_radius=3.8, output_csv=None, visualize=False, visualize_first=False):
    """
    批量处理两个目录中的CIF文件对，根据文件名前缀匹配
    
    参数:
        cif1_dir: 第一个CIF文件目录
        cif2_dir: 第二个CIF文件目录
        search_radius: 搜索半径，默认为3.8
        output_csv: 输出结果的CSV文件路径，默认为None
        visualize: 是否可视化所有文件对的重叠区域
        visualize_first: 是否仅可视化第一个文件对的重叠区域
    """
    import csv
    from tqdm import tqdm
    
    # 获取两个目录中的所有CIF文件
    cif1_files = [f for f in os.listdir(cif1_dir) if f.lower().endswith('.cif')]
    cif2_files = [f for f in os.listdir(cif2_dir) if f.lower().endswith('.cif')]
    
    print(f"目录 {cif1_dir} 中找到 {len(cif1_files)} 个CIF文件")
    print(f"目录 {cif2_dir} 中找到 {len(cif2_files)} 个CIF文件")
    
    # 提取文件前缀和完整路径的映射
    cif1_prefixes = {get_prefix(f): os.path.join(cif1_dir, f) for f in cif1_files}
    cif2_prefixes = {get_prefix(f): os.path.join(cif2_dir, f) for f in cif2_files}
    
    # 找到共有的前缀
    common_prefixes = set(cif1_prefixes.keys()) & set(cif2_prefixes.keys())
    print(f"找到 {len(common_prefixes)} 个匹配的文件对")
    
    if not common_prefixes:
        print("没有找到匹配的文件对")
        return
    
    # 存储结果
    results = []
    
    # 处理每个匹配的文件对
    for i, prefix in enumerate(tqdm(common_prefixes, desc="处理文件对")):
        cif1_path = cif1_prefixes[prefix]
        cif2_path = cif2_prefixes[prefix]
        
        print(f"\n处理第 {i+1}/{len(common_prefixes)} 个文件对: {prefix}")
        print(f"CIF1: {cif1_path}")
        print(f"CIF2: {cif2_path}")
        
        # 决定是否可视化
        vis = (visualize and not visualize_first) or (visualize_first and i == 0)
        
        # 处理文件对
        overlap_1, overlap_2 = process_cif_pair(cif1_path, cif2_path, search_radius, vis)
        
        if overlap_1 is not None and overlap_2 is not None:
            results.append({
                'prefix': prefix,
                'cif1_file': os.path.basename(cif1_path),
                'cif2_file': os.path.basename(cif2_path),
                'cif1_overlap': overlap_1,
                'cif2_overlap': overlap_2,
                'low_overlap': 1 if (overlap_1 < 0.3 or overlap_2 < 0.3) else 0
            })
    
    # 输出统计信息
    if results:
        cif1_overlaps = [r['cif1_overlap'] for r in results]
        cif2_overlaps = [r['cif2_overlap'] for r in results]
        
        print("\n统计信息:")
        print(f"CIF1平均重叠率: {np.mean(cif1_overlaps):.4f}")
        print(f"CIF2平均重叠率: {np.mean(cif2_overlaps):.4f}")
        print(f"低重叠率文件对数量: {sum(r['low_overlap'] for r in results)} / {len(results)}")
        
        # 将结果写入CSV文件
        if output_csv:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['prefix', 'cif1_file', 'cif2_file', 'cif1_overlap', 'cif2_overlap', 'low_overlap'])
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\n结果已保存到 {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='计算CIF文件之间的点云重叠率')
    parser.add_argument('--cif1', type=str, help='第一个CIF文件路径')
    parser.add_argument('--cif2', type=str, help='第二个CIF文件路径')
    parser.add_argument('--cif1_dir', type=str, help='第一个CIF文件目录，用于批处理')
    parser.add_argument('--cif2_dir', type=str, help='第二个CIF文件目录，用于批处理')
    parser.add_argument('--search_radius', type=float, default=3.8, help='搜索半径，默认为3.8')
    parser.add_argument('--output_csv', type=str, help='输出 CSV 文件路径')
    parser.add_argument('--visualize', action='store_true', help='是否可视化重叠区域')
    parser.add_argument('--visualize_first', action='store_true', help='只可视化第一个匹配的文件对')
    
    args = parser.parse_args()
    
    # 单文件处理和批处理模式
    if args.cif1 and args.cif2:
        # 单文件模式
        if not os.path.exists(args.cif1):
            print(f"错误: 文件 {args.cif1} 不存在")
            return
        
        if not os.path.exists(args.cif2):
            print(f"错误: 文件 {args.cif2} 不存在")
            return
        
        # 处理CIF文件对
        process_cif_pair(args.cif1, args.cif2, args.search_radius, args.visualize)
    
    elif args.cif1_dir and args.cif2_dir:
        # 批处理模式
        if not os.path.exists(args.cif1_dir):
            print(f"错误: 目录 {args.cif1_dir} 不存在")
            return
        
        if not os.path.exists(args.cif2_dir):
            print(f"错误: 目录 {args.cif2_dir} 不存在")
            return
        
        # 批量处理目录
        process_batch(
            args.cif1_dir, 
            args.cif2_dir, 
            args.search_radius, 
            args.output_csv, 
            args.visualize, 
            args.visualize_first
        )
    else:
        parser.print_help()
        print("\n错误: 请提供 CIF 文件路径或目录")

if __name__ == '__main__':
    main()
