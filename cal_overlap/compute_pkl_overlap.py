#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def load_pkl(pkl_path):
    """
    加载 pkl 文件，并提取源点云、目标点云、变换矩阵和归一化信息
    
    参数:
        pkl_path: pkl 文件路径
    
    返回:
        source_points: 源点云
        target_points: 目标点云
        rotation: 旋转矩阵 (如果存在)
        translation: 平移向量 (如果存在)
        normalization_info: 归一化信息字典，包含缩放因子和中心点 (如果存在)
    """
    print(f"正在加载 pkl 文件: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 打印 pkl 文件的键，便于调试
    print(f"pkl 文件包含以下键: {list(data.keys())}")
    
    # 提取源点云、目标点云、变换矩阵和归一化信息
    rotation = None
    translation = None
    normalization_info = {}
    normalization_info['is_normalized'] = False
    
    # 提取源点云和目标点云
    if 'source_points' in data and 'target_points' in data:
        source_points = data['source_points']
        target_points = data['target_points']
    else:
        # 尝试其他可能的键名
        keys = list(data.keys())
        potential_source_keys = [k for k in keys if 'source' in k.lower() or 'src' in k.lower()]
        potential_target_keys = [k for k in keys if 'target' in k.lower() or 'tgt' in k.lower()]
        
        if potential_source_keys and potential_target_keys:
            source_points = data[potential_source_keys[0]]
            target_points = data[potential_target_keys[0]]
        else:
            raise ValueError(f"无法从 pkl 文件中提取源点云和目标点云，可用的键: {keys}")
    
    # 提取变换矩阵
    if 'rotation_matrix' in data:
        rotation = data['rotation_matrix']
    elif 'rotation' in data:
        rotation = data['rotation']
    
    if 'translation_vector' in data:
        translation = data['translation_vector']
    elif 'translation' in data:
        translation = data['translation']
    
    # 确保点云是 [N, 3] 格式
    if source_points.shape[0] == 3 and source_points.shape[1] != 3:
        source_points = source_points.T
    if target_points.shape[0] == 3 and target_points.shape[1] != 3:
        target_points = target_points.T
    
    print(f"源点云形状: {source_points.shape}")
    print(f"目标点云形状: {target_points.shape}")
    if rotation is not None:
        print(f"旋转矩阵形状: {rotation.shape}")
    if translation is not None:
        print(f"平移向量形状: {translation.shape}")
    
    # 检查并提取归一化信息
    normalization_info['is_normalized'] = False
    normalization_info['scale_factor'] = 1.0  # 默认值，不进行缩放
    normalization_info['centroid'] = None
    
    # 检查并打印 data['grid_info'] 的内容以便调试
    if 'grid_info' in data:
        print(f"grid_info的类型: {type(data['grid_info'])}")
        if isinstance(data['grid_info'], dict):
            print(f"grid_info内容: {data['grid_info']}")
            
            # 直接从 grid_info 中提取 scale_factor 和 original_centroid
            if 'scale_factor' in data['grid_info']:
                normalization_info['scale_factor'] = data['grid_info']['scale_factor']
                normalization_info['is_normalized'] = True
                print(f"从 grid_info 中提取到缩放因子: {normalization_info['scale_factor']}")
            
            if 'original_centroid' in data['grid_info']:
                # 可能是列表格式，转换为 numpy 数组
                centroid = data['grid_info']['original_centroid']
                if isinstance(centroid, list):
                    centroid = np.array(centroid, dtype=np.float32)
                normalization_info['centroid'] = centroid
                print(f"从 grid_info 中提取到质心点: {normalization_info['centroid']}")
            
            # 检查 normalized 标志
            if 'normalized' in data['grid_info']:
                normalization_info['is_normalized'] = data['grid_info']['normalized']
                print(f"从 grid_info 中检测到归一化标志: {normalization_info['is_normalized']}")
        else:
            print(f"grid_info值: {data['grid_info']}")
    else:
        print("pkl文件中没有grid_info字段")
    
    # 如果没有从 grid_info 中获取到，则检查并提取其他位置的归一化信息
    if not normalization_info['is_normalized']:
        if 'scale_factor' in data:
            normalization_info['scale_factor'] = data['scale_factor']
            normalization_info['is_normalized'] = True
            print(f"从数据根目录提取到缩放因子: {normalization_info['scale_factor']}")
        
        if 'centroid' in data:
            normalization_info['centroid'] = data['centroid']
            print(f"从数据根目录检测到质心点: {normalization_info['centroid']}")
    
    # 打印最终使用的归一化信息
    print(f"最终使用的归一化状态: {normalization_info['is_normalized']}")
    if normalization_info['is_normalized']:
        print(f"最终使用的缩放因子: {normalization_info['scale_factor']}")
        if normalization_info['centroid'] is not None:
            print(f"最终使用的质心点: {normalization_info['centroid']}")
    else:
        print("未检测到归一化信息，将使用原始点云")
    
    return source_points, target_points, rotation, translation, normalization_info

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
    # 将numpy数组或PyTorch张量转换为Open3D点云对象
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
    
    # 计算重叠率
    overlap_ratio_src = np.sum(has_corr_src) / len(src_np)
    overlap_ratio_tgt = np.sum(has_corr_tgt) / len(tgt_np)
    
    return has_corr_src, has_corr_tgt, src_tgt_corr, overlap_ratio_src, overlap_ratio_tgt

def visualize_overlap(source_points, target_points, has_corr_src, has_corr_tgt):
    """
    Visualize source and target point clouds, and their overlapping regions
    
    Parameters:
        source_points: Source point cloud array with shape [N, 3]
        target_points: Target point cloud array with shape [M, 3]
        has_corr_src: Boolean array indicating whether each source point is in the overlap region
        has_corr_tgt: Boolean array indicating whether each target point is in the overlap region
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw source point cloud
    src_overlap = source_points[has_corr_src]  # Overlap region
    src_non_overlap = source_points[~has_corr_src]  # Non-overlap region
    
    # Draw target point cloud
    tgt_overlap = target_points[has_corr_tgt]  # Overlap region
    tgt_non_overlap = target_points[~has_corr_tgt]  # Non-overlap region
    
    # Use different colors to distinguish overlap and non-overlap regions
    ax.scatter(src_overlap[:, 0], src_overlap[:, 1], src_overlap[:, 2], c='red', marker='.', s=10, label='Source Cloud Overlap')
    ax.scatter(src_non_overlap[:, 0], src_non_overlap[:, 1], src_non_overlap[:, 2], c='blue', marker='.', s=5, label='Source Cloud Non-overlap')
    ax.scatter(tgt_overlap[:, 0], tgt_overlap[:, 1], tgt_overlap[:, 2], c='green', marker='.', s=10, label='Target Cloud Overlap')
    ax.scatter(tgt_non_overlap[:, 0], tgt_non_overlap[:, 1], tgt_non_overlap[:, 2], c='purple', marker='.', s=5, label='Target Cloud Non-overlap')
    
    # Add legend and title
    ax.legend()
    ax.set_title('Point Cloud Overlap Visualization')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    # Keep axis scale consistent
    max_range = np.array([
        max(source_points[:, 0].max(), target_points[:, 0].max()) - min(source_points[:, 0].min(), target_points[:, 0].min()),
        max(source_points[:, 1].max(), target_points[:, 1].max()) - min(source_points[:, 1].min(), target_points[:, 1].min()),
        max(source_points[:, 2].max(), target_points[:, 2].max()) - min(source_points[:, 2].min(), target_points[:, 2].min())
    ]).max() / 2.0
    
    mid_x = (source_points[:, 0].mean() + target_points[:, 0].mean()) / 2
    mid_y = (source_points[:, 1].mean() + target_points[:, 1].mean()) / 2
    mid_z = (source_points[:, 2].mean() + target_points[:, 2].mean()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

def transform_point_cloud(point_cloud, rotation, translation=None, inverse=False):
    """
    Transform point cloud using rotation matrix and translation vector
    
    Parameters:
        point_cloud: Input point cloud with shape [N, 3] or [3, N]
        rotation: Rotation matrix with shape [3, 3]
        translation: Translation vector with shape [3], default is None
        inverse: Whether to perform inverse transformation (restoration), default is False
        
    Returns:
        transformed_point_cloud: Transformed point cloud with the same shape as input
    """
    # Convert PyTorch tensors to numpy if needed
    is_torch_tensor = False
    if hasattr(point_cloud, 'numpy'):
        is_torch_tensor = True
        point_cloud_np = point_cloud.detach().cpu().numpy()
    else:
        point_cloud_np = point_cloud
        
    if hasattr(rotation, 'numpy'):
        rotation_np = rotation.detach().cpu().numpy()
    else:
        rotation_np = rotation
        
    if translation is not None and hasattr(translation, 'numpy'):
        translation_np = translation.detach().cpu().numpy()
    else:
        translation_np = translation
    
    # Ensure point cloud is in [N, 3] format
    transpose_needed = False
    if point_cloud_np.shape[1] == 3 and point_cloud_np.shape[0] != 3:
        pc_working = point_cloud_np.copy()
    elif point_cloud_np.shape[0] == 3 and point_cloud_np.shape[1] != 3:
        pc_working = point_cloud_np.T.copy()
        transpose_needed = True
    else:
        pc_working = point_cloud_np.copy()
    
    if inverse:
        # Perform inverse transformation (first inverse translation, then inverse rotation)
        if translation_np is not None:
            pc_working = pc_working - translation_np.reshape(1, 3)  # Inverse translation
        
        # Inverse rotation (transpose of rotation matrix is its inverse, as rotation matrix is orthogonal)
        pc_working = np.matmul(pc_working, rotation_np.T)  # Note: right multiply by transpose
    else:
        # Perform forward transformation (first rotate, then translate)
        pc_working = np.matmul(pc_working, rotation_np)  # Rotation
        
        if translation_np is not None:
            pc_working = pc_working + translation_np.reshape(1, 3)  # Translation
    
    # If needed, transpose back to original shape
    if transpose_needed:
        pc_working = pc_working.T
    
    # Convert back to PyTorch tensor if input was a tensor
    if is_torch_tensor:
        import torch
        pc_working = torch.from_numpy(pc_working).to(point_cloud.device)
        
    return pc_working

def denormalize_point_cloud(point_cloud, scale_factor, centroid=None):
    """
    将归一化的点云还原到原始尺寸
    
    参数:
        point_cloud: 输入点云，形状为[N, 3]或[3, N]
        scale_factor: 缩放因子
        centroid: 中心点，默认为None
        
    返回:
        denormalized_points: 还原后的点云，与输入点云形状相同
    """
    # 确保处理的是numpy数组
    is_torch_tensor = False
    if hasattr(point_cloud, 'numpy'):
        is_torch_tensor = True
        point_cloud_np = point_cloud.detach().cpu().numpy()
    else:
        point_cloud_np = point_cloud
    
    # 确保点云是[N, 3]格式
    transpose_needed = False
    if point_cloud_np.shape[1] == 3 and point_cloud_np.shape[0] != 3:
        pc_working = point_cloud_np.copy()
    elif point_cloud_np.shape[0] == 3 and point_cloud_np.shape[1] != 3:
        pc_working = point_cloud_np.T.copy()
        transpose_needed = True
    else:
        pc_working = point_cloud_np.copy()
    
    # 打印还原前点云的范围
    print(f"还原前点云的范围: [{np.min(pc_working)}, {np.max(pc_working)}]")
    
    # 应用缩放因子，还原原始尺寸
    # 注意：这里使用的是乘法，与cif_to_point_cloud.py中的normalize_point_cloud函数相对
    pc_working = pc_working * scale_factor
    
    # 打印缩放后点云的范围
    print(f"缩放后点云的范围: [{np.min(pc_working)}, {np.max(pc_working)}]")
    
    # 如果提供了中心点，则平移回原位置
    if centroid is not None:
        if hasattr(centroid, 'numpy'):
            centroid_np = centroid.detach().cpu().numpy()
        else:
            centroid_np = centroid
            
        # 确保 centroid 是一个大小为 3 的数组
        if isinstance(centroid_np, (list, tuple)) and len(centroid_np) == 3:
            centroid_np = np.array(centroid_np)
            
        print(f"添加质心偏移: {centroid_np}")
        pc_working = pc_working + centroid_np.reshape(1, 3)
        
        # 打印偏移后点云的范围
        print(f"偏移后点云的范围: [{np.min(pc_working)}, {np.max(pc_working)}]")
    
    # 如果需要，转置回原始形状
    if transpose_needed:
        pc_working = pc_working.T
    
    # 转换回PyTorch张量（如果输入是张量）
    if is_torch_tensor:
        import torch
        pc_working = torch.from_numpy(pc_working).to(point_cloud.device)
    
    return pc_working

def process_single_pkl(pkl_path, search_voxel_size=0.1, visualize=False, use_restored=True, restore_size=False):
    """
    处理单个 pkl 文件，计算源点云和目标点云之间的重叠率
    
    参数:
        pkl_path: pkl 文件路径
        search_voxel_size: 搜索半径，默认为 0.1
        visualize: 是否可视化重叠区域，默认为 False
        use_restored: 是否使用变换还原后的点云计算重叠率，默认为 True
        restore_size: 是否还原点云到原始尺寸，默认为 False
    
    返回:
        overlap_ratio_src: 源点云的重叠率
        overlap_ratio_tgt: 目标点云的重叠率
    """
    try:
        # 加载 pkl 文件
        source_points, target_points, rotation, translation, normalization_info = load_pkl(pkl_path)
        
        # 检查是否需要还原点云尺寸
        if restore_size and normalization_info['is_normalized']:
            print("还原点云到原始尺寸...")
            scale_factor = normalization_info['scale_factor']
            centroid = normalization_info.get('centroid')
            
            # 还原源点云和目标点云到原始尺寸
            source_points_orig = denormalize_point_cloud(source_points, scale_factor, centroid)
            target_points_orig = denormalize_point_cloud(target_points, scale_factor, centroid)
            
            print(f"还原前源点云范围: [{source_points.min()}, {source_points.max()}]")
            print(f"还原后源点云范围: [{source_points_orig.min()}, {source_points_orig.max()}]")
            
            # 如果同时需要还原尺寸和应用变换，先应用变换再还原尺寸
            if use_restored and rotation is not None:
                # 应用变换还原
                print("使用变换还原的点云并还原到原始尺寸计算重叠率...")
                restored_target = transform_point_cloud(target_points, rotation, translation, inverse=True)
                # 然后还原尺寸
                restored_target_orig = denormalize_point_cloud(restored_target, scale_factor, centroid)
                
                # 保持搜索半径不变
                print(f"使用原始搜索半径: {search_voxel_size}")
                
                # 计算重叠区域
                has_corr_src, has_corr_tgt, src_tgt_corr, overlap_ratio_src, overlap_ratio_tgt = compute_overlap(
                    source_points_orig, restored_target_orig, search_voxel_size
                )
                
                print("\n原始尺寸下的变换还原点云重叠率:")
            else:
                # 只还原尺寸，不应用变换
                print("使用原始尺寸的点云计算重叠率...")
                
                # 保持搜索半径不变
                print(f"使用原始搜索半径: {search_voxel_size}")
                
                has_corr_src, has_corr_tgt, src_tgt_corr, overlap_ratio_src, overlap_ratio_tgt = compute_overlap(
                    source_points_orig, target_points_orig, search_voxel_size
                )
                
                print("\n原始尺寸点云的重叠率:")
        else:
            # 不还原尺寸，只处理变换
            if use_restored and rotation is not None:
                print("使用变换还原后的点云计算重叠率...")
                
                # 还原目标点云（将目标点云变换回源点云坐标系）
                restored_target = transform_point_cloud(target_points, rotation, translation, inverse=True)
                
                print(f"还原后的目标点云形状: {restored_target.shape}")
                
                # 计算还原后的重叠区域
                has_corr_src, has_corr_tgt, src_tgt_corr, overlap_ratio_src, overlap_ratio_tgt = compute_overlap(
                    source_points, restored_target, search_voxel_size
                )
                
                print("\n原始点云和变换还原点云的重叠率:")
            else:
                print("使用原始点云计算重叠率...")
                # 计算原始的重叠区域
                has_corr_src, has_corr_tgt, src_tgt_corr, overlap_ratio_src, overlap_ratio_tgt = compute_overlap(
                    source_points, target_points, search_voxel_size
                )
                
                print("\n原始点云的重叠率:")
        
        print(f"源点云重叠率: {overlap_ratio_src:.4f}")
        print(f"目标点云重叠率: {overlap_ratio_tgt:.4f}")
        print(f"源点云重叠点数: {np.sum(has_corr_src)} / {len(source_points)}")
        print(f"目标点云重叠点数: {np.sum(has_corr_tgt)} / {len(target_points)}")
        print(f"互相匹配的点对数: {src_tgt_corr.shape[1]}")
        
        # 可视化重叠区域
        if visualize:
            # 根据还原设置选择要可视化的点云
            if restore_size and normalization_info['is_normalized']:
                if use_restored and rotation is not None:
                    visualize_overlap(source_points_orig, restored_target_orig, has_corr_src, has_corr_tgt)
                else:
                    visualize_overlap(source_points_orig, target_points_orig, has_corr_src, has_corr_tgt)
            else:
                if use_restored and rotation is not None:
                    visualize_overlap(source_points, restored_target, has_corr_src, has_corr_tgt)
                else:
                    visualize_overlap(source_points, target_points, has_corr_src, has_corr_tgt)
        
        return overlap_ratio_src, overlap_ratio_tgt
        
    except Exception as e:
        print(f"处理 {pkl_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def process_batch(pkl_dir, search_voxel_size=0.1, output_csv=None, visualize=False, use_restored=True, restore_size=False):
    """
    批量处理目录中的所有 pkl 文件，计算源点云和目标点云之间的重叠率，并输出到 CSV 文件
    
    参数:
        pkl_dir: pkl 文件目录
        search_voxel_size: 搜索半径，默认为 0.1
        output_csv: 输出 CSV 文件路径，默认为 None (不输出到文件)
        visualize: 是否可视化第一个文件的重叠区域，默认为 False
        use_restored: 是否使用变换还原后的点云计算重叠率，默认为 True
        restore_size: 是否还原点云到原始尺寸，默认为 False
    """
    import csv
    
    # 查找所有 pkl 文件
    pkl_files = []
    for root, _, files in os.walk(pkl_dir):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    
    print(f"找到 {len(pkl_files)} 个 pkl 文件")
    
    # 存储结果
    results = []
    
    # 处理每个 pkl 文件
    for i, pkl_path in enumerate(tqdm(pkl_files, desc="处理 pkl 文件")):
        print(f"\n处理第 {i+1}/{len(pkl_files)} 个文件: {pkl_path}")
        
        # 对第一个文件进行可视化
        vis = visualize and i == 0
        
        overlap_src, overlap_tgt = process_single_pkl(pkl_path, search_voxel_size, vis, use_restored, restore_size)
        
        if overlap_src is not None and overlap_tgt is not None:
            results.append({
                'pkl_file': os.path.basename(pkl_path),
                'source_overlap': overlap_src,
                'target_overlap': overlap_tgt,
                'low_overlap': 1 if (overlap_src < 0.3 or overlap_tgt < 0.3) else 0
            })
    
    # 输出统计信息
    if results:
        src_overlaps = [r['source_overlap'] for r in results]
        tgt_overlaps = [r['target_overlap'] for r in results]
        
        print("\n统计信息:")
        print(f"源点云平均重叠率: {np.mean(src_overlaps):.4f}")
        print(f"目标点云平均重叠率: {np.mean(tgt_overlaps):.4f}")
        print(f"低重叠率文件数量: {sum(r['low_overlap'] for r in results)} / {len(results)}")
        
        # 将结果写入 CSV 文件
        if output_csv:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['pkl_file', 'source_overlap', 'target_overlap', 'low_overlap'])
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\n结果已保存到 {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='计算 pkl 文件中源点云和目标点云之间的重叠率')
    parser.add_argument('--pkl_path', type=str, help='单个 pkl 文件路径')
    parser.add_argument('--pkl_dir', type=str, help='pkl 文件目录，用于批处理')
    parser.add_argument('--search_radius', type=float, default=0.1, help='搜索半径，默认为 0.1')
    parser.add_argument('--output_csv', type=str, help='输出 CSV 文件路径')
    parser.add_argument('--visualize', action='store_true', help='是否可视化重叠区域')
    parser.add_argument('--original', action='store_true', help='使用原始点云计算重叠率，而非变换还原后的点云')
    parser.add_argument('--restore_size', action='store_true', help='还原点云到原始尺寸（取消归一化）')
    
    args = parser.parse_args()
    
    # 检查输入参数
    if args.pkl_path:
        # 单个文件处理
        if not os.path.exists(args.pkl_path):
            print(f"错误: 文件 {args.pkl_path} 不存在")
            return
        
        process_single_pkl(args.pkl_path, args.search_radius, args.visualize, not args.original, args.restore_size)
        
    elif args.pkl_dir:
        # 批处理
        if not os.path.exists(args.pkl_dir):
            print(f"错误: 目录 {args.pkl_dir} 不存在")
            return
        
        process_batch(args.pkl_dir, args.search_radius, args.output_csv, args.visualize, not args.original, args.restore_size)
        
    else:
        parser.print_help()
        print("\n错误: 请提供 pkl 文件路径或目录")

if __name__ == '__main__':
    main()
