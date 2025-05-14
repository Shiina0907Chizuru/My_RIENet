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
import time
import math  # 添加math模块以支持RRE计算

# 从pkldataprocess.py导入所需函数
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
        print(f"点云质心: {centroid}")
        print(f"点云缩放因子: {scale_factor}")
        print(f"归一化前大小: {np.max(np.abs(centered_points))}")
        print(f"归一化后大小: {np.max(np.abs(normalized_points))}")
    
    # 如果原始输入是[3, N]格式，则返回[3, N]格式
    if transpose_needed:
        normalized_points = normalized_points.T
    
    return normalized_points, scale_factor, centroid

def random_rotation_matrix():
    """生成随机旋转矩阵 - 与cif_to_point_cloud.py相同的方法"""
    # 使用QR分解生成随机旋转矩阵
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    
    # 确保是特殊正交矩阵(行列式为1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    
    # 验证正交性和行列式
    orthogonal_check = np.matmul(Q, Q.T)
    det_check = np.linalg.det(Q)
    print(f"旋转矩阵正交性检查 (R·R^T):\n{orthogonal_check}")
    print(f"旋转矩阵行列式: {det_check:.6f}")
    
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
    
    # 使用与cif_to_point_cloud.py相同的变换方式
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
    
    print("使用与cif_to_point_cloud.py一致的点云变换方法")
    print(f"旋转矩阵类型: {rotation.dtype}, 形状: {rotation.shape}")
    if translation is not None:
        print(f"平移向量类型: {translation.dtype}, 形状: {translation.shape}")
    
    return transformed_point_cloud

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
    # 根据normalize参数自动设置voxel_size和grid_shape
    if normalize:
        # 归一化情况下，点云范围是[-1, 1]，所以一个单位包含1个点
        if voxel_size is None:
            voxel_size = 2.0 / grid_shape  # 总长度2.0（-1到1）除以网格数量
        
        # 计算点云范围
        point_cloud_range = np.array([
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0]
        ], dtype=np.float32)
    else:
        # 非归一化情况下，我们需要合理的默认值
        if voxel_size is None:
            voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        # 计算点云范围
        half_size = grid_shape * voxel_size / 2
        point_cloud_range = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],
            [half_size[0], half_size[1], half_size[2]]
        ], dtype=np.float32)
    
    # 创建网格信息字典
    grid_info = {
        'grid_shape': grid_shape,
        'voxel_size': voxel_size,
        'point_cloud_range': point_cloud_range
    }
    
    return grid_info

def create_pkl_file(source_points, target_points, output_path, grid_info, rotation=None, translation=None, normalize=False, verbose=True, source_centroid=None, source_scale=None):
    """
    创建 pkl 文件，保存为 PointCloud_ca_Dataset 格式
    
    参数:
        source_points: 源点云，形状为[N, 3]或[3, N]
        target_points: 目标点云，形状为[N, 3]或[3, N]
        output_path: 输出 pkl 文件路径
        grid_info: 网格信息字典，包含grid_shape、voxel_size、point_cloud_range
        rotation: 旋转矩阵，形状为[3, 3]，默认为 None
        translation: 平移向量，形状为[3]，默认为 None
        normalize: 是否已经归一化，默认为 False
        verbose: 是否打印详细信息，默认为 True
        source_centroid: 源点云原始质心，形状为[3]，默认为 None
        source_scale: 源点云缩放因子，形状为标量，默认为 None
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 检查格式并转置（如果需要）
    if source_points.shape[0] == 3 and source_points.shape[1] != 3:
        source_points = source_points.T
    
    if target_points.shape[0] == 3 and target_points.shape[1] != 3:
        target_points = target_points.T
    
    # 转换为PyTorch的FloatTensor
    source_points_tensor = torch.from_numpy(source_points.astype(np.float32))
    target_points_tensor = torch.from_numpy(target_points.astype(np.float32))
    
    # 创建数据集字典 - 使用'source'和'target'键代替'points'以兼容predict_from_pkl.py
    data = {
        'source': source_points_tensor,  # 源点云
        'target': target_points_tensor,   # 目标点云
        'grid_shape': torch.from_numpy(grid_info['grid_shape']),
        'voxel_size': torch.from_numpy(grid_info['voxel_size']),
        'point_cloud_range': torch.from_numpy(grid_info['point_cloud_range']),
        'has_normals': False,  # 是否包含法线信息
        'has_features': False,  # 是否包含特征信息
        'is_normalized': normalize,  # 是否已经归一化
    }
    
    # 添加旋转矩阵和平移向量（如果提供）
    if rotation is not None:
        # 转置旋转矩阵以符合cif_to_point_cloud.py中的语义
        # 原始旋转矩阵的语义与cif_to_point_cloud.py相反，需要取逆矩阵(对正交矩阵，逆等于转置)
        rotation_to_save = rotation.T.astype(np.float32)
        data['rotation'] = torch.from_numpy(rotation_to_save)
        if verbose:
            print(f"原始旋转矩阵:\n{rotation}")
            print(f"保存(转置)后的旋转矩阵:\n{rotation_to_save}")
            print("已对旋转矩阵进行转置，现在该旋转矩阵与cif_to_point_cloud.py中的语义一致")
    
    if translation is not None:
        data['translation'] = torch.from_numpy(translation.astype(np.float32))
        if verbose:
            print(f"保存平移向量: {translation}")
            print("该平移向量表示从源点云到目标点云的变换，与cif_to_point_cloud.py中的语义一致")
    
    # 添加原始质心和缩放因子 - 使用真实的质心和缩放因子
    if source_centroid is not None:
        data['original_centroid'] = torch.from_numpy(source_centroid.astype(np.float32)) if isinstance(source_centroid, np.ndarray) else source_centroid
        if verbose:
            print(f"保存原始质心: {source_centroid}")
    else:
        data['original_centroid'] = torch.zeros(3, dtype=torch.float32)
        if verbose:
            print("警告: 使用默认的零质心")
    
    if source_scale is not None:
        data['scale_factor'] = torch.tensor(float(source_scale), dtype=torch.float32)
        if verbose:
            print(f"保存缩放因子: {source_scale}")
    else:
        data['scale_factor'] = torch.tensor(1.0, dtype=torch.float32)
        if verbose:
            print("警告: 使用默认的单位缩放因子")
            
    # 同时将归一化信息保存到grid_info中 - 确保与预测脚本兼容
    if 'grid_info' not in data:
        data['grid_info'] = {}
    
    # 设置grid_info中的归一化信息
    data['grid_info']['normalized'] = normalize
    if source_centroid is not None:
        data['grid_info']['original_centroid'] = source_centroid.astype(np.float32) if isinstance(source_centroid, np.ndarray) else source_centroid.numpy()
    else:
        data['grid_info']['original_centroid'] = np.zeros(3, dtype=np.float32)
        
    if source_scale is not None:
        data['grid_info']['scale_factor'] = float(source_scale)
    else:
        data['grid_info']['scale_factor'] = 1.0
        
    if verbose:
        print(f"同时将归一化信息保存到grid_info中, normalized={normalize}")
    
    # 保存为 pkl 文件
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    if verbose:
        print(f"已保存点云数据集到 {output_path}")
        print(f"源点云形状: {source_points.shape}")
        print(f"目标点云形状: {target_points.shape}")
        print(f"网格形状: {grid_info['grid_shape']}")
        print(f"体素大小: {grid_info['voxel_size']}")
        print(f"点云范围: {grid_info['point_cloud_range']}")
    
    return output_path

def save_as_cif_direct(coordinates, output_path, chain_id='A', residue_name='ALA', atom_name='CA'):
    """
    直接从点云坐标生成CIF文件，不需要参考CIF文件
    格式与pdb2cif.py生成的CIF文件一致
    
    参数:
        coordinates: 点云坐标，形状为[N, 3]
        output_path: 输出CIF文件路径
        chain_id: 链ID，默认为'A'
        residue_name: 残基名称，默认为'ALA'
        atom_name: 原子名称，默认为'CA'
    """
    # 确保点云是[N, 3]格式
    if coordinates.shape[0] == 3 and coordinates.shape[1] != 3:
        coordinates = coordinates.T
    
    # CIF文件头部
    header = [
        "data_protein\n",
        "#\n",
        "_entry.id protein\n",
        "#\n",
        "loop_\n",
        "_atom_site.group_PDB\n",
        "_atom_site.id\n",
        "_atom_site.type_symbol\n",
        "_atom_site.label_atom_id\n",
        "_atom_site.label_alt_id\n",
        "_atom_site.label_comp_id\n",
        "_atom_site.label_asym_id\n",
        "_atom_site.label_entity_id\n",
        "_atom_site.label_seq_id\n",
        "_atom_site.pdbx_PDB_ins_code\n",
        "_atom_site.Cartn_x\n",
        "_atom_site.Cartn_y\n",
        "_atom_site.Cartn_z\n",
        "_atom_site.occupancy\n",
        "_atom_site.B_iso_or_equiv\n",
        "_atom_site.pdbx_formal_charge\n",
        "_atom_site.auth_seq_id\n",
        "_atom_site.auth_comp_id\n",
        "_atom_site.auth_asym_id\n",
        "_atom_site.auth_atom_id\n",
        "_atom_site.pdbx_PDB_model_num\n"
    ]
    
    # 生成原子记录
    atom_records = []
    for i, coord in enumerate(coordinates):
        x, y, z = coord
        
        # 生成与pdb2cif.py相同格式的原子记录行
        record = f"ATOM {i+1} C {atom_name} . {residue_name} {chain_id} 1 {i+1} ? {x:.6f} {y:.6f} {z:.6f} 1.00 0.00 ? {i+1} {residue_name} {chain_id} {atom_name} 1\n"
        atom_records.append(record)
    
    # 写入CIF文件
    with open(output_path, 'w') as f:
        f.writelines(header)
        f.writelines(atom_records)
    
    print(f"已直接生成CIF文件: {output_path}，包含 {len(coordinates)} 个原子")
    return len(coordinates)

def visualize_point_clouds(source, target, transformed=None, title="点云可视化"):
    """可视化源点云和目标点云"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制源点云 (红色)
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], c='r', marker='.', s=10, label='源点云')
    
    # 绘制目标点云 (蓝色)
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='b', marker='.', s=10, label='目标点云')
    
    # 绘制变换后的点云 (绿色)
    if transformed is not None:
        ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c='g', marker='.', s=10, label='预测点云')
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 添加标题和图例
    ax.set_title(title)
    ax.legend()
    
    return fig

def test_rotation_translation(source_cif, target_cif, output_dir, rotation_only=False, visualize=True):
    """
    测试旋转平移矩阵的存储和应用
    
    参数:
        source_cif: 源CIF文件路径
        target_cif: 目标CIF文件路径
        output_dir: 输出目录
        rotation_only: 是否只应用旋转矩阵，默认为False
        visualize: 是否可视化点云，默认为True
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取文件名（不包含扩展名）用于命名
    source_name = os.path.splitext(os.path.basename(source_cif))[0]
    target_name = os.path.splitext(os.path.basename(target_cif))[0]
    
    print(f"\n开始处理: {source_name} -> {target_name}")
    
    try:
        # 1. 解析CIF文件
        print("\n第1步: 解析CIF文件...")
        source_points, source_atom_types, source_atom_labels, source_cif_content, source_atom_lines = parse_cif(source_cif)
        target_points, target_atom_types, target_atom_labels, target_cif_content, target_atom_lines = parse_cif(target_cif)
        
        # 2. 归一化点云
        print("\n第2步: 归一化点云...")
        source_norm, source_scale, source_centroid = normalize_point_cloud(source_points)
        target_norm, target_scale, target_centroid = normalize_point_cloud(target_points)
        
        # 3. 生成随机旋转矩阵和平移向量
        print("\n第3步: 生成随机旋转和平移...")
        rotation = random_rotation_matrix()
        translation = None if rotation_only else random_translation_vector()
        
        # 4. 应用旋转和平移到源点云
        print("\n第4步: 应用变换到源点云...")
        transformed_source = transform_point_cloud(source_norm, rotation, translation, rotation_only)
        
        # 5. 创建网格信息
        grid_info = create_grid_info(normalize=True)
        
        # 6. 创建并保存PKL文件
        print("\n第5步: 创建PKL文件...")
        pkl_name = f"{source_name}_to_{target_name}.pkl"
        pkl_path = os.path.join(output_dir, pkl_name)
        create_pkl_file(transformed_source, target_norm, pkl_path, grid_info, rotation, translation, 
                      normalize=True, source_centroid=source_centroid, source_scale=source_scale)
        
        # 7. 读取PKL文件（模拟预测过程）
        print("\n第6步: 读取PKL文件，应用存储的变换矩阵...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 从PKL获取关键数据
        rotation_from_pkl = data['rotation'].numpy()
        translation_from_pkl = None if rotation_only else data['translation'].numpy()
        
        # 8. 应用反向变换（"预测"）
        print("\n第7步: 应用反向变换（模拟预测结果）...")
        # 计算反向变换 - 使用与cif_to_point_cloud.py一致的方式
        inverse_rotation = np.transpose(rotation_from_pkl)  # 正交矩阵的逆是其转置

        # 检查逆矩阵的正确性
        identity_check = np.matmul(inverse_rotation, rotation_from_pkl)
        print(f"验证逆旋转矩阵: R^T·R 应接近单位矩阵:\n{identity_check}")

        if translation_from_pkl is not None:
            # 与cif_to_point_cloud.py一致的平移向量计算
            inverse_translation = -np.matmul(inverse_rotation, translation_from_pkl)
            print(f"计算的逆平移向量: {inverse_translation}")
        else:
            inverse_translation = None
        
        # 应用反向变换
        predicted_source = transform_point_cloud(transformed_source, inverse_rotation, inverse_translation, rotation_only)
        
        # 打印关键信息以便验证
        print("\n旋转矩阵相关信息:")
        print(f"原始旋转矩阵:\n{rotation}")
        print(f"从PKL获取的旋转矩阵:\n{rotation_from_pkl}")
        print(f"逆旋转矩阵:\n{inverse_rotation}")
        if translation_from_pkl is not None:
            print(f"原始平移向量: {translation}")
            print(f"从PKL获取的平移向量: {translation_from_pkl}")
            print(f"逆平移向量: {inverse_translation}")
            
        # 计算当前方法的RRE值 - 与predict_from_pkl.py一致的计算方式
        def calculate_rre(gt_rot, pred_rot):
            """计算相对旋转误差(RRE)，单位为度，与predict_from_pkl.py计算方式一致"""
            # 使用numpy计算
            if isinstance(gt_rot, torch.Tensor):
                gt_rot = gt_rot.numpy()
            if isinstance(pred_rot, torch.Tensor):
                pred_rot = pred_rot.numpy()
                
            # 计算旋转矩阵的差异 - predict_from_pkl.py中使用: pred_rotation.T @ gt_rotation
            mat = np.matmul(pred_rot.T, gt_rot)
            trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
            cos_theta = (trace - 1.0) * 0.5
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            rre_deg = theta * 180.0 / math.pi
            
            # 打印计算过程
            print(f"\nRRE计算方式1 (pred_rot.T @ gt_rot):")
            print(f"矩阵乘积:\n{mat}")
            print(f"矩阵乘积的trace值: {trace:.6f}")
            print(f"cos_theta值: {cos_theta:.6f}")
            print(f"得到的角度: {rre_deg:.6f}度")
            
            return rre_deg
        
        # 计算方式2: 调换矩阵顺序
        def calculate_rre_alt(gt_rot, pred_rot):
            """计算替代的RRE方法，交换矩阵顺序"""
            if isinstance(gt_rot, torch.Tensor):
                gt_rot = gt_rot.numpy()
            if isinstance(pred_rot, torch.Tensor):
                pred_rot = pred_rot.numpy()
                
            # 使用替代顺序: gt_rotation.T @ pred_rotation
            mat = np.matmul(gt_rot.T, pred_rot)
            trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
            cos_theta = (trace - 1.0) * 0.5
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            rre_deg = theta * 180.0 / math.pi
            
            # 打印计算过程
            print(f"\nRRE计算方式2 (gt_rot.T @ pred_rot):")
            print(f"矩阵乘积:\n{mat}")
            print(f"矩阵乘积的trace值: {trace:.6f}")
            print(f"cos_theta值: {cos_theta:.6f}")
            print(f"得到的角度: {rre_deg:.6f}度")
            
            return rre_deg
            
        # 计算RRE - 直接使用rotation和inverse_rotation
        print("\n计算旋转误差 (RRE):")
        rre_value = calculate_rre(rotation, inverse_rotation)
        rre_value2 = calculate_rre_alt(rotation, inverse_rotation)
        print(f"\n使用方式1计算的RRE: {rre_value:.6f}度")
        print(f"使用方式2计算的RRE: {rre_value2:.6f}度")
        
        # 计算RRE - 使用从PKL读取的旋转矩阵和我们计算的逆矩阵
        print("\n使用PKL旋转矩阵计算RRE:")
        rre_value_pkl = calculate_rre(rotation_from_pkl, inverse_rotation)
        rre_value_pkl2 = calculate_rre_alt(rotation_from_pkl, inverse_rotation)
        print(f"\n使用方式1计算的RRE (PKL矩阵): {rre_value_pkl:.6f}度")
        print(f"使用方式2计算的RRE (PKL矩阵): {rre_value_pkl2:.6f}度")
        
        # 9. 还原点云到原始坐标系
        print("\n第8步: 还原点云到原始坐标系...")
        # 还原到原始尺度
        original_source = source_norm * source_scale + source_centroid.reshape(1, 3)
        original_target = target_norm * target_scale + target_centroid.reshape(1, 3)
        transformed_original = transformed_source * source_scale + source_centroid.reshape(1, 3)
        predicted_original = predicted_source * source_scale + source_centroid.reshape(1, 3)
        
        # 10. 保存CIF文件
        print("\n第9步: 保存点云为CIF文件...")
        source_cif_output = os.path.join(output_dir, f"{source_name}_original.cif")
        target_cif_output = os.path.join(output_dir, f"{target_name}_original.cif")
        transformed_cif_output = os.path.join(output_dir, f"{source_name}_transformed.cif")
        predicted_cif_output = os.path.join(output_dir, f"{source_name}_predicted.cif")
        
        save_as_cif_direct(original_source, source_cif_output)
        save_as_cif_direct(original_target, target_cif_output)
        save_as_cif_direct(transformed_original, transformed_cif_output)
        save_as_cif_direct(predicted_original, predicted_cif_output)
        
        # 评估预测准确度
        print("\n第10步: 评估预测准确度...")
        mse = np.mean(np.sum((original_source - predicted_original) ** 2, axis=1))
        print(f"预测MSE误差: {mse:.8f}")
        
        if np.allclose(original_source, predicted_original, atol=1e-5):
            print("预测结果与原始源点云完全匹配!")
        else:
            print("预测结果与原始源点云有轻微差异。")
        
        # 11. 可视化（如果需要）
        if visualize:
            print("\n第11步: 可视化点云...")
            # 可视化归一化点云
            fig1 = visualize_point_clouds(source_norm, target_norm, transformed_source, title="归一化点云")
            fig2 = visualize_point_clouds(source_norm, transformed_source, predicted_source, title="源点云、变换点云和预测点云")
            
            # 保存可视化结果
            fig1_path = os.path.join(output_dir, f"{source_name}_to_{target_name}_normalized.png")
            fig2_path = os.path.join(output_dir, f"{source_name}_to_{target_name}_predicted.png")
            fig1.savefig(fig1_path)
            fig2.savefig(fig2_path)
            plt.close(fig1)
            plt.close(fig2)
            print(f"已保存可视化图像: {fig1_path} 和 {fig2_path}")
        
        print("\n处理完成!")
        return True
    
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="测试旋转平移矩阵的存储和应用")
    parser.add_argument("--source_cif", type=str, required=True, help="源CIF文件路径")
    parser.add_argument("--target_cif", type=str, required=True, help="目标CIF文件路径")
    parser.add_argument("--output_dir", type=str, default="./test_output", help="输出目录")
    parser.add_argument("--rotation_only", action="store_true", help="是否只应用旋转（不应用平移）")
    parser.add_argument("--no_visualize", action="store_true", help="不生成可视化图像")
    
    args = parser.parse_args()
    
    # 打印参数
    print("\n测试参数:")
    print(f"  源CIF文件: {args.source_cif}")
    print(f"  目标CIF文件: {args.target_cif}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  只应用旋转: {args.rotation_only}")
    print(f"  可视化: {not args.no_visualize}")
    
    # 调用测试函数
    start_time = time.time()
    success = test_rotation_translation(
        args.source_cif,
        args.target_cif,
        args.output_dir,
        args.rotation_only,
        not args.no_visualize
    )
    end_time = time.time()
    
    if success:
        print(f"\n测试完成! 用时: {end_time - start_time:.2f} 秒")
    else:
        print(f"\n测试失败! 用时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
