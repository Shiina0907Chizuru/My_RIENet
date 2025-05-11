#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import argparse
import shutil
from tqdm import tqdm
import numpy as np

def load_atom_count_file(file_path):
    """
    加载蛋白质原子数信息文件
    文件格式预期为：每行包含蛋白质ID和原子数，以冒号分隔
    例如：
    # PDB ID: CA原子数量
    6c24: 1771
    6coz: 320
    """
    protein_atom_counts = {}
    with open(file_path, 'r') as f:
        for line in f:
            # 跳过注释行和空行
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # 处理格式为"6c24: 1771"的行
            if ':' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    protein_id = parts[0].strip()
                    try:
                        atom_count = int(parts[1].strip())
                        protein_atom_counts[protein_id] = atom_count
                    except ValueError:
                        print(f"警告: 无法解析原子数 '{parts[1].strip()}' 对于蛋白质 {protein_id}")
    
    print(f"已加载 {len(protein_atom_counts)} 个蛋白质的原子数信息")
    return protein_atom_counts

def extract_protein_id(pkl_file):
    """
    从PKL文件名中提取蛋白质ID
    常见的蛋白质ID格式为：PDBID（6c24、6coz等）
    文件名格式可能为：6c24_transform.pkl、6c24_point_cloud.pkl等
    """
    base_name = os.path.basename(pkl_file)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # 先尝试从文件名开头提取PDBID（通常是4个字符，如6c24）
    if "_" in name_without_ext:
        # 从第一个下划线前提取
        protein_id = name_without_ext.split('_')[0]
    else:
        # 没有下划线，使用整个文件名
        protein_id = name_without_ext
    
    # 确保提取的ID符合形式（一般是4个字符）
    # 我们会保留它并警告，但若ID过长可能需要人工检查
    if len(protein_id) > 6:
        print(f"警告: 提取的ID '{protein_id}' 不符合标准的PDBID格式，可能需要手动检查")
    
    return protein_id

def filter_pkl_files(pkl_dir, output_dir, atom_counts, threshold, copy=True):
    """
    筛选PKL文件，将原子数低于阈值的PKL文件移动或复制到输出目录
    
    参数:
        pkl_dir: 包含PKL文件的目录
        output_dir: 筛选后的PKL文件存放目录
        atom_counts: 蛋白质原子数字典
        threshold: 原子数阈值
        copy: 是否复制文件（True）或移动文件（False）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有PKL文件
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]
    
    if not pkl_files:
        print(f"警告: 在 {pkl_dir} 中未找到PKL文件")
        return
    
    filtered_count = 0
    missing_info_count = 0
    
    for pkl_file in tqdm(pkl_files, desc="筛选PKL文件"):
        protein_id = extract_protein_id(pkl_file)
        
        if protein_id in atom_counts:
            atom_count = atom_counts[protein_id]
            
            # 检查是否满足阈值条件
            if atom_count < threshold:
                source_path = os.path.join(pkl_dir, pkl_file)
                dest_path = os.path.join(output_dir, pkl_file)
                
                if copy:
                    shutil.copy2(source_path, dest_path)
                else:
                    shutil.move(source_path, dest_path)
                
                filtered_count += 1
        else:
            missing_info_count += 1
            print(f"警告: 未找到蛋白质 {protein_id} 的原子数信息")
    
    print(f"筛选完成! 共处理 {len(pkl_files)} 个PKL文件")
    print(f"其中 {filtered_count} 个文件原子数低于阈值 {threshold}")
    print(f"有 {missing_info_count} 个PKL文件未找到对应的原子数信息")
    
    return filtered_count

def check_pkl_atom_count(pkl_file):
    """
    检查PKL文件中的点云原子数
    """
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # 尝试不同可能的键名获取点云
        if 'source' in data:
            points = data['source']
        elif 'points' in data:
            points = data['points']
        else:
            print(f"无法在PKL文件中找到点云数据")
            return None
        
        # 转换为numpy数组以获取形状
        if isinstance(points, np.ndarray):
            point_count = points.shape[0] if points.shape[0] != 3 else points.shape[1]
        else:  # 假设是torch.Tensor
            points_np = points.detach().cpu().numpy()
            point_count = points_np.shape[0] if points_np.shape[0] != 3 else points_np.shape[1]
        
        return point_count
    
    except Exception as e:
        print(f"读取PKL文件时出错: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='根据蛋白质原子数筛选PKL文件')
    parser.add_argument('--atom_count_file', type=str, required=True,
                        help='包含蛋白质原子数信息的文件')
    parser.add_argument('--pkl_dir', type=str, required=True,
                        help='包含PKL文件的目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='筛选后的PKL文件输出目录')
    parser.add_argument('--threshold', type=int, default=1000,
                        help='原子数阈值（默认：1000）')
    parser.add_argument('--move', action='store_true',
                        help='移动文件而非复制')
    
    args = parser.parse_args()
    
    # 加载原子数信息
    atom_counts = load_atom_count_file(args.atom_count_file)
    
    # 筛选PKL文件
    filter_pkl_files(args.pkl_dir, args.output_dir, atom_counts, args.threshold, copy=not args.move)

if __name__ == '__main__':
    main()