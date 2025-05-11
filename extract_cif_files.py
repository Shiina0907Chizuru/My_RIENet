#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取大文件夹中所有子文件夹的cif文件到指定输出文件夹
"""

import os
import shutil
import argparse
from tqdm import tqdm
import concurrent.futures

def extract_cif_files(input_dir, output_dir, max_workers=None, use_parallel=True):
    """
    从输入目录下的所有子文件夹中提取cif文件到输出目录
    
    参数:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        max_workers (int, optional): 最大并行工作线程数，默认为None（自动）
        use_parallel (bool, optional): 是否使用并行处理，默认为True
    
    返回:
        int: 成功提取的cif文件数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有子文件夹路径
    subdirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) 
               if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"找到 {len(subdirs)} 个子文件夹")
    
    # 定义处理单个子文件夹的函数
    def process_subdir(subdir):
        cif_files = []
        # 查找该子文件夹中的所有cif文件
        for root, _, files in os.walk(subdir):
            for file in files:
                if file.lower().endswith('.cif'):
                    cif_files.append(os.path.join(root, file))
        
        # 复制cif文件到输出目录
        subdir_name = os.path.basename(subdir)  # 子文件夹名即为蛋白质名称
        for cif_file in cif_files:
            # 构建目标文件名（蛋白质名称_mrc2cif.cif）
            target_filename = f"{subdir_name}_mrc2cif.cif"
            target_path = os.path.join(output_dir, target_filename)
            
            # 复制文件
            shutil.copy2(cif_file, target_path)
        
        return len(cif_files)
    
    # 处理所有子文件夹
    total_files = 0
    if use_parallel and len(subdirs) > 1:
        # 使用并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_subdir, subdirs),
                total=len(subdirs),
                desc="处理子文件夹"
            ))
            total_files = sum(results)
    else:
        # 使用串行处理
        for subdir in tqdm(subdirs, desc="处理子文件夹"):
            total_files += process_subdir(subdir)
    
    print(f"共提取了 {total_files} 个cif文件到 {output_dir}")
    return total_files

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='从大文件夹中提取所有子文件夹的cif文件')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录路径（包含多个子文件夹）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径（存放提取的cif文件）')
    parser.add_argument('--max_workers', type=int, default=None, help='最大并行工作线程数，默认为自动')
    parser.add_argument('--no_parallel', action='store_true', help='禁用并行处理')
    
    args = parser.parse_args()
    
    # 提取cif文件
    extract_cif_files(
        args.input_dir, 
        args.output_dir, 
        max_workers=args.max_workers,
        use_parallel=not args.no_parallel
    )

if __name__ == "__main__":
    main()
