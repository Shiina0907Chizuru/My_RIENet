#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
from tqdm import tqdm

def collect_c_pkl_files(source_dir, target_dir):
    """
    遍历源目录的所有子文件夹，提取名称中包含'_c'的pkl文件，并复制到目标目录
    
    参数:
        source_dir: 源目录路径
        target_dir: 目标目录路径
    """
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标目录: {target_dir}")
    
    # 收集所有符合条件的文件
    collected_files = []
    
    # 遍历源目录及其所有子目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 检查文件是否为pkl文件且名称中包含'_c'
            if file.endswith('.pkl') and '_c' in file:
                source_file = os.path.join(root, file)
                collected_files.append((source_file, file))
    
    # 显示找到的文件总数
    print(f"找到 {len(collected_files)} 个符合条件的文件")
    
    # 使用tqdm显示进度条，复制文件
    for source_file, filename in tqdm(collected_files, desc="复制文件"):
        target_file = os.path.join(target_dir, filename)
        shutil.copy2(source_file, target_file)
    
    print(f"所有文件已复制到: {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='收集包含_c的pkl文件')
    parser.add_argument('--source', type=str, required=True, help='源目录路径')
    parser.add_argument('--target', type=str, required=True, help='目标目录路径')
    
    args = parser.parse_args()
    
    collect_c_pkl_files(args.source, args.target)