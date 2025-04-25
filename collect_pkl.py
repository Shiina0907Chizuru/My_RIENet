#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
from tqdm import tqdm

def collect_train_pkl(source_dir, target_dir):
    """
    收集指定目录及其子目录中所有符合条件的训练数据pkl文件，复制到目标目录。
    收集的文件应该是像6c24_point_train.pkl这样的格式，但排除6c24_point_train_c.pkl这样带_c后缀的文件。
    
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
            # 检查文件是否为pkl文件且名称中不以_c结尾（排除_c.pkl结尾的文件）
            if file.endswith('.pkl') and not file.endswith('_c.pkl'):
                # 确保是训练数据文件
                if '_point_train' in file:
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
    parser = argparse.ArgumentParser(description='收集训练用的普通pkl文件（排除_c版本）')
    parser.add_argument('--source', type=str, required=True, help='源目录路径')
    parser.add_argument('--target', type=str, required=True, help='目标目录路径')
    
    args = parser.parse_args()
    collect_train_pkl(args.source, args.target)
