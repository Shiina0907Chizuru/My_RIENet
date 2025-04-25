#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import argparse
from tqdm import tqdm
import csv

def collect_pkl_files(input_dir, output_dir):
    """
    收集输入目录及其子目录中的所有PKL文件，直接复制到输出目录
    
    Args:
        input_dir: 包含PKL文件的输入目录及其子目录
        output_dir: 将所有PKL文件移动到的输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建CSV文件记录处理结果
    csv_path = os.path.join(output_dir, "pkl_collection_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["原始PKL路径", "新PKL路径"])
        
        # 收集所有PKL文件
        all_pkl_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.pkl'):
                    all_pkl_files.append(os.path.join(root, file))
        
        print(f"找到 {len(all_pkl_files)} 个PKL文件")
        
        # 处理已存在文件的计数
        already_exists_count = 0
        copied_count = 0
        
        # 处理并移动每个PKL文件
        for pkl_file in tqdm(all_pkl_files, desc="处理PKL文件"):
            # 获取原始文件名
            file_name = os.path.basename(pkl_file)
            new_pkl_path = os.path.join(output_dir, file_name)
            
            # 检查目标文件是否已存在
            if os.path.exists(new_pkl_path):
                already_exists_count += 1
                print(f"警告: 文件 {file_name} 已存在于输出目录，跳过")
                continue
            
            # 复制PKL文件到输出目录
            shutil.copy2(pkl_file, new_pkl_path)
            copied_count += 1
            
            # 记录到CSV
            csv_writer.writerow([pkl_file, new_pkl_path])
    
    print(f"处理完成! 共复制 {copied_count} 个PKL文件到 {output_dir}")
    if already_exists_count > 0:
        print(f"有 {already_exists_count} 个文件因重名而跳过")
    print(f"处理结果已保存至: {csv_path}")
    
    return copied_count

def main():
    parser = argparse.ArgumentParser(description='收集并整合PKL文件到一个目录')
    parser.add_argument('--input_dir', required=True, help='包含PKL文件的输入目录及其子目录')
    parser.add_argument('--output_dir', required=True, help='将所有PKL文件移动到的输出目录')
    
    args = parser.parse_args()
    
    print(f"开始收集PKL文件...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    
    collect_pkl_files(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
