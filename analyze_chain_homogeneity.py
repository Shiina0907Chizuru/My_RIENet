#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import csv
import shutil
import argparse
import re
from collections import defaultdict

def get_tm_score(usalign_output):
    """从USalign输出中提取TM-score"""
    tm_score = None
    for line in usalign_output.split('\n'):
        if "TM-score= " in line and "(normalized by length of Structure_2" in line:
            tm_score = float(line.split("TM-score= ")[1].split(" (normalized")[0])
            break
    return tm_score

def compare_chains(chain1_path, chain2_path, usalign_path):
    """比对两个链并返回TM-score"""
    try:
        # 调用USalign进行结构比对，并捕获输出
        result = subprocess.run(
            [usalign_path, chain1_path, chain2_path, "-outfmt", "0"], 
            capture_output=True, 
            text=True,
            check=True
        )
        tm_score = get_tm_score(result.stdout)
        return tm_score
    except subprocess.CalledProcessError as e:
        print(f"比对错误: {e}")
        print(f"错误输出: {e.stderr}")
        return None

def process_protein_folder(protein_folder, usalign_path, csv_writer, tm_threshold=0.9):
    """处理单个蛋白质文件夹中的所有链"""
    protein_name = os.path.basename(protein_folder)
    print(f"\n处理蛋白质: {protein_name}")
    
    # 获取所有链文件
    chain_files = [f for f in os.listdir(protein_folder) if f.endswith('.cif') and not f.endswith('_point.cif')]
    chain_files.sort()  # 排序确保比对结果的一致性
    
    if len(chain_files) <= 1:
        print(f"警告: {protein_name} 文件夹中链文件数量不足 ({len(chain_files)})")
        return False, []
    
    # 初始化TM-score矩阵
    tm_matrix = {}
    high_similarity_found = False
    
    # 两两比对所有链
    for i, chain1 in enumerate(chain_files):
        chain1_path = os.path.join(protein_folder, chain1)
        chain1_id = chain1.split('_')[-1].split('.')[0]  # 提取链ID
        
        for j, chain2 in enumerate(chain_files):
            if i >= j:  # 跳过自身比较和重复比较
                continue
                
            chain2_path = os.path.join(protein_folder, chain2)
            chain2_id = chain2.split('_')[-1].split('.')[0]  # 提取链ID
            
            print(f"  比对 {chain1} 和 {chain2}")
            tm_score = compare_chains(chain1_path, chain2_path, usalign_path)
            
            if tm_score is not None:
                # 记录到矩阵
                key = (chain1_id, chain2_id)
                tm_matrix[key] = tm_score
                
                # 写入CSV
                csv_writer.writerow([
                    protein_name, 
                    chain1_id, 
                    chain2_id,
                    tm_score,
                    "同构" if tm_score > tm_threshold else "非同构"
                ])
                
                # 检查是否有高相似度
                if tm_score > tm_threshold:
                    high_similarity_found = True
                    print(f"    发现高相似度链对: {chain1_id}-{chain2_id}, TM-score={tm_score:.4f}")
    
    # 收集所有链文件路径
    chain_paths = [os.path.join(protein_folder, chain) for chain in chain_files]
    
    return high_similarity_found, chain_paths

def main():
    parser = argparse.ArgumentParser(description='蛋白质链比对与同构性分析')
    parser.add_argument('--input_dir', required=True, help='输入目录，包含PDB子文件夹')
    parser.add_argument('--output_dir', required=True, help='输出目录，存放非同构蛋白质的链')
    parser.add_argument('--usalign', required=True, help='USalign可执行文件路径')
    parser.add_argument('--threshold', type=float, default=0.9, help='TM-score阈值，默认0.9')
    
    args = parser.parse_args()
    
    # 检查USalign是否可用
    if not os.path.isfile(args.usalign):
        print(f"错误: USalign可执行文件未找到: {args.usalign}")
        sys.exit(1)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建CSV文件记录比对结果
    csv_path = os.path.join(args.output_dir, "chain_comparison_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["蛋白质", "链1", "链2", "TM-score", "同构性"])
        
        # 统计数据
        total_proteins = 0
        homogeneous_proteins = 0
        non_homogeneous_proteins = 0
        
        # 遍历输入目录中的所有PDB子文件夹
        for pdb_folder_name in sorted(os.listdir(args.input_dir)):
            pdb_folder_path = os.path.join(args.input_dir, pdb_folder_name)
            
            # 检查是否是目录且是PDB文件夹
            if not os.path.isdir(pdb_folder_path) or not pdb_folder_name.startswith("PDB-"):
                continue
            
            # 获取蛋白质ID
            protein_id = pdb_folder_name.replace("PDB-", "")
            
            # 查找实际的蛋白质文件夹(如5jzh)
            protein_folders = [d for d in os.listdir(pdb_folder_path) 
                              if os.path.isdir(os.path.join(pdb_folder_path, d)) 
                              and d.lower().startswith(protein_id.lower())]
            
            for protein_folder in protein_folders:
                total_proteins += 1
                protein_path = os.path.join(pdb_folder_path, protein_folder)
                
                # 处理蛋白质文件夹
                is_homogeneous, chain_paths = process_protein_folder(
                    protein_path, args.usalign, csv_writer, args.threshold
                )
                
                # 更新统计
                if is_homogeneous:
                    homogeneous_proteins += 1
                    print(f"{protein_folder} 是同构蛋白质")
                else:
                    non_homogeneous_proteins += 1
                    print(f"{protein_folder} 是非同构蛋白质，将复制到输出目录")
                    
                    # 创建输出子目录
                    output_subdir = os.path.join(args.output_dir, pdb_folder_name, protein_folder)
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    # 复制所有链文件
                    for chain_path in chain_paths:
                        chain_filename = os.path.basename(chain_path)
                        shutil.copy2(chain_path, os.path.join(output_subdir, chain_filename))
        
        # 写入统计数据
        csv_writer.writerow([])
        csv_writer.writerow(["总蛋白质数", total_proteins])
        csv_writer.writerow(["同构蛋白质数", homogeneous_proteins])
        csv_writer.writerow(["非同构蛋白质数", non_homogeneous_proteins])
    
    print("\n分析完成!")
    print(f"总蛋白质数: {total_proteins}")
    print(f"同构蛋白质数: {homogeneous_proteins}")
    print(f"非同构蛋白质数: {non_homogeneous_proteins}")
    print(f"比对结果已保存至: {csv_path}")
    print(f"非同构蛋白质链已复制到: {args.output_dir}")

if __name__ == "__main__":
    main()
