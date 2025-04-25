#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import glob
from pathlib import Path

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_and_organize_files(chaindata_dir, cifdata_dir, output_dir):
    """
    从chaindata和cifdata目录提取并整合CIF文件到输出目录
    
    Args:
        chaindata_dir: 包含提取链的CIF文件的目录，如/zhaoxuanj/Point_dataset/chaindata_428/
        cifdata_dir: 包含原始CIF文件的目录，如/zhaoxuanj/Point_dataset/cifdata_428/
        output_dir: 输出目录，如/zhaoxuanj/Point_dataset/prepare2chainpkl/
    """
    # 确保输出根目录存在
    ensure_dir(output_dir)
    
    # 获取所有蛋白质目录
    pdb_folders = [f for f in os.listdir(chaindata_dir) if os.path.isdir(os.path.join(chaindata_dir, f)) and f.startswith("PDB-")]
    
    print(f"找到 {len(pdb_folders)} 个蛋白质文件夹")
    
    for pdb_folder in sorted(pdb_folders):
        pdb_id = pdb_folder.replace("PDB-", "").lower()  # 如 8h1p
        
        # 构建路径
        chaindata_pdb_path = os.path.join(chaindata_dir, pdb_folder)
        
        # 查找蛋白质子文件夹（如 8h1p）
        protein_folders = [d for d in os.listdir(chaindata_pdb_path) 
                           if os.path.isdir(os.path.join(chaindata_pdb_path, d)) 
                           and d.lower().startswith(pdb_id)]
        
        if not protein_folders:
            print(f"警告：在 {chaindata_pdb_path} 中未找到蛋白质子文件夹")
            continue
        
        for protein_folder in protein_folders:
            print(f"处理蛋白质: {protein_folder}")
            
            # 构建蛋白质相关的所有路径
            chaindata_protein_path = os.path.join(chaindata_pdb_path, protein_folder)
            cifdata_pdb_path = os.path.join(cifdata_dir, pdb_folder)
            output_protein_path = os.path.join(output_dir, pdb_folder, protein_folder)
            
            # 确保输出目录存在
            ensure_dir(output_protein_path)
            
            # 1. 复制链CIF文件
            chain_files = [f for f in os.listdir(chaindata_protein_path) if f.endswith(".cif")]
            for chain_file in chain_files:
                src_path = os.path.join(chaindata_protein_path, chain_file)
                dst_path = os.path.join(output_protein_path, chain_file)
                shutil.copy2(src_path, dst_path)
                print(f"  复制链文件: {chain_file}")
            
            # 2. 查找并复制主CIF文件
            if os.path.exists(cifdata_pdb_path):
                # 查找匹配的CIF文件（排除_point.cif）
                main_cif_pattern = os.path.join(cifdata_pdb_path, f"{protein_folder.lower()}.cif")
                main_cif_files = glob.glob(main_cif_pattern)
                
                if main_cif_files:
                    for main_cif in main_cif_files:
                        # 只复制非_point.cif文件
                        if not main_cif.endswith("_point.cif"):
                            dst_path = os.path.join(output_protein_path, os.path.basename(main_cif))
                            shutil.copy2(main_cif, dst_path)
                            print(f"  复制主CIF文件: {os.path.basename(main_cif)}")
                else:
                    print(f"  警告: 未找到蛋白质 {protein_folder} 的主CIF文件")
            else:
                print(f"  警告: 未找到对应的cifdata目录 {cifdata_pdb_path}")

def main():
    parser = argparse.ArgumentParser(description="整合链CIF文件和主CIF文件")
    parser.add_argument("--chaindata", default="/zhaoxuanj/Point_dataset/chaindata_428", 
                        help="包含链CIF文件的目录")
    parser.add_argument("--cifdata", default="/zhaoxuanj/Point_dataset/cifdata_428", 
                        help="包含主CIF文件的目录")
    parser.add_argument("--output", default="/zhaoxuanj/Point_dataset/prepare2chainpkl", 
                        help="输出目录")
    
    args = parser.parse_args()
    
    print("开始整合CIF文件...")
    print(f"链数据目录: {args.chaindata}")
    print(f"CIF数据目录: {args.cifdata}")
    print(f"输出目录: {args.output}")
    
    extract_and_organize_files(args.chaindata, args.cifdata, args.output)
    
    print("处理完成！")

if __name__ == "__main__":
    main()
