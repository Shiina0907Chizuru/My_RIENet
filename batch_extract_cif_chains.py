#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import time
from collections import defaultdict
import csv
import re

def extract_chains_from_cif(input_cif_path, output_dir):
    """
    从CIF文件中提取每个链并分别保存为独立的CIF文件
    
    参数：
        input_cif_path: 输入CIF文件的路径
        output_dir: 输出目录
        
    返回：
        chains: 提取的链ID集合
        chain_atom_counts: 每个链的原子数量字典
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取输入文件的文件名（不包括扩展名）
    base_name = os.path.basename(input_cif_path).split('.')[0]
    
    # 打开CIF文件
    with open(input_cif_path, 'r') as cif_file:
        lines = cif_file.readlines()
    
    # 查找所有链ID
    chains = set()
    atom_lines_by_chain = defaultdict(list)
    header_lines = []
    atom_section_started = False
    
    # 第一次遍历：收集头部信息和所有链ID
    for i, line in enumerate(lines):
        if not atom_section_started and not line.startswith('ATOM ') and not line.startswith('HETATM '):
            header_lines.append(line)
        
        # 检测ATOM记录部分的开始
        if line.startswith('ATOM ') or line.startswith('HETATM '):
            atom_section_started = True
            parts = line.split()
            # 根据CIF格式，链ID通常在label_asym_id或auth_asym_id字段
            # 从样例看，第7个字段（索引6）是label_asym_id
            if len(parts) > 6:
                chain_id = parts[6]  # label_asym_id
                chains.add(chain_id)
                atom_lines_by_chain[chain_id].append(line)
    
    # 记录每个链的原子数量
    chain_atom_counts = {chain_id: len(atoms) for chain_id, atoms in atom_lines_by_chain.items()}
    
    # 为每个链创建单独的CIF文件
    for chain_id in chains:
        output_cif_path = os.path.join(output_dir, f"{base_name}_{chain_id}.cif")
        
        with open(output_cif_path, 'w') as out_file:
            # 写入头部信息
            out_file.write(f"data_{base_name}_{chain_id}\n")
            
            # 写入一些基本元数据
            for line in header_lines:
                if not line.startswith('data_'):  # 排除原始data_行
                    out_file.write(line)
            
            # 写入ATOM记录
            for line in atom_lines_by_chain[chain_id]:
                out_file.write(line)
            
            # 写入END标记
            out_file.write("# \n")
    
    return chains, chain_atom_counts

def batch_process_cif_files(input_dir, output_dir):
    """
    批量处理目录中的所有CIF文件
    
    参数：
        input_dir: 输入目录
        output_dir: 输出目录
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建CSV文件以保存结果摘要
    summary_file_path = os.path.join(output_dir, "chain_extraction_summary.csv")
    with open(summary_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["蛋白质ID", "子文件夹", "CIF文件", "链数量", "链ID", "每链原子数"])
        
        # 遍历输入目录的所有子文件夹
        total_proteins = 0
        total_chains = 0
        
        print(f"开始处理位于 {input_dir} 的CIF文件...")
        start_time = time.time()
        
        for root, dirs, files in os.walk(input_dir):
            # 获取相对于输入目录的子文件夹路径
            rel_path = os.path.relpath(root, input_dir)
            if rel_path == '.':
                rel_path = ''
            
            # 对每个子文件夹中的CIF文件进行处理
            cif_files = [f for f in files if f.lower().endswith('.cif') and not f.lower().endswith('_point.cif')]
            
            for cif_file in cif_files:
                # 提取蛋白质ID（通常是文件名的前4个字符，如7yzi）
                protein_id_match = re.match(r'([0-9a-zA-Z]+)\.cif', cif_file, re.IGNORECASE)
                if protein_id_match:
                    protein_id = protein_id_match.group(1)
                else:
                    protein_id = os.path.splitext(cif_file)[0]
                
                input_cif_path = os.path.join(root, cif_file)
                
                # 创建相应的输出子文件夹
                output_subdir = os.path.join(output_dir, rel_path, protein_id)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                
                try:
                    print(f"处理 {input_cif_path}...")
                    chains, chain_atom_counts = extract_chains_from_cif(input_cif_path, output_subdir)
                    
                    # 为CSV文件准备数据
                    chain_ids_str = ", ".join(sorted(chains))
                    atoms_per_chain = "; ".join([f"{chain}:{count}" for chain, count in chain_atom_counts.items()])
                    
                    # 写入CSV文件
                    csv_writer.writerow([
                        protein_id,
                        rel_path,
                        cif_file,
                        len(chains),
                        chain_ids_str,
                        atoms_per_chain
                    ])
                    
                    print(f"已从 {cif_file} 中提取 {len(chains)} 条链: {chain_ids_str}")
                    total_proteins += 1
                    total_chains += len(chains)
                    
                except Exception as e:
                    print(f"处理 {input_cif_path} 时出错: {str(e)}")
                    # 记录错误到CSV
                    csv_writer.writerow([
                        protein_id,
                        rel_path,
                        cif_file,
                        "错误",
                        str(e),
                        ""
                    ])
        
        # 计算并记录总结信息
        end_time = time.time()
        processing_time = end_time - start_time
        
        summary_info = [
            ["总处理蛋白质数", total_proteins],
            ["总提取链数", total_chains],
            ["处理时间(秒)", f"{processing_time:.2f}"]
        ]
        
        for info in summary_info:
            csv_writer.writerow(info)
        
        print(f"\n批处理完成!")
        print(f"处理了 {total_proteins} 个蛋白质，共提取 {total_chains} 条链")
        print(f"总共用时: {processing_time:.2f} 秒")
        print(f"摘要信息已保存到: {summary_file_path}")
    
    return summary_file_path

def main():
    parser = argparse.ArgumentParser(description='批量处理CIF文件并提取每个链为独立的CIF文件')
    parser.add_argument('--input_dir', required=True, help='输入目录，包含CIF文件的文件夹')
    parser.add_argument('--output_dir', required=True, help='输出目录，保存提取的链CIF文件')
    
    args = parser.parse_args()
    
    summary_path = batch_process_cif_files(args.input_dir, args.output_dir)
    print(f"处理摘要已保存至: {summary_path}")

if __name__ == "__main__":
    main()
