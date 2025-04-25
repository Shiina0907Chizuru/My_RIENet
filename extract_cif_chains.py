#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from collections import defaultdict

def extract_chains_from_cif(input_cif_path, output_dir=None):
    """
    从CIF文件中提取每个链并分别保存为独立的CIF文件
    
    参数：
        input_cif_path: 输入CIF文件的路径
        output_dir: 输出目录，如果为None则使用输入文件所在目录
    """
    # 如果未指定输出目录，使用输入文件所在目录
    if output_dir is None:
        output_dir = os.path.dirname(input_cif_path)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取输入文件的文件名（不包括扩展名）
    base_name = os.path.basename(input_cif_path).split('.')[0]
    
    print(f"处理CIF文件: {input_cif_path}")
    print(f"输出目录: {output_dir}")
    
    # 打开CIF文件
    with open(input_cif_path, 'r') as cif_file:
        lines = cif_file.readlines()
    
    # 查找所有链ID
    chains = set()
    atom_lines_by_chain = defaultdict(list)
    header_lines = []
    atom_section_started = False
    
    print(f"共读取 {len(lines)} 行")
    
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
    
    print(f"找到链: {chains}")
    for chain_id, atoms in atom_lines_by_chain.items():
        print(f"链 {chain_id} 有 {len(atoms)} 个原子记录")
    
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
        
        print(f"已创建链 {chain_id} 的CIF文件: {output_cif_path}")
    
    return chains

def main():
    parser = argparse.ArgumentParser(description='从CIF文件中提取每个链并保存为独立的CIF文件')
    parser.add_argument('input_cif', help='输入CIF文件路径')
    parser.add_argument('--output_dir', help='输出目录（可选）', default=None)
    
    args = parser.parse_args()
    
    chains = extract_chains_from_cif(args.input_cif, args.output_dir)
    print(f"总共从{args.input_cif}中提取了{len(chains)}个链：{', '.join(sorted(chains))}")

if __name__ == "__main__":
    main()
