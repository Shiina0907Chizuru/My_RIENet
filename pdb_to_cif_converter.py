#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import glob
import csv
import argparse
from Bio.PDB import PDBParser
import numpy as np

def save_as_cif_direct(coordinates, output_path, chain_id='A', residue_name='ALA', atom_name='CA'):
    """
    直接从点云坐标生成CIF文件
    
    参数:
        coordinates: 点云坐标，格式为[N, 3]
        output_path: 输出CIF文件路径
        chain_id: 链ID，默认为'A'
        residue_name: 残基名称，默认为'ALA'
        atom_name: 原子名称，默认为'CA'
    """
    # 确保点云是[N, 3]格式
    if isinstance(coordinates, np.ndarray):
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
    return len(coordinates)  # 返回原子数量

def extract_coordinates_from_pdb(pdb_file):
    """
    从PDB文件中提取原子坐标
    
    参数:
        pdb_file: PDB文件路径
    
    返回:
        coordinates: numpy数组，形状为[N, 3]，包含所有原子的坐标
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        
        # 提取所有原子的坐标
        coordinates = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coordinates.append(atom.get_coord())
        
        return np.array(coordinates)
    
    except Exception as e:
        print(f"读取PDB文件 {pdb_file} 出错: {str(e)}")
        return np.array([])

def convert_pdb_to_cif(input_dir, output_dir, pattern="*_sample.pdb"):
    """
    将指定目录下所有匹配模式的PDB文件转换为CIF格式
    
    参数:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        pattern: 文件名模式，默认为"*_sample.pdb"
    
    返回:
        results: 包含文件名和点数量的列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有匹配模式的PDB文件
    pdb_files = glob.glob(os.path.join(input_dir, pattern))
    print(f"找到 {len(pdb_files)} 个匹配的PDB文件")
    
    results = []
    
    for pdb_file in pdb_files:
        # 构建输出CIF文件路径
        base_name = os.path.basename(pdb_file)
        name_without_ext = os.path.splitext(base_name)[0]
        cif_file = os.path.join(output_dir, f"{name_without_ext}.cif")
        
        print(f"正在处理: {base_name}")
        
        # 从PDB文件中提取坐标
        coordinates = extract_coordinates_from_pdb(pdb_file)
        
        if len(coordinates) > 0:
            # 将坐标保存为CIF格式
            points_count = save_as_cif_direct(coordinates, cif_file)
            
            # 保存文件名和点数量的信息
            results.append({
                'file_name': base_name,
                'cif_file': os.path.basename(cif_file),
                'points_count': points_count
            })
        else:
            print(f"警告: 无法从 {pdb_file} 中提取坐标")
    
    return results

def save_statistics_to_csv(results, output_file):
    """
    将统计结果保存到CSV文件
    
    参数:
        results: 包含文件名和点数量的列表
        output_file: 输出CSV文件路径
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['file_name', 'cif_file', 'points_count']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"已将统计结果保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="将PDB文件转换为CIF格式并统计点数量")
    parser.add_argument("--input_dir", type=str, required=True, help="输入目录路径")
    parser.add_argument("--output_dir", type=str, default="./cif_output", help="输出CIF文件的目录路径")
    parser.add_argument("--pattern", type=str, default="*_sample.pdb", help="文件名匹配模式")
    parser.add_argument("--csv_output", type=str, default="points_statistics.csv", help="输出CSV文件路径")
    
    args = parser.parse_args()
    
    print("开始转换PDB文件到CIF格式...")
    results = convert_pdb_to_cif(args.input_dir, args.output_dir, args.pattern)
    
    # 保存统计结果
    if len(results) > 0:
        save_statistics_to_csv(results, args.csv_output)
        print(f"\n处理完成！共转换 {len(results)} 个文件")
    else:
        print("\n没有找到匹配的PDB文件或处理过程中出错")

if __name__ == "__main__":
    main()
