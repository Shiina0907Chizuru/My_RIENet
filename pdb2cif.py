#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, MMCIFIO
import re
from tqdm import tqdm
import sys

def parse_pdb_file(pdb_file, ca_only=True):
    """
    解析PDB文件，提取原子坐标和相关信息
    
    参数:
        pdb_file (str): PDB文件路径
        ca_only (bool): 是否只提取CA原子
        
    返回:
        atom_records (list): 原子记录列表，每条记录包含原子信息
    """
    print(f"正在解析PDB文件: {pdb_file}")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    atom_records = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " ":  # 跳过异常残基（水分子等）
                    continue
                    
                for atom in residue:
                    if ca_only and atom.get_name() != "CA":
                        continue
                        
                    coord = atom.get_coord()
                    atom_record = {
                        "atom_name": atom.get_name(),
                        "atom_serial": atom.get_serial_number(),
                        "residue_name": residue.get_resname(),
                        "chain_id": chain.get_id(),
                        "residue_id": residue.get_id()[1],
                        "x": coord[0],
                        "y": coord[1],
                        "z": coord[2],
                        "occupancy": atom.get_occupancy() if hasattr(atom, "get_occupancy") else 1.0,
                        "temp_factor": atom.get_bfactor() if hasattr(atom, "get_bfactor") else 0.0,
                        "element": atom.element
                    }
                    atom_records.append(atom_record)
    
    print(f"成功解析 {len(atom_records)} 个原子")
    return atom_records

def parse_pdb_file_manually(pdb_file, ca_only=True):
    """
    手动解析PDB文件，不依赖BioPython，提取原子坐标和相关信息
    
    参数:
        pdb_file (str): PDB文件路径
        ca_only (bool): 是否只提取CA原子
        
    返回:
        atom_records (list): 原子记录列表，每条记录包含原子信息
    """
    print(f"正在手动解析PDB文件: {pdb_file}")
    
    atom_records = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM  "):
                atom_name = line[12:16].strip()
                
                if ca_only and atom_name != "CA":
                    continue
                
                atom_serial = int(line[6:11])
                residue_name = line[17:20].strip()
                chain_id = line[21:22].strip()
                if not chain_id:
                    chain_id = "A"  # 默认链ID
                residue_id = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                try:
                    occupancy = float(line[54:60])
                except:
                    occupancy = 1.0
                
                try:
                    temp_factor = float(line[60:66])
                except:
                    temp_factor = 0.0
                
                element = line[76:78].strip()
                if not element:
                    # 根据原子名称推断元素
                    if atom_name.startswith("C"):
                        element = "C"
                    elif atom_name.startswith("N"):
                        element = "N"
                    elif atom_name.startswith("O"):
                        element = "O"
                    elif atom_name.startswith("S"):
                        element = "S"
                    elif atom_name.startswith("H"):
                        element = "H"
                    elif atom_name.startswith("P"):
                        element = "P"
                    else:
                        element = atom_name[0]
                
                atom_record = {
                    "atom_name": atom_name,
                    "atom_serial": atom_serial,
                    "residue_name": residue_name,
                    "chain_id": chain_id,
                    "residue_id": residue_id,
                    "x": x,
                    "y": y,
                    "z": z,
                    "occupancy": occupancy,
                    "temp_factor": temp_factor,
                    "element": element
                }
                atom_records.append(atom_record)
    
    print(f"成功手动解析 {len(atom_records)} 个原子")
    return atom_records

def generate_cif_file(atom_records, output_cif):
    """
    根据原子记录生成CIF格式文件
    
    参数:
        atom_records (list): 原子记录列表
        output_cif (str): 输出的CIF文件路径
    """
    print(f"正在生成CIF文件: {output_cif}")
    
    # 生成CIF头部信息
    basename = os.path.splitext(os.path.basename(output_cif))[0]
    header = f"""data_{basename}
#
_entry.id   {basename}
#
_audit_conform.dict_name       mmcif_pdbx.dic
_audit_conform.dict_version    5.339
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
"""
    
    # 生成原子记录
    atom_lines = []
    for atom in atom_records:
        atom_line = f"ATOM {atom['atom_serial']} {atom['element']} {atom['atom_name']} . {atom['residue_name']} {atom['chain_id']} 1 {atom['residue_id']} ? {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f} {atom['occupancy']:.2f} {atom['temp_factor']:.2f} ? {atom['residue_id']} {atom['residue_name']} {atom['chain_id']} {atom['atom_name']} 1"
        atom_lines.append(atom_line)
    
    # 写入CIF文件
    with open(output_cif, 'w') as f:
        f.write(header)
        f.write("\n".join(atom_lines))
        f.write("\n#\n")
    
    print(f"成功将 {len(atom_records)} 个原子保存为CIF文件")

def pdb2cif(input_pdb, output_cif, ca_only=True, use_biopython=True):
    """
    将PDB文件转换为CIF格式
    
    参数:
        input_pdb (str): 输入的PDB文件路径
        output_cif (str): 输出的CIF文件路径
        ca_only (bool): 是否只提取CA原子
        use_biopython (bool): 是否使用BioPython进行解析
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_pdb):
        print(f"错误: 输入文件 {input_pdb} 不存在")
        return
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_cif)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 尝试使用BioPython的MMCIFIO进行直接转换
        if use_biopython and not ca_only:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", input_pdb)
            
            io = MMCIFIO()
            io.set_structure(structure)
            io.save(output_cif)
            
            print(f"成功使用BioPython将{input_pdb}转换为{output_cif}")
            return
    except Exception as e:
        print(f"BioPython直接转换失败: {e}")
        print("将使用手动解析和生成方法...")
    
    # 使用手动方法
    try:
        # 解析PDB文件
        if use_biopython:
            atom_records = parse_pdb_file(input_pdb, ca_only)
        else:
            atom_records = parse_pdb_file_manually(input_pdb, ca_only)
        
        # 生成CIF文件
        generate_cif_file(atom_records, output_cif)
        
        print(f"转换完成: {input_pdb} -> {output_cif}")
    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        raise

def batch_convert(input_dir, output_dir, ca_only=True, use_biopython=True, recursive=False):
    """
    批量转换目录中的PDB文件为CIF格式
    
    参数:
        input_dir (str): 输入目录
        output_dir (str): 输出目录
        ca_only (bool): 是否只提取CA原子
        use_biopython (bool): 是否使用BioPython进行解析
        recursive (bool): 是否递归处理子目录
    """
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有PDB文件
    if recursive:
        pdb_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".pdb"):
                    pdb_files.append(os.path.join(root, file))
    else:
        pdb_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".pdb")]
    
    print(f"找到 {len(pdb_files)} 个PDB文件")
    
    # 转换每个文件
    for pdb_file in tqdm(pdb_files, desc="转换进度"):
        # 确定输出文件路径
        if recursive:
            rel_path = os.path.relpath(pdb_file, input_dir)
            output_subdir = os.path.dirname(rel_path)
            output_path = os.path.join(output_dir, output_subdir)
            if output_subdir and not os.path.exists(output_path):
                os.makedirs(output_path)
            output_cif = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".cif")
        else:
            output_cif = os.path.join(output_dir, os.path.splitext(os.path.basename(pdb_file))[0] + ".cif")
        
        try:
            pdb2cif(pdb_file, output_cif, ca_only, use_biopython)
        except Exception as e:
            print(f"处理文件 {pdb_file} 时出错: {e}")
            continue
    
    print("批量转换完成!")

def main():
    parser = argparse.ArgumentParser(description='将PDB文件转换为CIF格式')
    parser.add_argument('--input', type=str, required=True, help='输入的PDB文件或目录')
    parser.add_argument('--output', type=str, help='输出的CIF文件或目录（如果不指定，将自动生成）')
    parser.add_argument('--ca_only', action='store_true', default=False, help='是否只提取CA原子')
    parser.add_argument('--manual', action='store_true', default=False, help='使用手动解析方法（不使用BioPython）')
    parser.add_argument('--recursive', action='store_true', default=False, help='递归处理子目录（当输入是目录时）')
    
    args = parser.parse_args()
    
    input_path = args.input
    
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入路径 {input_path} 不存在")
        return
    
    # 确定是文件还是目录
    if os.path.isdir(input_path):
        # 如果输入是目录，进行批量转换
        output_dir = args.output if args.output else os.path.join(os.path.dirname(input_path), "cif_output")
        batch_convert(input_path, output_dir, args.ca_only, not args.manual, args.recursive)
    else:
        # 如果输入是文件，转换单个文件
        if not input_path.lower().endswith(".pdb"):
            print(f"警告: 输入文件 {input_path} 不是PDB格式，将尝试转换...")
        
        output_path = args.output if args.output else os.path.splitext(input_path)[0] + ".cif"
        pdb2cif(input_path, output_path, args.ca_only, not args.manual)

if __name__ == "__main__":
    main()
