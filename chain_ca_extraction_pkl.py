#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import csv
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser, MMCIFParser, MMCIFIO, Select
import torch

class CASelect(Select):
    """只选择C阿尔法原子的选择器"""
    def accept_atom(self, atom):
        return atom.get_name() == "CA"

def parse_cif_ca_only(cif_file):
    """解析CIF文件并只提取C阿尔法原子"""
    parser = MMCIFParser()
    try:
        structure = parser.get_structure("structure", cif_file)
        
        # 提取CA原子坐标
        ca_atoms = []
        atom_types = []
        atom_labels = []
        atom_lines = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == "CA":
                            ca_atoms.append(atom.get_coord())
                            atom_types.append("CA")
                            # 构建原子标签：链ID_残基ID_原子名
                            atom_label = f"{chain.get_id()}_{residue.get_id()[1]}_{atom.get_name()}"
                            atom_labels.append(atom_label)
                            # 创建原子行
                            atom_line = (atom.get_name(), residue.get_resname(), chain.get_id(), 
                                        residue.get_id()[1], atom.get_coord())
                            atom_lines.append(atom_line)
        
        # 如果没有找到CA原子，返回None
        if not ca_atoms:
            return None, None, None, None, None
            
        # 返回点云和原子信息
        return np.array(ca_atoms), atom_types, atom_labels, None, atom_lines
        
    except Exception as e:
        print(f"解析CIF文件 {cif_file} 时出错: {str(e)}")
        return None, None, None, None, None

def extract_ca_and_save_cif(input_cif, output_cif):
    """提取CIF文件中的CA原子并保存为新的CIF文件"""
    parser = MMCIFParser()
    io = MMCIFIO()
    
    try:
        structure = parser.get_structure("structure", input_cif)
        
        # 计算CA原子数量
        ca_count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == "CA":
                            ca_count += 1
        
        # 如果没有CA原子，返回0
        if ca_count == 0:
            return 0
            
        # 保存只含CA原子的结构
        io.set_structure(structure)
        io.save(output_cif, CASelect())
        
        return ca_count
        
    except Exception as e:
        print(f"处理CIF文件 {input_cif} 时出错: {str(e)}")
        return -1  # 错误

def create_grid_info():
    """创建网格信息"""
    # 这部分沿用cif_to_point_cloud.py中的逻辑
    # 生成一些默认值，实际应用中需要根据点云范围来确定
    grid_shape = np.array([64, 64, 64], dtype=np.int32)
    
    # 将标量值改为一维数组，避免索引越界问题
    x_origin = np.array([0.0], dtype=np.float32)
    y_origin = np.array([0.0], dtype=np.float32)
    z_origin = np.array([0.0], dtype=np.float32)
    x_voxel = np.array([0.05], dtype=np.float32)  # 体素大小
    nstart = np.array([32], dtype=np.int32)  # 起始索引
    
    return grid_shape, x_origin, y_origin, z_origin, x_voxel, nstart

def create_pkl_file(source_points, target_points, output_pkl, grid_info):
    """创建PKL文件（旋转平移矩阵置零）"""
    # 解包网格信息
    grid_shape, x_origin, y_origin, z_origin, x_voxel, nstart = grid_info
    
    # 创建Numpy零矩阵用于旋转和平移
    rotation = np.zeros((3, 3), dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    
    # 检查点云形状并打印信息
    print(f"保存前源点云形状: {source_points.shape}")
    print(f"保存前目标点云形状: {target_points.shape}")
    
    # 转换为torch张量
    source_points_tensor = torch.from_numpy(source_points).float()
    target_points_tensor = torch.from_numpy(target_points).float()
    rotation_tensor = torch.from_numpy(rotation).float()
    translation_tensor = torch.from_numpy(translation).float()
    
    print(f"保存为torch张量后源点云形状: {source_points_tensor.shape}")
    print(f"保存为torch张量后目标点云形状: {target_points_tensor.shape}")
    
    # 保存为字典格式，保持与cif_to_point_cloud.py一致的格式
    data_dict = {
        'source': source_points_tensor,  # 源点云
        'target': target_points_tensor,  # 目标点云
        'rotation': rotation_tensor,     # 旋转矩阵
        'translation': translation_tensor,  # 平移向量
        'grid_shape': grid_shape,
        'x_origin': x_origin,
        'y_origin': y_origin,
        'z_origin': z_origin,
        'x_voxel': x_voxel,
        'nstart': nstart
    }
    
    # 写入PKL文件
    with open(output_pkl, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"保存PKL文件: {output_pkl}")
    print(f"保存的源点云形状: {source_points_tensor.shape}")
    print(f"保存的目标点云形状: {target_points_tensor.shape}")

def process_protein_folder(protein_folder, ca_output_dir, pkl_output_dir, csv_writer, min_ca_atoms):
    """处理蛋白质文件夹，提取CA原子，创建PKL文件"""
    protein_name = os.path.basename(protein_folder)
    print(f"\n处理蛋白质: {protein_name}")
    
    # 查找主CIF文件 (例如 8h1p.cif)
    main_cif_pattern = os.path.join(protein_folder, f"{protein_name.lower()}.cif")
    main_cif_files = glob.glob(main_cif_pattern)
    
    if not main_cif_files:
        print(f"警告: 未找到蛋白质 {protein_name} 的主CIF文件")
        csv_writer.writerow([protein_name, "主CIF文件缺失", "", "", 0, 0])
        return
        
    main_cif = main_cif_files[0]
    
    # 创建输出目录
    protein_ca_dir = os.path.join(ca_output_dir, os.path.basename(os.path.dirname(protein_folder)), protein_name)
    os.makedirs(protein_ca_dir, exist_ok=True)
    
    # PKL文件直接存放在输出目录，不创建子文件夹
    protein_pkl_dir = pkl_output_dir
    
    # 处理主CIF文件，提取CA原子
    main_ca_cif = os.path.join(protein_ca_dir, f"{protein_name}_CA.cif")
    main_ca_count = extract_ca_and_save_cif(main_cif, main_ca_cif)
    
    if main_ca_count <= 0:
        print(f"警告: 主CIF文件 {main_cif} 中没有CA原子或处理出错")
        csv_writer.writerow([protein_name, "主CIF文件无CA原子或处理出错", main_cif, "", main_ca_count, 0])
        return
    
    # 主CIF的点云数据
    main_points, main_atom_types, main_atom_labels, _, main_atom_lines = parse_cif_ca_only(main_ca_cif)
    
    if main_points is None:
        print(f"警告: 无法解析主CIF文件 {main_ca_cif} 中的CA原子")
        csv_writer.writerow([protein_name, "无法解析主CIF文件中的CA原子", main_cif, "", 0, 0])
        return
    
    print(f"  主CIF文件中共有 {main_ca_count} 个CA原子")
    
    # 查找链CIF文件 (例如 8h1p_A.cif, 8h1p_B.cif 等)
    chain_cif_pattern = os.path.join(protein_folder, f"{protein_name.lower()}_*.cif")
    chain_cif_files = [f for f in glob.glob(chain_cif_pattern) if not f.endswith("_point.cif")]
    
    if not chain_cif_files:
        print(f"警告: 未找到蛋白质 {protein_name} 的链CIF文件")
        csv_writer.writerow([protein_name, "链CIF文件缺失", main_cif, "", main_ca_count, 0])
        return
    
    print(f"  发现 {len(chain_cif_files)} 个链CIF文件")
    
    # 记录生成的PKL数量
    pkl_count = 0
    chains_without_ca = []
    
    # 处理每个链CIF文件
    for chain_cif in chain_cif_files:
        chain_name = os.path.splitext(os.path.basename(chain_cif))[0]
        chain_id = chain_name.split('_')[-1]
        
        print(f"  处理链: {chain_id}")
        
        # 提取CA原子
        chain_ca_cif = os.path.join(protein_ca_dir, f"{chain_name}_CA.cif")
        chain_ca_count = extract_ca_and_save_cif(chain_cif, chain_ca_cif)
        
        if chain_ca_count <= 0:
            print(f"    链 {chain_id} 中没有CA原子或处理出错，跳过")
            chains_without_ca.append(chain_id)
            continue
        
        # 链CIF的点云数据
        chain_points, chain_atom_types, chain_atom_labels, _, chain_atom_lines = parse_cif_ca_only(chain_ca_cif)
        
        if chain_points is None:
            print(f"    无法解析链 {chain_id} 的CA原子，跳过")
            chains_without_ca.append(chain_id)
            continue
        
        print(f"    链 {chain_id} 中共有 {chain_ca_count} 个CA原子")
        
        # 检查CA原子数量是否达到阈值
        if chain_ca_count < min_ca_atoms:
            print(f"    链 {chain_id} 中的CA原子数量 {chain_ca_count} 低于阈值 {min_ca_atoms}，跳过")
            chains_without_ca.append(chain_id)
            continue
        
        # 创建PKL文件，文件名中包含蛋白质名称以避免冲突
        pdb_id = os.path.basename(os.path.dirname(protein_folder))  # 获取PDB ID (如 PDB-8H1P)
        pkl_filename = f"{pdb_id}_{protein_name}_{chain_name}_to_{protein_name}.pkl"
        pkl_path = os.path.join(protein_pkl_dir, pkl_filename)
        
        # 确保点云为[3, N]格式
        chain_points_t = chain_points.T if chain_points.shape[1] == 3 else chain_points
        main_points_t = main_points.T if main_points.shape[1] == 3 else main_points
        
        # 创建网格信息
        grid_info = create_grid_info()
        
        # 创建并保存PKL文件
        create_pkl_file(chain_points_t, main_points_t, pkl_path, grid_info)
        pkl_count += 1
    
    # 记录处理结果
    chains_without_ca_str = ",".join(chains_without_ca) if chains_without_ca else "无"
    csv_writer.writerow([
        protein_name, 
        "处理完成", 
        main_cif, 
        chains_without_ca_str,
        main_ca_count,
        pkl_count
    ])
    
    print(f"  蛋白质 {protein_name} 处理完成，生成了 {pkl_count} 个PKL文件")
    print(f"  没有CA原子的链: {chains_without_ca_str}")
    
    return pkl_count

def main():
    parser = argparse.ArgumentParser(description='提取CIF文件中的C阿尔法原子并创建PKL文件')
    parser.add_argument('--input_dir', required=True, help='包含蛋白质文件夹的输入目录')
    parser.add_argument('--ca_output_dir', required=True, help='CA原子CIF文件的输出目录')
    parser.add_argument('--pkl_output_dir', required=True, help='PKL文件的输出目录')
    parser.add_argument('--min_ca_atoms', type=int, default=30, help='链中最少的CA原子数量，少于此数量的链将被过滤，默认为30')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.ca_output_dir, exist_ok=True)
    os.makedirs(args.pkl_output_dir, exist_ok=True)
    
    # 创建CSV文件记录处理结果
    csv_path = os.path.join(args.pkl_output_dir, "ca_extraction_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["蛋白质", "处理状态", "主CIF文件", "无CA原子的链", "主CIF中CA原子数", "生成的PKL数量"])
        
        # 统计数据
        total_proteins = 0
        total_pkls = 0
        
        # 遍历输入目录中的所有蛋白质文件夹
        for pdb_folder_name in sorted(os.listdir(args.input_dir)):
            pdb_folder_path = os.path.join(args.input_dir, pdb_folder_name)
            
            # 检查是否是目录且是PDB文件夹
            if not os.path.isdir(pdb_folder_path) or not pdb_folder_name.startswith("PDB-"):
                continue
            
            # 获取蛋白质ID
            protein_id = pdb_folder_name.replace("PDB-", "").lower()  # 如 8h1p
            
            # 查找实际的蛋白质文件夹(如 8h1p)
            protein_folders = [d for d in os.listdir(pdb_folder_path) 
                              if os.path.isdir(os.path.join(pdb_folder_path, d)) 
                              and d.lower().startswith(protein_id)]
            
            for protein_folder in protein_folders:
                total_proteins += 1
                protein_path = os.path.join(pdb_folder_path, protein_folder)
                
                # 处理蛋白质文件夹
                pkl_count = process_protein_folder(
                    protein_path, args.ca_output_dir, args.pkl_output_dir, csv_writer, args.min_ca_atoms
                )
                
                if pkl_count:
                    total_pkls += pkl_count
        
        # 写入统计数据
        csv_writer.writerow([])
        csv_writer.writerow(["总蛋白质数", total_proteins])
        csv_writer.writerow(["总生成PKL数", total_pkls])
    
    print("\n处理完成!")
    print(f"总蛋白质数: {total_proteins}")
    print(f"总生成PKL数: {total_pkls}")
    print(f"处理结果已保存至: {csv_path}")

if __name__ == "__main__":
    main()
