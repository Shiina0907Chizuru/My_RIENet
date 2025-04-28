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
import re

class CASelect(Select):
    """只选择C阿尔法原子的选择器"""
    def accept_atom(self, atom):
        return atom.get_name() == "CA"

def parse_cif_ca_only(cif_path):
    """解析CIF文件，只提取CA原子的坐标，并保存行索引"""
    # 打开并读取CIF文件内容
    with open(cif_path, 'r') as f:
        cif_content = f.readlines()
    
    # 使用BioPython解析CIF文件
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", cif_path)
    except Exception as e:
        print(f"无法解析CIF文件: {cif_path}, 错误: {e}")
        return None, None, None, None, None
    
    # 提取CA原子坐标
    ca_coords = []
    atom_types = []
    atom_labels = []
    atom_lines = []
    
    # 使用正则表达式查找CA原子的行
    ca_atom_pattern = re.compile(r'^ATOM\s+\d+\s+CA\s+')
    
    # 首先找到所有CA原子的行索引
    for i, line in enumerate(cif_content):
        if line.strip().startswith('ATOM') and ('CA' in line):
            fields = line.strip().split()
            if len(fields) > 13 and fields[3] == 'CA':  # 检查是否是CA原子
                atom_lines.append(i)  # 保存行索引
    
    # 使用BioPython提取原子信息和坐标
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == 'CA':
                        # 获取原子坐标
                        ca_coords.append(atom.get_coord())
                        # 保存原子类型和标签信息
                        atom_types.append('CA')
                        atom_labels.append(f"{residue.get_resname()}{residue.get_id()[1]}")
    
    # 检查是否找到任何CA原子
    if len(ca_coords) == 0:
        print(f"警告: 在文件中未找到CA原子: {cif_path}")
        return None, None, None, None, None
    
    # 检查原子坐标数量与行索引数量是否一致
    if len(ca_coords) != len(atom_lines):
        print(f"警告: CA原子数量({len(ca_coords)})与行索引数量({len(atom_lines)})不一致")
        # 如果不一致，使用BioPython解析的坐标数量作为准，重新解析
        atom_lines = []
        for i, line in enumerate(cif_content):
            if len(atom_lines) < len(ca_coords) and line.strip().startswith('ATOM') and ('CA' in line):
                fields = line.strip().split()
                if len(fields) > 13 and fields[3] == 'CA':
                    atom_lines.append(i)
    
    # 转换为NumPy数组
    ca_coords = np.array(ca_coords, dtype=np.float32)
    
    return ca_coords, atom_types, atom_labels, cif_content, atom_lines

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

def create_pkl_file(source_points, target_points, rotation, translation, output_pkl, grid_info):
    """创建PKL文件，将旋转平移矩阵保存到文件中"""
    # 解包网格信息
    grid_shape, x_origin, y_origin, z_origin, x_voxel, nstart = grid_info
    
    # 检查点云形状并打印信息
    print(f"源点云形状: {source_points.shape}")
    print(f"目标点云形状: {target_points.shape}")
    print(f"旋转矩阵形状: {rotation.shape}")
    print(f"平移向量: {translation}")
    print(f"输出文件: {output_pkl}")
    
    # 转换为torch张量
    source_points_tensor = torch.from_numpy(source_points).float()
    target_points_tensor = torch.from_numpy(target_points).float()
    rotation_tensor = torch.from_numpy(rotation).float()
    translation_tensor = torch.from_numpy(translation).float()
    
    # 转换网格信息为torch张量
    grid_shape_tensor = torch.from_numpy(grid_shape).int()
    x_origin_tensor = torch.from_numpy(x_origin).float()
    y_origin_tensor = torch.from_numpy(y_origin).float()
    z_origin_tensor = torch.from_numpy(z_origin).float()
    x_voxel_tensor = torch.from_numpy(x_voxel).float()
    nstart_tensor = torch.from_numpy(nstart).int()
    
    # 保存为字典格式，保持与cif_to_point_cloud.py一致的格式
    data_dict = {
        'source': source_points_tensor,       # 源点云
        'target': target_points_tensor,       # 目标点云
        'rotation': rotation_tensor,          # 旋转矩阵
        'translation': translation_tensor,    # 平移向量
        'grid_shape': grid_shape_tensor,      # 格点形状
        'x_origin': x_origin_tensor,          # x原点
        'y_origin': y_origin_tensor,          # y原点
        'z_origin': z_origin_tensor,          # z原点
        'x_voxel': x_voxel_tensor,            # 体素大小
        'nstart': nstart_tensor               # 起始索引
    }
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(output_pkl)), exist_ok=True)
    
    # 写入PKL文件
    with open(output_pkl, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"已保存PKL文件: {output_pkl}")

def random_rotation_matrix():
    """生成随机旋转矩阵"""
    # 使用QR分解生成随机旋转矩阵
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    
    # 确保是特殊正交矩阵(行列式为1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    
    return Q.astype(np.float32)

def random_translation_vector(scale=100.0):
    """生成随机平移向量"""
    return np.random.uniform(-scale, scale, size=(3,)).astype(np.float32)

def transform_point_cloud_numpy(point_cloud, rotation, translation, rotation_only=False):
    """使用numpy将点云进行变换 - 使用右乘方式：点云 = 旋转矩阵 * 点云 + 平移向量"""
    # 确保点云是[3, N]格式
    if point_cloud.shape[1] == 3 and point_cloud.shape[0] != 3:
        print(f"转置点云: 从{point_cloud.shape}到", end="")
        point_cloud = point_cloud.T
        print(f"{point_cloud.shape}")
    
    # 现在point_cloud已经是[3, N]格式，可以直接应用旋转和平移
    # R[3,3] * point_cloud[3,N] + t[3,1] = transformed_point_cloud[3,N]
    if rotation_only:
        # 只应用旋转，不应用平移
        transformed_point_cloud = np.matmul(rotation, point_cloud)
        print("应用旋转变换（无平移）")
    else:
        # 应用旋转和平移
        transformed_point_cloud = np.matmul(rotation, point_cloud) + translation.reshape(3, 1)
    
    return transformed_point_cloud

def calculate_centroid(point_cloud):
    """计算点云的质心"""
    # 确保点云是[3, N]格式
    if point_cloud.shape[1] == 3 and point_cloud.shape[0] != 3:
        point_cloud = point_cloud.T
    
    # 计算质心
    centroid = np.mean(point_cloud, axis=1)
    return centroid

def save_as_cif(coordinates, atom_types, atom_labels, original_cif, atom_lines, output_path):
    """将点云坐标保存为CIF格式"""
    if original_cif is None or len(original_cif) == 0:
        print(f"警告：无法保存CIF文件 {output_path}，原始CIF内容为空")
        return
    
    if atom_lines is None or len(atom_lines) == 0:
        print(f"警告：无法保存CIF文件 {output_path}，原子行索引为空")
        return
    
    # 检查atom_lines的格式，确保是整数（行索引）
    if not all(isinstance(idx, int) for idx in atom_lines):
        print(f"警告：atom_lines不是行索引格式，无法保存CIF文件 {output_path}")
        print(f"atom_lines类型: {type(atom_lines[0])}")
        return
    
    # 确保coordinates是[N, 3]格式
    if coordinates.shape[0] == 3 and coordinates.shape[1] != 3:
        coordinates = coordinates.T
    
    # 确保坐标数量与行索引数量一致
    if len(coordinates) != len(atom_lines):
        print(f"警告：坐标数量({len(coordinates)})与行索引数量({len(atom_lines)})不一致")
        return
    
    new_cif = original_cif.copy()
    
    # 更新原子坐标
    for i, (coord, line_idx) in enumerate(zip(coordinates, atom_lines)):
        if line_idx >= len(original_cif):
            print(f"警告：行索引{line_idx}超出原始CIF内容范围{len(original_cif)}")
            continue
            
        x, y, z = coord
        line = original_cif[line_idx].strip().split()
        
        # 检查行是否有足够的字段
        if len(line) < 13:
            print(f"警告：CIF文件行{line_idx}格式不正确: {line}")
            continue
        
        # 保持原始行的格式，只更新坐标部分
        line[10] = f"{x:.6f}"
        line[11] = f"{y:.6f}"
        line[12] = f"{z:.6f}"
        
        new_cif[line_idx] = "  ".join(line) + "\n"
    
    # 添加注释说明这是变换后的结构
    header_comment = f"# This is a transformed structure\n"
    new_cif.insert(0, header_comment)
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 写入新的CIF文件
    with open(output_path, 'w') as f:
        f.writelines(new_cif)
    
    print(f"已保存点云为CIF格式: {output_path}")

def process_protein_folder(input_dir, ca_output_dir, pkl_output_dir, transform_cif_dir=None):
    """处理一个蛋白质文件夹，提取CA原子，生成随机旋转和平移，并创建PKL文件"""
    # 确保输出目录存在
    os.makedirs(ca_output_dir, exist_ok=True)
    os.makedirs(pkl_output_dir, exist_ok=True)
    if transform_cif_dir:
        os.makedirs(transform_cif_dir, exist_ok=True)

    # 获取输入文件夹中的所有CIF文件
    cif_files = glob.glob(os.path.join(input_dir, "*.cif"))
    print(f"找到 {len(cif_files)} 个CIF文件")
    
    # 用于统计处理的文件数量
    ca_count = 0
    pkl_count = 0
    transform_count = 0
    
    for cif_file in tqdm(cif_files, desc="处理CIF文件"):
        # 获取文件名（不含扩展名）
        filename = os.path.basename(cif_file)
        base_name = os.path.splitext(filename)[0]
        
        # 构建输出文件路径
        ca_output_path = os.path.join(ca_output_dir, f"{base_name}_CA.cif")
        pkl_path = os.path.join(pkl_output_dir, f"{base_name}.pkl")
        
        # 步骤1: 提取CA原子并保存为新的CIF文件
        if not os.path.exists(ca_output_path):
            try:
                extract_ca_and_save_cif(cif_file, ca_output_path)
                ca_count += 1
            except Exception as e:
                print(f"处理文件 {cif_file} 时出错: {str(e)}")
                continue
        else:
            print(f"CA文件已存在: {ca_output_path}")
        
        # 步骤2: 从CA文件中解析点云并创建PKL文件
        try:
            # 从CA文件中读取点云
            chain_ca_coords, atom_types, atom_labels, cif_content, atom_lines = parse_cif_ca_only(ca_output_path)
            
            if chain_ca_coords is None or len(chain_ca_coords) == 0:
                print(f"警告: 无法从CA文件中提取点云: {ca_output_path}")
                continue
                
            # 确保转置点云为[3, N]格式，便于应用变换
            if chain_ca_coords.shape[1] == 3 and chain_ca_coords.shape[0] != 3:
                chain_points_t = chain_ca_coords.T
            else:
                chain_points_t = chain_ca_coords
                
            # 生成随机旋转矩阵和平移向量
            rotation = random_rotation_matrix()
            translation = random_translation_vector(scale=100.0)
            
            # 应用变换，生成目标点云
            transformed_target = transform_point_cloud_numpy(chain_points_t, rotation, translation)
            
            # 计算网格信息
            grid_info = create_grid_info()
            
            # 创建并保存PKL文件，包含旋转矩阵和平移向量
            create_pkl_file(chain_points_t, transformed_target, rotation, translation, pkl_path, grid_info)
            pkl_count += 1
            
            # 如果提供了transform_cif_dir，则保存变换后的CIF文件用于验证
            if transform_cif_dir:
                # 确保点云为[N, 3]格式用于保存
                if transformed_target.shape[0] == 3 and transformed_target.shape[1] != 3:
                    transformed_target_for_cif = transformed_target.T
                else:
                    transformed_target_for_cif = transformed_target
                    
                if chain_points_t.shape[0] == 3 and chain_points_t.shape[1] != 3:
                    source_for_cif = chain_points_t.T
                else:
                    source_for_cif = chain_points_t
                
                # 构建输出文件路径（变换后的目标）
                transform_target_path = os.path.join(transform_cif_dir, f"{base_name}_target.cif")
                
                # 构建输出文件路径（原始源）
                transform_source_path = os.path.join(transform_cif_dir, f"{base_name}_source.cif")
                
                # 保存变换后的目标点云为CIF文件
                if cif_content is not None and atom_lines is not None:
                    save_as_cif(transformed_target_for_cif, atom_types, atom_labels, cif_content, atom_lines, transform_target_path)
                    
                    # 保存原始源点云为CIF文件（用于比较）
                    save_as_cif(source_for_cif, atom_types, atom_labels, cif_content, atom_lines, transform_source_path)
                    
                    transform_count += 1
                else:
                    print(f"警告: 无法保存CIF文件，缺少原始CIF内容或行索引信息: {ca_output_path}")
                    
        except Exception as e:
            print(f"处理PKL文件 {pkl_path} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印处理统计
    print(f"成功处理的CA文件数: {ca_count}")
    print(f"成功创建的PKL文件数: {pkl_count}")
    if transform_cif_dir:
        print(f"成功创建的变换CIF文件数: {transform_count}")
        
    return ca_count, pkl_count, transform_count

def process_protein_folder(protein_folder, ca_output_dir, pkl_output_dir, csv_writer, min_ca_atoms, transform_cif_dir=None):
    """处理蛋白质文件夹，提取CA原子，创建PKL文件"""
    protein_name = os.path.basename(protein_folder)
    print(f"\n处理蛋白质: {protein_name}")
    
    # 创建输出目录
    protein_ca_dir = os.path.join(ca_output_dir, os.path.basename(os.path.dirname(protein_folder)), protein_name)
    os.makedirs(protein_ca_dir, exist_ok=True)
    
    # PKL文件直接存放在输出目录，不创建子文件夹
    protein_pkl_dir = pkl_output_dir
    
    # 如果提供了变换CIF输出目录，创建相应的目录
    transform_protein_dir = None
    if transform_cif_dir:
        transform_protein_dir = os.path.join(transform_cif_dir, protein_name)
        os.makedirs(transform_protein_dir, exist_ok=True)
    
    # 查找主CIF文件 (例如 8h1p.cif)
    main_cif_pattern = os.path.join(protein_folder, f"{protein_name.lower()}.cif")
    main_cif_files = glob.glob(main_cif_pattern)
    
    if not main_cif_files:
        print(f"警告: 未找到蛋白质 {protein_name} 的主CIF文件")
        csv_writer.writerow([protein_name, "主CIF文件缺失", "", "", 0, 0])
        return
        
    main_cif = main_cif_files[0]
    
    # 处理主CIF文件，提取CA原子
    main_ca_cif = os.path.join(protein_ca_dir, f"{protein_name}_CA.cif")
    main_ca_count = extract_ca_and_save_cif(main_cif, main_ca_cif)
    
    if main_ca_count <= 0:
        print(f"警告: 主CIF文件 {main_cif} 中没有CA原子或处理出错")
        csv_writer.writerow([protein_name, "主CIF文件无CA原子或处理出错", main_cif, "", main_ca_count, 0])
        return
    
    # 读取原始的CIF文件内容，用于保存变换后的CIF
    main_cif_content = None
    if transform_cif_dir:
        try:
            with open(main_ca_cif, 'r') as f:
                main_cif_content = f.readlines()
        except Exception as e:
            print(f"警告: 读取主CIF文件内容失败: {e}")
    
    # 主CIF的点云数据
    main_points, main_atom_types, main_atom_labels, main_cif_content, main_atom_lines = parse_cif_ca_only(main_ca_cif)
    
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
        
        # 读取链CIF文件内容，用于保存变换后的CIF
        chain_cif_content = None
        if transform_cif_dir:
            try:
                with open(chain_ca_cif, 'r') as f:
                    chain_cif_content = f.readlines()
            except Exception as e:
                print(f"警告: 读取链CIF文件内容失败: {e}")
        
        # 链CIF的点云数据
        chain_points, chain_atom_types, chain_atom_labels, chain_cif_content, chain_atom_lines = parse_cif_ca_only(chain_ca_cif)
        
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
        
        # 生成随机旋转矩阵和平移向量
        rotation = random_rotation_matrix()
        translation = random_translation_vector(scale=100.0)  # -100到100范围
        
        # 应用变换到目标点云
        transformed_target = transform_point_cloud_numpy(main_points_t, rotation, translation)
        
        # 打印变换信息
        print("变换矩阵信息:")
        print("旋转矩阵:")
        print(rotation)
        print("平移向量:")
        print(translation)
        
        # 创建网格信息
        grid_info = create_grid_info()
        
        # 创建并保存PKL文件 - 注意，现在传递rotation和translation参数
        create_pkl_file(chain_points_t, transformed_target, rotation, translation, pkl_path, grid_info)
        pkl_count += 1
        
        # 如果提供了transform_cif_dir，则保存变换后的CIF文件用于验证
        if transform_protein_dir and main_cif_content is not None and chain_cif_content is not None:
            # 将变换后的目标点云转换为[N, 3]格式用于保存CIF
            transformed_target_cif = transformed_target.T if transformed_target.shape[0] == 3 else transformed_target
            
            # 保存变换后的目标点云为CIF
            target_transformed_cif = os.path.join(transform_protein_dir, f"{protein_name}_{chain_id}_target_transformed.cif")
            save_as_cif(
                transformed_target_cif,
                main_atom_types,
                main_atom_labels,
                main_cif_content,
                main_atom_lines,
                target_transformed_cif
            )
            
            # 应用相同的变换到源点云
            transformed_source = transform_point_cloud_numpy(chain_points_t, rotation, translation)
            
            # 将变换后的源点云转换为[N, 3]格式用于保存CIF
            transformed_source_cif = transformed_source.T if transformed_source.shape[0] == 3 else transformed_source
            
            # 保存变换后的源点云为CIF
            source_transformed_cif = os.path.join(transform_protein_dir, f"{protein_name}_{chain_id}_source_transformed.cif")
            save_as_cif(
                transformed_source_cif,
                chain_atom_types,
                chain_atom_labels,
                chain_cif_content,
                chain_atom_lines,
                source_transformed_cif
            )
            
            print(f"    已保存变换后的CIF文件用于验证")
    
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
    parser.add_argument('--transform_cif_dir', help='变换后CIF文件的输出目录')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.ca_output_dir, exist_ok=True)
    os.makedirs(args.pkl_output_dir, exist_ok=True)
    if args.transform_cif_dir:
        os.makedirs(args.transform_cif_dir, exist_ok=True)
    
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
                    protein_path, args.ca_output_dir, args.pkl_output_dir, csv_writer, args.min_ca_atoms, args.transform_cif_dir
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
