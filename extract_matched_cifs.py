#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import argparse
import numpy as np
from collections import defaultdict
import random

def extract_protein_id(filename):
    """从文件名中提取蛋白质ID"""
    base_name = os.path.basename(filename)
    # 处理类似6c24_backbone.cif或6c24_segment.cif的文件名
    if "_" in base_name:
        return base_name.split("_")[0].lower()
    else:
        # 如果没有下划线，则移除扩展名
        return os.path.splitext(base_name)[0].lower()

def count_ca_atoms(cif_file):
    """计算CIF文件中的CA原子数量"""
    # 尝试不同的文件编码
    encodings = ['utf-8', 'gbk', 'latin1', 'ascii']
    
    for encoding in encodings:
        try:
            ca_count = 0
            atom_section = False
            atom_found = False
            
            with open(cif_file, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    
                    # 标记找到了原子部分
                    if 'ATOM' in line or '_atom_site.' in line:
                        atom_found = True
                    
                    # 检测原子记录部分的开始和结束
                    if line.startswith('_atom_site.') or line.startswith('loop_'):
                        atom_section = True
                    elif line.startswith('#') and atom_section:
                        atom_section = False
                    
                    # 更精确地检测 CA 原子
                    if atom_section and line and not line.startswith('_') and not line.startswith('#'):
                        # 分割行成字段
                        fields = line.split()
                        if len(fields) >= 3:
                            # 检查是否是CA原子，可能的位置很多
                            for i, field in enumerate(fields):
                                if (field == 'CA' or field == '"CA"' or 
                                    field == "'CA'" or field == 'CA.' or
                                    field == 'CA"' or field == "CA'"):
                                    # 找到CA原子
                                    ca_count += 1
                                    break
                                
                                # 有时CA原子可能以" CA "形式出现
                                if '" CA "' in field or ' CA ' in field or ".CA." in field:
                                    ca_count += 1
                                    break
            
            # 如果没有找到原子记录，可能还有其他方式来记录CA原子
            if ca_count == 0 and atom_found:
                with open(cif_file, 'r', encoding=encoding) as f:
                    content = f.read()
                    # 检查ATOM记录行中是否有CA原子
                    ca_count = content.count(' CA ') + content.count("'CA'") + content.count('"CA"')
            
            return ca_count
            
        except UnicodeDecodeError:
            # 尝试下一种编码
            continue
        except Exception as e:
            print(f"  警告: 使用{encoding}编码解析文件 {os.path.basename(cif_file)} 时出错: {str(e)}")
            continue
    
    # 所有编码都失败了
    print(f"  错误: 无法解析文件 {os.path.basename(cif_file)}")
    return 0

def read_cif_coordinates(cif_file):
    """读取CIF文件中的原子坐标"""
    coordinates = []
    encodings = ['utf-8', 'gbk', 'latin1', 'ascii']
    
    for encoding in encodings:
        try:
            atom_section = False
            with open(cif_file, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    
                    # 检测原子记录部分的开始和结束
                    if line.startswith('_atom_site.') or line.startswith('loop_'):
                        atom_section = True
                    elif line.startswith('#') and atom_section:
                        atom_section = False
                    
                    # 解析原子坐标
                    if atom_section and line and not line.startswith('_') and not line.startswith('#'):
                        fields = line.split()
                        if len(fields) >= 10:  # 确保有足够的字段
                            try:
                                # 尝试提取X, Y, Z坐标
                                # 不同的CIF文件可能坐标位置不同，这里假设最后三个数值是坐标
                                x = float(fields[-3])
                                y = float(fields[-2])
                                z = float(fields[-1])
                                coordinates.append([x, y, z])
                            except (ValueError, IndexError):
                                continue
            
            if coordinates:
                return np.array(coordinates)
                
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  警告: 读取坐标时出错: {str(e)}")
            continue
    
    return np.array([])

def write_cif_with_coords(source_file, target_file, coords):
    """写入新的CIF文件，使用给定的坐标"""
    encodings = ['utf-8', 'gbk', 'latin1', 'ascii']
    
    for encoding in encodings:
        try:
            # 读取原CIF文件
            with open(source_file, 'r', encoding=encoding) as f:
                content = f.readlines()
            
            # 创建新CIF文件
            with open(target_file, 'w', encoding='utf-8') as f:
                atom_section = False
                coord_index = 0
                
                for line in content:
                    line_strip = line.strip()
                    
                    # 检测原子记录部分
                    if line_strip.startswith('_atom_site.') or line_strip.startswith('loop_'):
                        atom_section = True
                        f.write(line)
                        continue
                    elif line_strip.startswith('#') and atom_section:
                        atom_section = False
                        f.write(line)
                        continue
                    
                    # 处理原子坐标行
                    if (atom_section and line_strip and 
                        not line_strip.startswith('_') and 
                        not line_strip.startswith('#')):
                        
                        if coord_index < len(coords):
                            fields = line.split()
                            if len(fields) >= 10:  # 确保有足够的字段
                                try:
                                    # 替换最后三个数值为新坐标
                                    x, y, z = coords[coord_index]
                                    
                                    # 找到坐标字段的位置
                                    new_fields = list(fields)
                                    new_fields[-3] = f"{x:.3f}"
                                    new_fields[-2] = f"{y:.3f}"
                                    new_fields[-1] = f"{z:.3f}"
                                    
                                    # 重建行
                                    new_line = " ".join(new_fields) + "\n"
                                    f.write(new_line)
                                    coord_index += 1
                                    continue
                                except (ValueError, IndexError):
                                    f.write(line)
                                    continue
                    
                    # 写入其他行
                    f.write(line)
            
            return True
                
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  警告: 写入CIF文件时出错: {str(e)}")
            continue
    
    return False

def farthest_point_sampling(points, npoint):
    """
    使用最远点采样算法从点云中选择代表性的点
    
    参数:
        points: 输入点云, 形状为 (N, 3)
        npoint: 要采样的点数量
    
    返回:
        采样后的点云, 形状为 (npoint, 3)
    """
    N = points.shape[0]
    if N <= npoint:
        return points  # 如果原始点数少于目标点数，直接返回
    
    # 初始化距离和索引
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    
    # 随机选择第一个点
    farthest = np.random.randint(0, N)
    
    # 迭代选择最远点
    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest, :]
        
        # 计算所有点到当前点的距离
        dist = np.sum((points - centroid) ** 2, axis=1)
        
        # 更新最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # 选择距离最大的点作为下一个中心
        farthest = np.argmax(distance)
    
    # 返回采样后的点
    return points[centroids]

def extract_and_process_pairs(folder1, folder2, output_folder1, output_folder2, max_points=0, use_fps=False):
    """
    提取两个文件夹中对应的CIF文件对，可选使用最远点采样平衡点数
    
    参数:
        folder1: 第一个CIF文件夹路径
        folder2: 第二个CIF文件夹路径
        output_folder1: 第一个输出文件夹路径
        output_folder2: 第二个输出文件夹路径
        max_points: 最大点数阈值，高于此值的配对将被忽略
        use_fps: 是否使用最远点采样平衡点数
    """
    # 创建输出目录
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)
    
    # 获取两个文件夹中的所有CIF文件
    folder1_files = glob.glob(os.path.join(folder1, "*.cif"))
    folder2_files = glob.glob(os.path.join(folder2, "*.cif"))
    
    # 提取蛋白质ID
    folder1_ids = {extract_protein_id(f): f for f in folder1_files}
    folder2_ids = {extract_protein_id(f): f for f in folder2_files}
    
    # 找出共有的蛋白质ID
    common_ids = sorted(set(folder1_ids.keys()) & set(folder2_ids.keys()))
    print(f"发现{len(common_ids)}对匹配的CIF文件")
    
    # 统计结果
    processed_pairs = 0
    skipped_pairs = 0
    fps_applied_pairs = 0
    
    # 处理每一对CIF文件
    for protein_id in common_ids:
        file1 = folder1_ids[protein_id]
        file2 = folder2_ids[protein_id]
        
        # 计算CA原子数量
        points1 = count_ca_atoms(file1)
        points2 = count_ca_atoms(file2)
        
        # 如果任一文件点数高于阈值，则跳过
        if max_points > 0 and max(points1, points2) > max_points:
            print(f"  跳过 {protein_id}: 点数 {points1} vs {points2} 高于阈值 {max_points}")
            skipped_pairs += 1
            continue
        
        # 检查点数是否相同
        if points1 != points2 and use_fps:
            print(f"  {protein_id}: 点数不一致 {points1} vs {points2}，应用最远点采样...")
            
            # 确定哪个文件点数更多
            if points1 > points2:
                # 文件1点数更多，需要降采样
                source_file = file1
                target_points = points2
                coords = read_cif_coordinates(file1)
                
                if len(coords) > 0:
                    # 应用最远点采样
                    sampled_coords = farthest_point_sampling(coords, target_points)
                    
                    # 创建新的目标文件路径
                    new_file1 = os.path.join(output_folder1, os.path.basename(file1))
                    
                    # 写入新的CIF文件
                    if write_cif_with_coords(file1, new_file1, sampled_coords):
                        # 复制文件2到输出目录
                        shutil.copy2(file2, os.path.join(output_folder2, os.path.basename(file2)))
                        print(f"  ✅ {protein_id}: 已从 {points1} 采样至 {target_points} 点")
                        fps_applied_pairs += 1
                        processed_pairs += 1
                        continue
                    else:
                        print(f"  ❌ {protein_id}: 写入采样后文件失败")
                        continue
                else:
                    print(f"  ❌ {protein_id}: 无法读取坐标")
                    continue
                    
            else:
                # 文件2点数更多，需要降采样
                source_file = file2
                target_points = points1
                coords = read_cif_coordinates(file2)
                
                if len(coords) > 0:
                    # 应用最远点采样
                    sampled_coords = farthest_point_sampling(coords, target_points)
                    
                    # 创建新的目标文件路径
                    new_file2 = os.path.join(output_folder2, os.path.basename(file2))
                    
                    # 写入新的CIF文件
                    if write_cif_with_coords(file2, new_file2, sampled_coords):
                        # 复制文件1到输出目录
                        shutil.copy2(file1, os.path.join(output_folder1, os.path.basename(file1)))
                        print(f"  ✅ {protein_id}: 已从 {points2} 采样至 {target_points} 点")
                        fps_applied_pairs += 1
                        processed_pairs += 1
                        continue
                    else:
                        print(f"  ❌ {protein_id}: 写入采样后文件失败")
                        continue
                else:
                    print(f"  ❌ {protein_id}: 无法读取坐标")
                    continue
        
        # 如果点数相同或不需要最远点采样，直接复制文件
        shutil.copy2(file1, os.path.join(output_folder1, os.path.basename(file1)))
        shutil.copy2(file2, os.path.join(output_folder2, os.path.basename(file2)))
        print(f"  ✅ {protein_id}: 点数相同 {points1} vs {points2}，直接复制")
        processed_pairs += 1
    
    # 输出统计信息
    print("\n处理完成，统计信息:")
    print(f"总共找到 {len(common_ids)} 对匹配的CIF文件")
    print(f"成功处理 {processed_pairs} 对")
    print(f"跳过 {skipped_pairs} 对（点数低于阈值）")
    print(f"应用最远点采样 {fps_applied_pairs} 对")

def main():
    parser = argparse.ArgumentParser(description="提取并处理匹配的CIF文件对")
    parser.add_argument("--folder1", type=str, required=True, help="第一个CIF文件夹路径")
    parser.add_argument("--folder2", type=str, required=True, help="第二个CIF文件夹路径")
    parser.add_argument("--output1", type=str, required=True, help="第一个输出文件夹路径")
    parser.add_argument("--output2", type=str, required=True, help="第二个输出文件夹路径")
    parser.add_argument("--max_points", type=int, default=0, help="最大点数阈值，高于此值的配对将被忽略（默认0表示不设限制）")
    parser.add_argument("--use_fps", action="store_true", help="使用最远点采样平衡点数")
    
    args = parser.parse_args()
    
    # 检查输入文件夹是否存在
    if not os.path.exists(args.folder1):
        print(f"错误: 文件夹 {args.folder1} 不存在!")
        return 1
    
    if not os.path.exists(args.folder2):
        print(f"错误: 文件夹 {args.folder2} 不存在!")
        return 1
    
    # 提取并处理CIF文件对
    extract_and_process_pairs(
        args.folder1, args.folder2, 
        args.output1, args.output2, 
        args.max_points, args.use_fps
    )
    
    return 0

if __name__ == "__main__":
    main()
