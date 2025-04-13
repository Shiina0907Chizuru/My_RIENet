#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from tqdm import tqdm
import shutil

def count_atoms_in_cif(cif_path):
    """统计CIF文件中的原子数量"""
    atom_count = 0
    atom_types = {}
    
    try:
        with open(cif_path, 'r') as f:
            lines = f.readlines()
        
        # 解析PDB格式的CIF文件，查找原子行（以ATOM开头）
        for line in lines:
            line = line.strip()
            if line.startswith('ATOM'):
                atom_count += 1
                
                # 统计不同原子类型的数量
                parts = line.split()
                if len(parts) >= 4:  # 确保有足够的列
                    atom_type = parts[3]  # 原子类型 (如 CA)
                    atom_types[atom_type] = atom_types.get(atom_type, 0) + 1
        
        return {
            'total': atom_count,
            'types': atom_types
        }
    except Exception as e:
        print(f"处理文件 {cif_path} 时出错: {e}")
        return {'total': 0, 'types': {}}

def process_single_file(cif_path, threshold, output_file=None, copy_dir=None):
    """处理单个CIF文件并输出结果"""
    filename = os.path.basename(cif_path)
    result = count_atoms_in_cif(cif_path)
    atom_count = result['total']
    
    # 判断原子数量是否超过阈值
    exceeds_threshold = atom_count > threshold
    status = "超过" if exceeds_threshold else "未超过"
    
    # 构建详细信息
    details = f"文件: {filename}\n"
    details += f"原子总数: {atom_count}\n"
    details += f"阈值: {threshold}\n"
    details += f"状态: {status}\n"
    
    # 添加原子类型统计
    if result['types']:
        details += "原子类型统计:\n"
        for atom_type, count in sorted(result['types'].items(), key=lambda x: x[1], reverse=True):
            details += f"  {atom_type}: {count}\n"
    
    details += "=" * 50 + "\n"
    
    # 打印到控制台
    print(details)
    
    # 如果指定了输出文件，则写入文件
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(details)
    
    # 如果指定了复制目录，并且原子数量低于阈值，则复制文件
    if copy_dir and not exceeds_threshold:
        try:
            destination_path = os.path.join(copy_dir, filename)
            shutil.copy2(cif_path, destination_path)
            print(f"已复制文件到: {destination_path}")
            
            # 在报告中添加复制信息
            copy_info = f"文件已复制到: {destination_path}\n"
            if output_file:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(copy_info)
        except Exception as e:
            error_msg = f"复制文件时出错: {e}\n"
            print(error_msg)
            if output_file:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg)
    
    return {
        'filename': filename,
        'atom_count': atom_count,
        'exceeds_threshold': exceeds_threshold
    }

def process_batch(input_dir, threshold, output_file, copy_dir=None):
    """批量处理目录中的CIF文件"""
    print(f"开始批处理目录: {input_dir}")
    
    # 确保输出文件所在目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 如果指定了复制目录，确保目录存在
    if copy_dir and not os.path.exists(copy_dir):
        os.makedirs(copy_dir)
        print(f"创建复制目标目录: {copy_dir}")
    
    # 清空输出文件（如果存在）
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"CIF文件原子数量分析报告 (阈值: {threshold})\n")
        if copy_dir:
            f.write(f"低于阈值的文件将被复制到: {copy_dir}\n")
        f.write("=" * 50 + "\n\n")
    
    # 搜索所有符合条件的CIF文件（递归查找子目录）
    cif_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            # 只处理名字为xxxx_point.cif格式的文件
            if file.endswith('_point.cif'):
                cif_files.append(os.path.join(root, file))
    
    if not cif_files:
        print(f"警告: 在{input_dir}中未找到符合条件的xxxx_point.cif文件")
        return
    
    # 处理结果统计
    results = {
        'total': 0,
        'exceeds': 0,
        'within': 0,
        'copied': 0
    }
    
    # 处理每个CIF文件
    for cif_path in tqdm(cif_files, desc="处理CIF文件"):
        result = process_single_file(cif_path, threshold, output_file, copy_dir)
        results['total'] += 1
        if result['exceeds_threshold']:
            results['exceeds'] += 1
        else:
            results['within'] += 1
            if copy_dir:
                results['copied'] += 1
    
    # 写入统计摘要
    summary = f"\n总结报告:\n"
    summary += f"总共处理CIF文件: {results['total']}\n"
    summary += f"超过阈值({threshold})的文件数: {results['exceeds']}\n"
    summary += f"未超过阈值的文件数: {results['within']}\n"
    if copy_dir:
        summary += f"已复制到{copy_dir}的文件数: {results['copied']}\n"
    
    # 打印到控制台
    print(summary)
    
    # 写入到输出文件
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"分析报告已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='分析CIF文件中的原子数量')
    parser.add_argument('input_path', type=str, help='输入的CIF文件或包含CIF文件的目录')
    parser.add_argument('--threshold', type=int, default=1000, help='原子数量阈值，默认为1000')
    parser.add_argument('--output', type=str, default='cif_atom_analysis.txt', help='输出报告文件路径')
    parser.add_argument('--copy_dir', type=str, help='复制低于阈值的CIF文件到指定目录')
    
    args = parser.parse_args()
    
    # 检查输入路径是否存在
    if not os.path.exists(args.input_path):
        print(f"错误：路径 {args.input_path} 不存在")
        return
    
    # 如果指定了复制目录，确保目录存在
    if args.copy_dir and not os.path.exists(args.copy_dir):
        os.makedirs(args.copy_dir)
        print(f"创建复制目标目录: {args.copy_dir}")
    
    # 判断输入是文件还是目录
    if os.path.isfile(args.input_path):
        # 单文件处理模式
        filename = os.path.basename(args.input_path)
        # 检查文件名是否符合xxxx_point.cif格式
        if not filename.endswith('_point.cif'):
            print(f"错误：文件 {args.input_path} 不符合xxxx_point.cif格式")
            return
        
        # 清空输出文件（如果存在）
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"CIF文件原子数量分析报告 (阈值: {args.threshold})\n")
            if args.copy_dir:
                f.write(f"低于阈值的文件将被复制到: {args.copy_dir}\n")
            f.write("=" * 50 + "\n\n")
        
        process_single_file(args.input_path, args.threshold, args.output, args.copy_dir)
        print(f"分析报告已保存到: {args.output}")
    else:
        # 目录处理模式
        process_batch(args.input_path, args.threshold, args.output, args.copy_dir)

if __name__ == '__main__':
    main()
