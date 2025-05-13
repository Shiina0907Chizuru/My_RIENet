#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from collections import defaultdict

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
            
            print(f"  文件 {os.path.basename(cif_file)} 中检测到 {ca_count} 个CA原子 [编码: {encoding}]")
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

def compare_cif_folders(folder1, folder2, output_file):
    """比较两个文件夹中的CIF文件"""
    # 获取两个文件夹中的所有CIF文件
    backbone_files = glob.glob(os.path.join(folder1, "*.cif"))
    segment_files = glob.glob(os.path.join(folder2, "*.cif"))
    
    # 提取蛋白质ID
    backbone_ids = {extract_protein_id(f): f for f in backbone_files}
    segment_ids = {extract_protein_id(f): f for f in segment_files}
    
    # 找出不成对的蛋白质
    backbone_only = set(backbone_ids.keys()) - set(segment_ids.keys())
    segment_only = set(segment_ids.keys()) - set(backbone_ids.keys())
    
    # 统计结果
    results = []
    
    # 输出不成对的蛋白质
    print("\n未找到成对CIF文件的蛋白质:")
    if backbone_only:
        print(f"  只在{folder1}中发现的蛋白质: {', '.join(sorted(backbone_only))}")
        results.append(f"只在{folder1}中发现的蛋白质: {', '.join(sorted(backbone_only))}")
    if segment_only:
        print(f"  只在{folder2}中发现的蛋白质: {', '.join(sorted(segment_only))}")
        results.append(f"只在{folder2}中发现的蛋白质: {', '.join(sorted(segment_only))}")
    
    # 比较每对文件中的CA原子数量
    print("\n统计每对文件的CA原子数量:")
    results.append("\n统计每对文件的CA原子数量:")
    
    # 存储格式化的结果
    different_pairs = []
    ca_counts = []
    
    # 对共有的蛋白质ID进行排序
    common_ids = sorted(set(backbone_ids.keys()) & set(segment_ids.keys()))
    
    for protein_id in common_ids:
        backbone_file = backbone_ids[protein_id]
        segment_file = segment_ids[protein_id]
        
        # 计算CA原子数量
        backbone_ca_count = count_ca_atoms(backbone_file)
        segment_ca_count = count_ca_atoms(segment_file)
        
        # 检查是否存在差异
        is_different = backbone_ca_count != segment_ca_count
        
        # 存储结果
        result_str = f"{protein_id}: {backbone_ca_count} vs {segment_ca_count}"
        if is_different:
            result_str += " [不一致]"
            different_pairs.append(protein_id)
        ca_counts.append(result_str)
    
    # 输出并保存结果
    for result in ca_counts:
        print(f"  {result}")
        results.append(result)
    
    # 输出不一致的总结
    if different_pairs:
        diff_summary = f"\n以下{len(different_pairs)}个蛋白质的CA原子数量不一致: {', '.join(different_pairs)}"
        print(diff_summary)
        results.append(diff_summary)
    else:
        print("\n所有蛋白质的CA原子数量都一致")
        results.append("\n所有蛋白质的CA原子数量都一致")
    
    # 保存结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    
    print(f"\n结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="比较两个文件夹中的CIF文件")
    parser.add_argument("--folder1", type=str, required=True, help="第一个CIF文件夹路径(backbone)")
    parser.add_argument("--folder2", type=str, required=True, help="第二个CIF文件夹路径(segment)")
    parser.add_argument("--output", type=str, default="cif_comparison_results.txt", help="输出文件路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder1):
        print(f"错误: 文件夹 {args.folder1} 不存在!")
        return 1
    
    if not os.path.exists(args.folder2):
        print(f"错误: 文件夹 {args.folder2} 不存在!")
        return 1
    
    compare_cif_folders(args.folder1, args.folder2, args.output)
    return 0

if __name__ == "__main__":
    main()