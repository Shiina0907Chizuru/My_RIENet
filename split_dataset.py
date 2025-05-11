#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
import random
from tqdm import tqdm

def split_dataset(input_dir, output_train_dir, output_test_dir, train_ratio=0.8, random_seed=42, copy_mode=True):
    """
    将输入目录中的PKL文件随机划分为训练集和测试集
    
    参数:
        input_dir: 包含PKL文件的输入目录
        output_train_dir: 训练集输出目录
        output_test_dir: 测试集输出目录
        train_ratio: 训练集的比例（默认0.8，即80%为训练集）
        random_seed: 随机种子，确保结果可重现
        copy_mode: 如果为True，复制文件；如果为False，移动文件
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 确保输出目录存在
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)
    
    # 获取所有PKL文件
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    
    if not pkl_files:
        print(f"警告: 在 {input_dir} 中未找到PKL文件")
        return

    # 随机打乱文件列表
    random.shuffle(pkl_files)
    
    # 计算训练集大小
    train_size = int(len(pkl_files) * train_ratio)
    
    # 划分数据集
    train_files = pkl_files[:train_size]
    test_files = pkl_files[train_size:]
    
    print(f"总共找到 {len(pkl_files)} 个PKL文件")
    print(f"训练集: {len(train_files)} 个文件 ({train_ratio*100:.1f}%)")
    print(f"测试集: {len(test_files)} 个文件 ({(1-train_ratio)*100:.1f}%)")
    
    # 复制/移动训练集文件
    print("处理训练集文件...")
    for file in tqdm(train_files):
        src_path = os.path.join(input_dir, file)
        dst_path = os.path.join(output_train_dir, file)
        if copy_mode:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)
    
    # 复制/移动测试集文件
    print("处理测试集文件...")
    for file in tqdm(test_files):
        src_path = os.path.join(input_dir, file)
        dst_path = os.path.join(output_test_dir, file)
        if copy_mode:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)
    
    print(f"数据集划分完成!")
    print(f"训练集保存至: {output_train_dir}")
    print(f"测试集保存至: {output_test_dir}")

def main():
    parser = argparse.ArgumentParser(description='将PKL文件随机划分为训练集和测试集')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='包含PKL文件的输入目录')
    parser.add_argument('--output_train_dir', type=str, required=True,
                        help='训练集输出目录')
    parser.add_argument('--output_test_dir', type=str, required=True,
                        help='测试集输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集的比例（默认：0.8）')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子（默认：42）')
    parser.add_argument('--move', action='store_true',
                        help='移动文件而非复制')
    
    args = parser.parse_args()
    
    split_dataset(
        args.input_dir,
        args.output_train_dir,
        args.output_test_dir,
        args.train_ratio,
        args.random_seed,
        not args.move
    )

if __name__ == '__main__':
    main()
