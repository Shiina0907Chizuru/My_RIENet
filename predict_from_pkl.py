#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pickle
from model import RIENET
from util import transform_point_cloud
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
from easydict import EasyDict

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

def parse_cif(cif_path):
    """解析CIF文件，提取原子坐标和完整CIF内容"""
    print(f"正在解析CIF文件: {cif_path}")
    
    atoms = []
    atom_lines = []
    atom_types = []
    atom_labels = []
    cif_content = []
    
    with open(cif_path, 'r') as f:
        lines = f.readlines()
    
    # 保存CIF文件的完整内容
    cif_content = lines.copy()
    
    # 解析PDB格式的CIF文件
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 检测原子行（以ATOM开头）
        if line.startswith('ATOM'):
            parts = line.split()
            if len(parts) >= 13:  # 确保有足够的列
                try:
                    atom_number = parts[1]
                    atom_type = parts[3]  # CA 原子类型
                    atom_label = parts[5]  # 残基名称（如 ALA）
                    
                    # 获取坐标（第10、11、12列，空格分隔）
                    x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                    
                    atoms.append([x, y, z])
                    atom_types.append(atom_type)
                    atom_labels.append(atom_label)
                    atom_lines.append(i)
                except (ValueError, IndexError) as e:
                    print(f"警告：无法解析第 {i+1} 行: {e}")
                    continue
    
    # 检查是否成功提取到原子坐标
    if not atoms:
        raise ValueError("未能从CIF文件中提取到原子坐标")
    
    print(f"成功解析 {len(atoms)} 个原子")
    
    # 转换为numpy数组
    return np.array(atoms, dtype=np.float32), atom_types, atom_labels, cif_content, atom_lines

class PredictionDataset:
    def __init__(self, pkl_path):
        """加载PKL文件，提取源点云和目标点云数据"""
        # 加载数据
        self.pkl_path = pkl_path
        print(f"正在加载PKL文件: {pkl_path}")
        
        # 加载PKL文件
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 提取点云数据 - 检查不同的可能键名
        if 'source' in data:
            self.source = data['source'].clone().detach() if isinstance(data['source'], torch.Tensor) else torch.tensor(data['source'], dtype=torch.float32)
        elif 'source_point_cloud' in data:
            self.source = data['source_point_cloud'].clone().detach() if isinstance(data['source_point_cloud'], torch.Tensor) else torch.tensor(data['source_point_cloud'], dtype=torch.float32)
        else:
            raise ValueError("在PKL文件中找不到源点云数据")
            
        # 检查是否存在目标点云，如果不存在，使用源点云代替
        if 'target' in data:
            self.target = data['target'].clone().detach() if isinstance(data['target'], torch.Tensor) else torch.tensor(data['target'], dtype=torch.float32)
        elif 'target_point_cloud' in data:
            self.target = data['target_point_cloud'].clone().detach() if isinstance(data['target_point_cloud'], torch.Tensor) else torch.tensor(data['target_point_cloud'], dtype=torch.float32)
        else:
            print("警告：PKL文件中不包含目标点云数据，将使用源点云作为替代")
            self.target = self.source.clone()
            
        # 尝试提取变换矩阵（如果有）- 仅用于参考
        if 'rotation' in data:
            self.rotation = data['rotation'].clone().detach() if isinstance(data['rotation'], torch.Tensor) else torch.tensor(data['rotation'], dtype=torch.float32)
        else:
            self.rotation = None
            
        if 'translation' in data:
            self.translation = data['translation'].clone().detach() if isinstance(data['translation'], torch.Tensor) else torch.tensor(data['translation'], dtype=torch.float32)
        else:
            self.translation = None
            
        # 提取网格信息
        if 'grid_shape' in data:
            self.grid_shape = data['grid_shape']
        else:
            # 默认网格形状
            self.grid_shape = np.array([32, 32, 32], dtype=np.int32)
            
        if 'x_origin' in data:
            self.x_origin = data['x_origin']
        else:
            self.x_origin = -1.0
            
        if 'y_origin' in data:
            self.y_origin = data['y_origin']
        else:
            self.y_origin = -1.0
            
        if 'z_origin' in data:
            self.z_origin = data['z_origin']
        else:
            self.z_origin = -1.0
            
        if 'x_voxel' in data:
            self.x_voxel = data['x_voxel']
        else:
            self.x_voxel = 2.0 / self.grid_shape[0]
            
        if 'nstart' in data:
            self.nstart = data['nstart']
        else:
            self.nstart = np.array([0], dtype=np.int32)
            
        # 检查是否有归一化信息
        self.is_normalized = False
        self.scale_factor = 1.0
        self.original_centroid = np.zeros(3, dtype=np.float32)
        
        # 检查grid_info中是否包含normalized键（新格式）
        if 'grid_info' in data and isinstance(data['grid_info'], dict):
            grid_info = data['grid_info']
            if 'normalized' in grid_info and grid_info['normalized']:
                self.is_normalized = True
                self.scale_factor = float(grid_info['scale_factor'])
                self.original_centroid = np.array(grid_info['original_centroid'], dtype=np.float32)
                print(f"检测到归一化数据, 缩放因子: {self.scale_factor}, 原始质心: {self.original_centroid}")
        # 直接检查顶层字典是否包含normalized键（旧格式）
        elif 'normalized' in data and data['normalized']:
            self.is_normalized = True
            self.scale_factor = float(data['scale_factor'])
            self.original_centroid = np.array(data['original_centroid'], dtype=np.float32)
            print(f"检测到归一化数据(旧格式), 缩放因子: {self.scale_factor}, 原始质心: {self.original_centroid}")
        
        # 确保点云形状正确（[3, N]格式）
        if self.source.shape[0] != 3 and self.source.shape[1] == 3:
            print(f"转置源点云形状: 从{self.source.shape}到", end="")
            self.source = self.source.T
            print(f"{self.source.shape}")
            
        if self.target.shape[0] != 3 and self.target.shape[1] == 3:
            print(f"转置目标点云形状: 从{self.target.shape}到", end="")
            self.target = self.target.T
            print(f"{self.target.shape}")
            
        # 输出形状信息
        print(f"源点云形状: {self.source.shape}")
        print(f"目标点云形状: {self.target.shape}")
        if self.rotation is not None:
            print(f"参考旋转矩阵形状: {self.rotation.shape}")
        if self.translation is not None:
            print(f"参考平移向量形状: {self.translation.shape}")
        print(f"网格信息: grid_shape={self.grid_shape}")
        if self.is_normalized:
            print(f"点云已归一化: scale_factor={self.scale_factor}, original_centroid={self.original_centroid}")
            
    def __len__(self):
        return 1  # 每个PKL文件只有一对点云
        
    def __getitem__(self, idx):
        # 返回点云数据和其他必要的参数
        if idx != 0:
            raise IndexError("索引超出范围")
            
        # 返回归一化信息
        normalization_info = {
            'is_normalized': self.is_normalized,
            'scale_factor': self.scale_factor,
            'original_centroid': self.original_centroid
        }
        
        return self.source, self.target, self.grid_shape, self.x_origin, self.y_origin, self.z_origin, self.x_voxel, self.nstart, normalization_info

def predict_transformation(args, net, dataset):
    """使用模型预测变换，不计算与真实值的loss"""
    net.eval()
    
    with torch.no_grad():
        # 获取源点云数据和目标点云数据以及其他需要的参数
        source, target, grid_shape, x_origin, y_origin, z_origin, x_voxel, nstart, normalization_info = dataset[0]
        
        # 确保数据格式正确 [B, 3, N]
        if source.shape[0] != 3:
            source = source.transpose(0, 1)
        if target.shape[0] != 3:
            target = target.transpose(0, 1)
            
        # 添加批次维度
        source = source.unsqueeze(0).cuda()
        target = target.unsqueeze(0).cuda()
        
        # 打印归一化信息
        is_normalized = normalization_info['is_normalized']
        if is_normalized:
            print(f"处理归一化数据，缩放因子: {normalization_info['scale_factor']}")
            print(f"原始质心位置: {normalization_info['original_centroid']}")
        
        # 使用模型预测变换
        rotation_pred, translation_pred, _, _, _, _ = net(source, target, grid_shape, x_origin, y_origin, z_origin, x_voxel, nstart)
        
        # 转换为numpy数组
        rotation_pred_np = rotation_pred.cpu().numpy()[0]
        translation_pred_np = translation_pred.cpu().numpy()[0]
        
        # 如果点云是归一化的，需要还原平移向量（旋转矩阵保持不变）
        if is_normalized:
            scale_factor = normalization_info['scale_factor']
            # 对于归一化数据，平移向量需要乘以缩放因子来还原到原始坐标系
            translation_pred_np = translation_pred_np * scale_factor
            print(f"已将平移向量从归一化空间还原到原始坐标系，乘以缩放因子: {scale_factor}")
        
        print("预测的旋转矩阵:")
        print(rotation_pred_np)
        print("预测的平移向量:")
        print(translation_pred_np)
        
    return rotation_pred_np, translation_pred_np, normalization_info

def save_as_cif(coordinates, atom_types, atom_labels, original_cif, atom_lines, output_path):
    """将点云坐标保存为CIF格式"""
    new_cif = original_cif.copy()
    
    # 更新原子坐标
    for i, (coord, line_idx) in enumerate(zip(coordinates, atom_lines)):
        x, y, z = coord
        line = original_cif[line_idx].strip().split()
        
        # 保持原始行的格式，只更新坐标部分
        line[10] = f"{x:.6f}"
        line[11] = f"{y:.6f}"
        line[12] = f"{z:.6f}"
        
        # 重新构建行
        new_line = " ".join(line)
        new_cif[line_idx] = new_line + "\n"
    
    # 写入到新文件
    with open(output_path, 'w') as f:
        f.writelines(new_cif)
    
    print(f"已保存CIF文件: {output_path}")

def save_as_cif_direct(coordinates, output_path, chain_id='A', residue_name='ALA', atom_name='CA'):
    """
    直接从点云坐标生成CIF文件，不需要参考CIF文件
    格式与pdb2cif.py生成的CIF文件一致
    
    参数:
        coordinates: 点云坐标，格式为[N, 3]
        output_path: 输出CIF文件路径
        chain_id: 链ID，默认为'A'
        residue_name: 残基名称，默认为'ALA'
        atom_name: 原子名称，默认为'CA'
    """
    # 确保点云是[N, 3]格式
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

def save_results_as_cif(dataset, source_points, target_points, predicted_points, 
                         atom_types=None, atom_labels=None, cif_content=None, atom_lines=None, output_dir=None):
    """将结果保存为CIF格式，支持有参考CIF和无参考CIF两种模式"""
    # 获取源文件名（不带路径和扩展名）
    base_name = os.path.splitext(os.path.basename(dataset.pkl_path))[0]
    
    # 创建保存CIF文件的路径
    source_cif_path = os.path.join(output_dir, f"{base_name}_source.cif")
    target_cif_path = os.path.join(output_dir, f"{base_name}_target.cif")
    predicted_cif_path = os.path.join(output_dir, f"{base_name}_predicted.cif")
    
    # 确保点云格式为[N, 3] - 转换如果需要
    if isinstance(source_points, torch.Tensor):
        source_points = source_points.cpu().numpy()
    if source_points.shape[0] == 3 and source_points.shape[1] != 3:
        source_points = source_points.T
        
    if isinstance(target_points, torch.Tensor):
        target_points = target_points.cpu().numpy()
    if target_points.shape[0] == 3 and target_points.shape[1] != 3:
        target_points = target_points.T
        
    if isinstance(predicted_points, torch.Tensor):
        predicted_points = predicted_points.cpu().numpy()
    if predicted_points.shape[0] == 3 and predicted_points.shape[1] != 3:
        predicted_points = predicted_points.T
    
    # 根据是否有参考CIF文件选择不同的保存方法
    if cif_content is not None and atom_lines is not None and atom_types is not None and atom_labels is not None:
        # 使用参考CIF文件保存
        print("使用参考CIF文件保存结果...")
        save_as_cif(source_points, atom_types, atom_labels, cif_content, atom_lines, source_cif_path)
        save_as_cif(target_points, atom_types, atom_labels, cif_content, atom_lines, target_cif_path)
        save_as_cif(predicted_points, atom_types, atom_labels, cif_content, atom_lines, predicted_cif_path)
    else:
        # 直接从点云生成CIF文件
        print("没有参考CIF文件，直接从点云生成CIF文件...")
        save_as_cif_direct(source_points, source_cif_path)
        save_as_cif_direct(target_points, target_cif_path)
        save_as_cif_direct(predicted_points, predicted_cif_path)
    
    print(f"已保存源点云为CIF格式: {source_cif_path}")
    print(f"已保存目标点云为CIF格式: {target_cif_path}")
    print(f"已保存预测点云为CIF格式: {predicted_cif_path}")
    
    return source_cif_path, target_cif_path, predicted_cif_path

def apply_transformation(source_points, rotation, translation, normalization_info=None):
    """将预测的变换应用到源点云
    
    参数:
        source_points: 源点云坐标，形状为[3, N]或[N, 3]
        rotation: 旋转矩阵，形状为[3, 3]
        translation: 平移向量，形状为[3]
        normalization_info: 归一化信息字典，包含is_normalized、scale_factor和original_centroid
        
    返回:
        transformed_points: 变换后的点云，与输入格式相同
    """
    # 检查输入是否是张量
    if not isinstance(source_points, torch.Tensor):
        source_points = torch.tensor(source_points, dtype=torch.float32)
    if not isinstance(rotation, torch.Tensor):
        rotation = torch.tensor(rotation, dtype=torch.float32)
    if not isinstance(translation, torch.Tensor):
        translation = torch.tensor(translation, dtype=torch.float32)
        
    # 记录原始点云的形状
    is_transposed = False
    original_shape = source_points.shape
    
    # 确保点云形状为[3, N]
    if source_points.shape[0] != 3 and source_points.shape[1] == 3:
        source_points = source_points.transpose(0, 1)
        is_transposed = True
        
    # 获取点数
    num_points = source_points.shape[1]
    
    # 注意：如果normalization_info不为None且is_normalized为True，
    # 则translation已经在predict_transformation函数中被还原到原始坐标系，
    # 所以这里不需要再次还原
    
    # 应用旋转变换
    transformed_points = rotation @ source_points
    
    # 应用平移变换（广播平移向量到每个点）
    if len(translation.shape) == 1:
        # 如果是1D向量[3]，扩展为[3, 1]然后广播
        transformed_points = transformed_points + translation.reshape(3, 1)
    else:
        # 如果已经是[3, 1]或其他形状
        transformed_points = transformed_points + translation
    
    # 如果是归一化数据，考虑还原到原始坐标系
    if normalization_info and normalization_info.get('is_normalized', False):
        original_centroid = normalization_info['original_centroid']
        # 不需要再乘以scale_factor，因为rotation和translation已经被还原
        # 只需要加回原始质心
        if isinstance(original_centroid, np.ndarray):
            original_centroid = torch.tensor(original_centroid, dtype=torch.float32)
        # 将原始质心添加回所有点
        transformed_points = transformed_points + original_centroid.reshape(3, 1)
        print(f"已将变换后的点云还原到原始坐标系，添加原始质心: {original_centroid}")
    
    # 如果输入是转置的，则转回原始形状
    if is_transposed:
        transformed_points = transformed_points.transpose(0, 1)
        
    return transformed_points

def visualize_point_clouds(source, target=None, predicted=None, title="点云可视化"):
    """可视化点云"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制源点云（蓝色）
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], c='blue', label='源点云', s=5, alpha=0.7)
    
    # 绘制目标点云（红色）
    if target is not None:
        ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='red', label='目标点云', s=5, alpha=0.7)
    
    # 绘制预测点云（绿色）
    if predicted is not None:
        ax.scatter(predicted[:, 0], predicted[:, 1], predicted[:, 2], c='green', label='预测点云', s=5, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig

def parse_args_from_yaml(yaml_path):
    """从YAML文件解析参数"""
    with open(yaml_path, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return EasyDict(config)

def batch_predict(dir_path, args, cmd_args):
    """批量处理目录中的所有PKL文件"""
    # 检查输入目录是否存在
    if not os.path.exists(dir_path):
        print(f"错误: 输入目录 {dir_path} 不存在")
        return
    
    # 创建输出目录
    if not os.path.exists(cmd_args.output_dir):
        os.makedirs(cmd_args.output_dir)
    
    # 创建总体日志文件
    batch_log_file = os.path.join(cmd_args.output_dir, 'batch_predict_log.txt')
    batch_textio = IOStream(batch_log_file)
    
    # 查找所有PKL文件
    pkl_files = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
    
    if not pkl_files:
        batch_textio.cprint(f"错误: 目录 {dir_path} 中没有找到PKL文件")
        batch_textio.close()
        return
    
    batch_textio.cprint(f"找到 {len(pkl_files)} 个PKL文件")
    batch_textio.cprint(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    batch_textio.cprint(f"模型路径: {args.model_path}")
    
    # 加载模型(只需要加载一次)
    if args.model == 'RIENET':
        net = RIENET(args).cuda()
        
        if not os.path.exists(args.model_path):
            batch_textio.cprint(f"错误: 找不到预训练模型 {args.model_path}")
            batch_textio.close()
            return
        
        try:
            checkpoint = torch.load(args.model_path)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                batch_textio.cprint(f"加载模型 epoch {checkpoint.get('epoch', 'unknown')}, 最佳结果: {checkpoint.get('best_result', 'unknown')}")
                net.load_state_dict(checkpoint['model'], strict=False)
            else:
                # 处理.t7格式或直接的state_dict
                net.load_state_dict(checkpoint)
                batch_textio.cprint(f"成功加载模型: {args.model_path}")
        except Exception as e:
            batch_textio.cprint(f"加载模型失败: {e}")
            batch_textio.close()
            return
        
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            batch_textio.cprint(f"使用 {torch.cuda.device_count()} GPUs!")
    else:
        batch_textio.cprint(f"错误: 不支持的模型类型 {args.model}")
        batch_textio.close()
        return
    
    # 读取参考CIF文件（如果提供）- 只需读取一次
    atom_types = None
    atom_labels = None
    cif_content = None
    atom_lines = None
    
    if cmd_args.reference_cif and os.path.exists(cmd_args.reference_cif):
        try:
            # 解析参考CIF文件
            _, atom_types, atom_labels, cif_content, atom_lines = parse_cif(cmd_args.reference_cif)
            batch_textio.cprint(f"成功解析参考CIF文件: {cmd_args.reference_cif}")
        except Exception as e:
            batch_textio.cprint(f"解析参考CIF文件失败: {e}")
            batch_textio.cprint("将使用直接生成CIF文件的方法")
    else:
        batch_textio.cprint("未提供参考CIF文件，将直接从点云生成CIF文件")
    
    # 处理结果统计
    success_count = 0
    failed_files = []
    
    # 批量处理每个PKL文件
    for i, pkl_file in enumerate(pkl_files):
        batch_textio.cprint(f"\n[{i+1}/{len(pkl_files)}] 处理文件: {pkl_file}")
        
        # 为每个PKL文件创建子目录
        pkl_name = os.path.splitext(pkl_file)[0]
        file_output_dir = os.path.join(cmd_args.output_dir, pkl_name)
        if not os.path.exists(file_output_dir):
            os.makedirs(file_output_dir)
        
        # 为当前PKL文件创建日志文件
        log_file = os.path.join(file_output_dir, f'predict_{pkl_name}.log')
        textio = IOStream(log_file)
        
        try:
            # 加载PKL文件
            pkl_path = os.path.join(dir_path, pkl_file)
            dataset = PredictionDataset(pkl_path)
            textio.cprint(f"已加载PKL文件: {pkl_path}")
            
            # 预测变换矩阵
            textio.cprint("开始预测变换...")
            rotation_pred, translation_pred, normalization_info = predict_transformation(args, net, dataset)
            
            # 打印预测结果
            textio.cprint("\n预测的旋转矩阵:")
            textio.cprint(np.array2string(rotation_pred, precision=4, suppress_small=True))
            textio.cprint("\n预测的平移向量:")
            textio.cprint(np.array2string(translation_pred, precision=4, suppress_small=True))
            
            # 获取源点云和目标点云
            source_points = dataset.source.clone()
            target_points = dataset.target.clone()
            
            # 应用变换到源点云
            transformed_points = apply_transformation(source_points, rotation_pred, translation_pred, normalization_info)
            textio.cprint(f"已应用变换到源点云")
            
            # 保存为CIF文件
            textio.cprint("正在生成CIF格式结果文件...")
            source_cif, target_cif, predicted_cif = save_results_as_cif(
                dataset, source_points, target_points, transformed_points,
                atom_types, atom_labels, cif_content, atom_lines, file_output_dir
            )
            
            # 可视化（如果需要）
            if cmd_args.visualize:
                source_np = source_points.cpu().numpy()
                if source_np.shape[0] == 3 and source_np.shape[1] != 3:
                    source_np = source_np.T
                    
                target_np = target_points.cpu().numpy()
                if target_np.shape[0] == 3 and target_np.shape[1] != 3:
                    target_np = target_np.T
                    
                transformed_np = transformed_points
                if isinstance(transformed_np, torch.Tensor):
                    transformed_np = transformed_np.cpu().numpy()
                if transformed_np.shape[0] == 3 and transformed_np.shape[1] != 3:
                    transformed_np = transformed_np.T
                    
                fig = visualize_point_clouds(source_np, target_np, transformed_np, title=f"源点云、目标点云和预测点云: {pkl_name}")
                fig_path = os.path.join(file_output_dir, f"{pkl_name}_visualization.png")
                fig.savefig(fig_path)
                plt.close(fig)
                textio.cprint(f"已保存可视化图像到: {fig_path}")
            
            textio.cprint("\n预测完成！结果已保存到 " + log_file)
            success_count += 1
            batch_textio.cprint(f"成功处理文件: {pkl_file}")
            
        except Exception as e:
            textio.cprint(f"处理失败: {e}")
            import traceback
            textio.cprint(traceback.format_exc())
            batch_textio.cprint(f"处理失败: {pkl_file}, 错误: {e}")
            failed_files.append(pkl_file)
        
        textio.close()
    
    # 输出批处理摘要
    batch_textio.cprint("\n\n批处理摘要:")
    batch_textio.cprint(f"总共处理 {len(pkl_files)} 个文件，成功: {success_count}，失败: {len(pkl_files) - success_count}")
    
    if failed_files:
        batch_textio.cprint("\n失败的文件:")
        for f in failed_files:
            batch_textio.cprint(f"  - {f}")
    
    batch_textio.cprint(f"\n批处理完成，详细日志保存在: {batch_log_file}")
    batch_textio.close()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从PKL文件预测点云变换并生成CIF文件')
    parser.add_argument('--config', type=str, default='config/train-ca.yaml', help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--pkl_path', type=str, help='包含源点云和目标点云的PKL文件路径')
    parser.add_argument('--pkl_dir', type=str, help='包含多个PKL文件的目录路径，用于批处理')
    parser.add_argument('--reference_cif', type=str, required=False, default=None, help='参考CIF文件，用于生成结果，可选')
    parser.add_argument('--output_dir', type=str, default='predict_results', help='输出结果目录')
    parser.add_argument('--visualize', action='store_true', help='是否可视化点云')
    
    cmd_args = parser.parse_args()
    
    # 加载YAML配置
    args = parse_args_from_yaml(cmd_args.config)
    
    # 命令行参数覆盖YAML配置
    args.model_path = cmd_args.model_path
    
    # 检查参数
    if cmd_args.pkl_path and cmd_args.pkl_dir:
        print("错误: 不能同时指定pkl_path和pkl_dir，请只选择一种模式")
        return
    
    if not cmd_args.pkl_path and not cmd_args.pkl_dir:
        print("错误: 必须指定pkl_path或pkl_dir")
        parser.print_help()
        return
    
    # 根据模式选择处理方法
    if cmd_args.pkl_path:
        # 单文件处理模式
        # 创建输出目录
        if not os.path.exists(cmd_args.output_dir):
            os.makedirs(cmd_args.output_dir)
        
        # 设置日志输出
        pkl_name = os.path.splitext(os.path.basename(cmd_args.pkl_path))[0]
        log_file = os.path.join(cmd_args.output_dir, f'predict_{pkl_name}.log')
        textio = IOStream(log_file)
        
        # 设置随机种子
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        
        # 输出基本信息
        textio.cprint(f"配置: {args}")
        textio.cprint(f"PKL文件: {cmd_args.pkl_path}")
        textio.cprint(f"模型路径: {args.model_path}")
        
        # 加载数据
        try:
            # 加载PKL文件
            dataset = PredictionDataset(cmd_args.pkl_path)
            
            # 初始化CIF文件相关变量
            atom_types = None
            atom_labels = None
            cif_content = None
            atom_lines = None
            
            # 读取参考CIF文件（如果提供）
            if cmd_args.reference_cif and os.path.exists(cmd_args.reference_cif):
                try:
                    # 解析参考CIF文件
                    _, atom_types, atom_labels, cif_content, atom_lines = parse_cif(cmd_args.reference_cif)
                    textio.cprint(f"成功解析参考CIF文件: {cmd_args.reference_cif}")
                except Exception as e:
                    textio.cprint(f"解析参考CIF文件失败: {e}")
                    textio.cprint("将使用直接生成CIF文件的方法")
            else:
                textio.cprint("未提供参考CIF文件，将直接从点云生成CIF文件")
        except Exception as e:
            textio.cprint(f"加载数据失败: {e}")
            return
        
        # 加载模型
        if args.model == 'RIENET':
            net = RIENET(args).cuda()
            
            if not os.path.exists(args.model_path):
                textio.cprint(f"错误: 找不到预训练模型 {args.model_path}")
                return
            
            checkpoint = torch.load(args.model_path)
            textio.cprint(f"加载模型 epoch {checkpoint.get('epoch', 'unknown')}, 最佳结果: {checkpoint.get('best_result', 'unknown')}")
            net.load_state_dict(checkpoint['model'], strict=False)
            
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
                textio.cprint(f"使用 {torch.cuda.device_count()} GPUs!")
        else:
            textio.cprint(f"错误: 不支持的模型类型 {args.model}")
            return
        
        # 预测变换矩阵
        textio.cprint("开始预测变换...")
        rotation_pred, translation_pred, normalization_info = predict_transformation(args, net, dataset)
        
        # 打印预测结果
        textio.cprint("\n预测的旋转矩阵:")
        textio.cprint(np.array2string(rotation_pred, precision=4, suppress_small=True))
        textio.cprint("\n预测的平移向量:")
        textio.cprint(np.array2string(translation_pred, precision=4, suppress_small=True))
        
        # 获取源点云和目标点云
        source_points = dataset.source.clone()
        target_points = dataset.target.clone()
        
        # 应用变换到源点云
        transformed_points = apply_transformation(source_points, rotation_pred, translation_pred, normalization_info)
        textio.cprint(f"已应用变换到源点云")
        
        # 保存为CIF文件
        textio.cprint("正在生成CIF格式结果文件...")
        source_cif, target_cif, predicted_cif = save_results_as_cif(
            dataset, source_points, target_points, transformed_points,
            atom_types, atom_labels, cif_content, atom_lines, cmd_args.output_dir
        )
        
        # 可视化（如果需要）
        if cmd_args.visualize:
            source_np = source_points.cpu().numpy()
            if source_np.shape[0] == 3 and source_np.shape[1] != 3:
                source_np = source_np.T
                
            target_np = target_points.cpu().numpy()
            if target_np.shape[0] == 3 and target_np.shape[1] != 3:
                target_np = target_np.T
                
            transformed_np = transformed_points
            if isinstance(transformed_np, torch.Tensor):
                transformed_np = transformed_np.cpu().numpy()
            if transformed_np.shape[0] == 3 and transformed_np.shape[1] != 3:
                transformed_np = transformed_np.T
                
            fig = visualize_point_clouds(source_np, target_np, transformed_np, title=f"源点云、目标点云和预测点云: {pkl_name}")
            fig_path = os.path.join(cmd_args.output_dir, f"{pkl_name}_visualization.png")
            fig.savefig(fig_path)
            plt.close(fig)
            textio.cprint(f"已保存可视化图像到: {fig_path}")
        
        textio.cprint("\n预测完成！结果已保存到 " + log_file)
        textio.close()
    else:
        # 批处理模式
        batch_predict(cmd_args.pkl_dir, args, cmd_args)
    
if __name__ == '__main__':
    main()
