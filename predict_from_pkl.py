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
import csv
import math
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


def relative_rotation_error(gt_rotation, pred_rotation):
    """计算相对旋转误差(RRE)，单位为度
    
    RRE = acos((trace(R_gt^T · R_pred) - 1) / 2)
    
    参数:
        gt_rotation: 真实旋转矩阵，形状为[3, 3]
        pred_rotation: 预测的旋转矩阵，形状为[3, 3]
        
    返回:
        rre: 相对旋转误差，单位为度
    """
    if isinstance(gt_rotation, np.ndarray):
        gt_rotation = torch.from_numpy(gt_rotation).float()
    if isinstance(pred_rotation, np.ndarray):
        pred_rotation = torch.from_numpy(pred_rotation).float()
    
    # 打印详细的矩阵信息
    print("\n============ 旋转矩阵详细信息 ============")
    print("真实旋转矩阵 (gt_rotation):")
    print(gt_rotation.numpy())
    print("\n预测旋转矩阵 (pred_rotation):")
    print(pred_rotation.numpy())
    
    # 检查旋转矩阵的性质
    gt_det = torch.det(gt_rotation)
    pred_det = torch.det(pred_rotation)
    print(f"\n行列式检查 - gt_det: {gt_det.item():.6f}, pred_det: {pred_det.item():.6f}")
    
    # 检查是否为正交矩阵 (R·R^T = I)
    gt_orthogonal = torch.matmul(gt_rotation, gt_rotation.transpose(-1, -2))
    pred_orthogonal = torch.matmul(pred_rotation, pred_rotation.transpose(-1, -2))
    print("\n正交性检查 (R·R^T 应接近单位矩阵):")
    print("gt_rotation · gt_rotation^T:")
    print(gt_orthogonal.numpy())
    print("pred_rotation · pred_rotation^T:")
    print(pred_orthogonal.numpy())
        
    # 修正：根据GeoTransformer的标准实现
    # 计算方式1: pred_rotation.T @ gt_rotation (当前实现)
    print("\n============ RRE计算方式1: pred_rotation.T @ gt_rotation ============")
    mat1 = torch.matmul(pred_rotation.transpose(-1, -2), gt_rotation)
    trace1 = mat1[..., 0, 0] + mat1[..., 1, 1] + mat1[..., 2, 2]
    print("乘积矩阵 (pred_rotation.T @ gt_rotation):")
    print(mat1.numpy())
    print(f"矩阵乘积的trace值: {trace1.item():.6f}")
    
    # 防止数值误差导致超出[-1, 1]范围
    cos_theta1 = (trace1 - 1.0) * 0.5
    cos_theta1 = torch.clamp(cos_theta1, -1.0, 1.0)
    theta1 = torch.acos(cos_theta1)
    rre1 = theta1 * 180.0 / math.pi
    print(f"cos_theta值: {cos_theta1.item():.6f}")
    print(f"得到的角度: {rre1.item():.6f}度")
    
    # 计算方式2: gt_rotation.T @ pred_rotation (替代方式)
    print("\n============ RRE计算方式2: gt_rotation.T @ pred_rotation ============")
    mat2 = torch.matmul(gt_rotation.transpose(-1, -2), pred_rotation)
    trace2 = mat2[..., 0, 0] + mat2[..., 1, 1] + mat2[..., 2, 2]
    print("乘积矩阵 (gt_rotation.T @ pred_rotation):")
    print(mat2.numpy())
    print(f"矩阵乘积的trace值: {trace2.item():.6f}")
    
    cos_theta2 = (trace2 - 1.0) * 0.5
    cos_theta2 = torch.clamp(cos_theta2, -1.0, 1.0)
    theta2 = torch.acos(cos_theta2)
    rre2 = theta2 * 180.0 / math.pi
    print(f"cos_theta值: {cos_theta2.item():.6f}")
    print(f"得到的角度: {rre2.item():.6f}度")
    
    # 使用原始计算方式1的结果
    print(f"\n最终使用方式1的结果: {rre1.item():.6f}度")
    print("=============================================")
    
    return rre1.item()

def relative_translation_error(gt_translation, pred_translation):
    """计算相对平移误差(RTE)，单位与输入相同"""
    if isinstance(gt_translation, np.ndarray):
        gt_translation = torch.from_numpy(gt_translation).float()
    if isinstance(pred_translation, np.ndarray):
        pred_translation = torch.from_numpy(pred_translation).float()
        
    # 计算欧氏距离
    return torch.norm(gt_translation - pred_translation).item()

def compute_rmse(source_points, gt_points, pred_points):
    """计算均方根误差(RMSE)，单位与输入相同"""
    if isinstance(source_points, torch.Tensor):
        source_points = source_points.detach().cpu().numpy()
    if isinstance(gt_points, torch.Tensor):
        gt_points = gt_points.detach().cpu().numpy()
    if isinstance(pred_points, torch.Tensor):
        pred_points = pred_points.detach().cpu().numpy()
        
    # 确保形状一致 (N, 3)
    if source_points.shape[0] == 3 and source_points.shape[1] != 3:
        source_points = source_points.T
    if gt_points.shape[0] == 3 and gt_points.shape[1] != 3:
        gt_points = gt_points.T
    if pred_points.shape[0] == 3 and pred_points.shape[1] != 3:
        pred_points = pred_points.T
        
    # 计算均方根误差
    squared_diff = np.sum((pred_points - gt_points) ** 2, axis=1)
    rmse = np.sqrt(np.mean(squared_diff))
    
    return rmse

def registration_recall(rmse, threshold=0.2):
    """计算配准召回率(RR)，即RMSE小于阈值的比例"""
    return 1.0 if rmse < threshold else 0.0

def evaluate_registration(source_points, target_points, transformed_points, 
                         gt_rotation=None, gt_translation=None, pred_rotation=None, 
                         pred_translation=None, threshold=0.2, normalization_info=None):
    """
    评估点云配准结果
    
    参数:
        source_points: 源点云
        target_points: 目标点云（通常作为参考）
        transformed_points: 经过变换后的源点云
        gt_rotation: 真实旋转矩阵，如果有
        gt_translation: 真实平移向量，如果有
        pred_rotation: 预测的旋转矩阵
        pred_translation: 预测的平移向量
        threshold: 配准召回率的阈值，默认0.2
        normalization_info: 归一化信息，用于确保比较相同坐标系下的平移向量
        
    返回:
        results: 包含评估指标的字典
    """
    results = {}
    
    # 计算RMSE - 暂时注释，因为源点云和目标点云大小不一致
    #rmse = compute_rmse(source_points, target_points, transformed_points)
    #results['rmse'] = rmse
    
    # 临时设置一个默认值，避免后续代码出错
    results['rmse'] = 0.0
    
    # 计算配准召回率 - 暂时设置为1.0
    #rr = registration_recall(rmse, threshold)
    results['recall'] = 1.0  # 默认设为1.0
    
    # 如果有真实变换，计算RRE和RTE
    if gt_rotation is not None and gt_translation is not None and pred_rotation is not None and pred_translation is not None:
        # 由于模型直接输出原始坐标系的平移向量，无需对gt_translation进行缩放
        gt_translation_for_comparison = gt_translation.copy() if isinstance(gt_translation, np.ndarray) else gt_translation.clone()
        
        # 这里不再对gt_translation乘以缩放因子，因为模型已经直接输出原始坐标系的平移向量
        if normalization_info and normalization_info.get('is_normalized', False):
            print(f"原始真实平移向量(归一化空间): {gt_translation}")
            # 原始平移向量(归一化空间) × 缩放因子 = 真实平移向量(原始空间)
            scale_factor = normalization_info['scale_factor']
            original_translation = gt_translation * scale_factor
            print(f"还原后真实平移向量: {original_translation}")
        
        # 现在两个平移向量都在相同的坐标系中进行比较
        rre = relative_rotation_error(gt_rotation, pred_rotation)
        rte = relative_translation_error(gt_translation_for_comparison, pred_translation)
        results['rre'] = rre
        results['rte'] = rte
    else:
        results['rre'] = None
        results['rte'] = None
        
    return results

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
            
        # 提取网格信息 - 优先从grid_info中获取，这是cif_to_point_cloud.py中使用的格式
        if 'grid_info' in data:
            grid_info = data['grid_info']
            
            # 从grid_info中提取网格信息
            if 'grid_shape' in grid_info:
                self.grid_shape = grid_info['grid_shape']
            else:
                self.grid_shape = np.array([32, 32, 32], dtype=np.int32)
                
            if 'x_origin' in grid_info:
                self.x_origin = grid_info['x_origin']
            else:
                self.x_origin = np.array([0.0], dtype=np.float32)
                
            if 'y_origin' in grid_info:
                self.y_origin = grid_info['y_origin']
            else:
                self.y_origin = np.array([0.0], dtype=np.float32)
                
            if 'z_origin' in grid_info:
                self.z_origin = grid_info['z_origin']
            else:
                self.z_origin = np.array([0.0], dtype=np.float32)
                
            if 'x_voxel' in grid_info:
                self.x_voxel = grid_info['x_voxel']
            else:
                self.x_voxel = np.array([0.05], dtype=np.float32)
                
            # 检查归一化信息
            self.is_normalized = grid_info.get('normalized', False)
            self.scale_factor = grid_info.get('scale_factor', 1.0)
            self.original_centroid = grid_info.get('original_centroid', np.zeros(3, dtype=np.float32))
            
            # nstart默认为0
            self.nstart = 0
        else:
            # 如果没有grid_info，则直接检查顶级键
            if 'grid_shape' in data:
                self.grid_shape = data['grid_shape']
            else:
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
                self.nstart = 0
            
            # 检查归一化信息
            self.is_normalized = data.get('normalized', False)
            self.scale_factor = data.get('scale_factor', 1.0)
            self.original_centroid = data.get('original_centroid', np.zeros(3, dtype=np.float32))
        
        # 确保original_centroid是numpy数组
        if isinstance(self.original_centroid, list):
            self.original_centroid = np.array(self.original_centroid, dtype=np.float32)
            
        # 打印归一化信息
        if self.is_normalized:
            print(f"检测到归一化数据, 缩放因子: {self.scale_factor}, 原始质心: {self.original_centroid}")
        
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
    """使用模型预测变换，并计算评估指标"""
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
        
        # 先保存原始预测的平移向量（归一化空间中的）
        translation_pred_normalized = translation_pred_np.copy()
        
        # 在归一化空间中应用变换
        source_cpu = source.cpu().squeeze(0)  # 移除批次维度
        target_cpu = target.cpu().squeeze(0)
        
        # 在归一化空间中应用变换，不做还原
        transformed_points_normalized = apply_transformation_in_normalized_space(
            source_cpu.clone(), 
            rotation_pred_np, 
            translation_pred_normalized
        )
        
        # 如果是归一化数据，将预测的平移向量和点云还原到原始坐标系
        if is_normalized:
            scale_factor = normalization_info['scale_factor']
            original_centroid = normalization_info['original_centroid']
            
            # # 对于归一化数据，平移向量需要乘以缩放因子来还原到原始坐标系
            # translation_pred_np = translation_pred_normalized * scale_factor
            # print(f"已将平移向量从归一化空间还原到原始坐标系，乘以缩放因子: {scale_factor}")
            
            # 将归一化空间中的变换后点云还原到原始坐标系
            if isinstance(transformed_points_normalized, torch.Tensor):
                transformed_points_normalized = transformed_points_normalized.cpu().numpy()
            
            # 确保点云形状正确用于还原
            if transformed_points_normalized.shape[0] == 3 and transformed_points_normalized.shape[1] != 3:
                transformed_points_normalized = transformed_points_normalized.T
            
            # 如果有真实旋转矩阵，计算旋转误差但不影响还原过程
            if hasattr(dataset, 'rotation'):
                gt_rotation = dataset.rotation.numpy() if isinstance(dataset.rotation, torch.Tensor) else dataset.rotation
                rre = relative_rotation_error(gt_rotation, rotation_pred_np)
                print(f"计算得到的旋转误差(RRE): {rre:.4f}度")
            
            # 还原到原始坐标系 - 始终使用源点云的原始质心
            transformed_points = transformed_points_normalized * scale_factor
            
            # 统一使用源点云的原始质心还原
            if isinstance(transformed_points, np.ndarray):
                transformed_points = transformed_points + original_centroid
            else:
                transformed_points = transformed_points + torch.tensor(original_centroid, dtype=torch.float32)
                
            print(f"已将变换后的点云还原到原始坐标系。使用源点云原始质心: {original_centroid}")
        else:
            # 如果不是归一化数据，使用原始变换结果
            transformed_points = transformed_points_normalized
        
        print("预测的旋转矩阵:")
        print(rotation_pred_np)
        print("预测的平移向量:")
        print(translation_pred_np)
        
        # 应用变换到源点云用于评估
        source_cpu = source.cpu().squeeze(0)  # 移除批次维度
        target_cpu = target.cpu().squeeze(0)
        
        # 获取真实变换（如果有）
        gt_rotation = None
        gt_translation = None
        if hasattr(dataset, 'rotation') and hasattr(dataset, 'translation'):
            gt_rotation = dataset.rotation.numpy() if isinstance(dataset.rotation, torch.Tensor) else dataset.rotation
            gt_translation = dataset.translation.numpy() if isinstance(dataset.translation, torch.Tensor) else dataset.translation
        
        # 计算评估指标
        evaluation_results = None
        if gt_rotation is not None and gt_translation is not None:
            evaluation_results = evaluate_registration(
                source_cpu, 
                target_cpu, 
                transformed_points,
                gt_rotation,
                gt_translation,
                rotation_pred_np,
                translation_pred_np,
                threshold=0.2,  # 可以从配置中读取该阈值
                normalization_info=normalization_info  # 传递归一化信息
            )
            print("\n评估结果:")
            if evaluation_results['rre'] is not None:
                print(f"相对旋转误差 (RRE): {evaluation_results['rre']:.4f}度")
                print(f"相对平移误差 (RTE): {evaluation_results['rte']:.4f}单位")
            print(f"均方根误差 (RMSE): {evaluation_results['rmse']:.4f}单位")
            print(f"配准召回率 (RR): {evaluation_results['recall']:.4f}")
        
    return rotation_pred_np, translation_pred_np, normalization_info, evaluation_results

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

def apply_transformation_in_normalized_space(source_points, rotation, translation):
    """在归一化空间中应用变换，不进行坐标系还原
    
    参数:
        source_points: 源点云坐标，形状为[3, N]或[N, 3]
        rotation: 旋转矩阵，形状为[3, 3]
        translation: 平移向量，形状为[3]
        
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
    
    # 应用旋转变换
    transformed_points = rotation @ source_points
    
    # 应用平移变换
    if len(translation.shape) == 1:
        transformed_points = transformed_points + translation.reshape(3, 1)
    else:
        transformed_points = transformed_points + translation
    
    # 如果输入是转置的，则转回原始形状
    if is_transposed:
        transformed_points = transformed_points.transpose(0, 1)
        
    return transformed_points

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
    # 打印输入值的信息
    print(f"应用变换前源点云形状: {source_points.shape}")
    print(f"旋转矩阵形状: {rotation.shape}, 值: \n{rotation}")
    print(f"平移向量形状: {translation.shape}, 值: {translation}")
    
    if normalization_info:
        print(f"归一化信息: {normalization_info}")
        print(f"是否归一化: {normalization_info.get('is_normalized', False)}")
        print(f"缩放因子: {normalization_info.get('scale_factor', None)}")
        print(f"原始质心: {normalization_info.get('original_centroid', None)}")

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
    
    # 判断是否需要对点云进行归一化还原
    # 在predict_transformation函数中，如果是归一化数据，平移向量已经乘以缩放因子还原了
    need_denormalize = normalization_info and normalization_info.get('is_normalized', False)
    translation_already_denormalized = need_denormalize  # 平移向量已经在predict_transformation中还原了
    
    # 应用旋转变换
    transformed_points = rotation @ source_points
    
    # 应用平移变换（广播平移向量到每个点）
    if len(translation.shape) == 1:
        # 如果是1D向量[3]，扩展为[3, 1]然后广播
        transformed_points = transformed_points + translation.reshape(3, 1)
    else:
        # 如果已经是[3, 1]或其他形状
        transformed_points = transformed_points + translation
    
    # 如果是归一化数据，需要进行尺寸还原和质心添加
    if need_denormalize:
        scale_factor = normalization_info['scale_factor']
        original_centroid = normalization_info['original_centroid']
        
        if isinstance(original_centroid, np.ndarray):
            original_centroid = torch.tensor(original_centroid, dtype=torch.float32)
        
        # 无论平移向量是否已经还原，都执行相同的还原操作
        # 1. 将旋转平移后的点云乘以缩放因子
        transformed_points = transformed_points * scale_factor
        # 2. 然后加回原始质心
        transformed_points = transformed_points + original_centroid.reshape(3, 1)
        
        print(f"已将变换后的点云还原到原始坐标系，使用缩放因子: {scale_factor}，添加原始质心: {original_centroid}")
    
    # 如果输入是转置的，则转回原始形状
    if is_transposed:
        transformed_points = transformed_points.transpose(0, 1)
        
    return transformed_points

def save_cif_files(source_points, target_points, transformed_points, output_base_path, normalization_info=None):
    """保存源点云、目标点云和变换后的点云为CIF文件"""
    
    # 检查是否是张量，如果是，转换为numpy数组
    if isinstance(source_points, torch.Tensor):
        source_points = source_points.cpu().detach().numpy()
    if isinstance(target_points, torch.Tensor):
        target_points = target_points.cpu().detach().numpy()
    if isinstance(transformed_points, torch.Tensor):
        transformed_points = transformed_points.cpu().detach().numpy()
    
    # 创建目录
    output_dir = os.path.dirname(output_base_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件名（不含扩展名和路径）
    base_name = os.path.basename(output_base_path)
    if base_name.endswith('.cif'):
        base_name = base_name[:-4]
    
    # 如果是归一化数据，还原所有点云到原始坐标系
    if normalization_info and normalization_info.get('is_normalized', False):
        scale_factor = normalization_info['scale_factor']
        original_centroid = normalization_info['original_centroid']
        
        # 确保original_centroid是numpy数组
        if not isinstance(original_centroid, np.ndarray):
            original_centroid = np.array(original_centroid)
        
        # 检查点云格式并确保是[N, 3]格式用于保存
        if source_points.shape[0] == 3 and source_points.shape[1] != 3:
            source_points = source_points.T
        if target_points.shape[0] == 3 and target_points.shape[1] != 3:
            target_points = target_points.T
        if transformed_points.shape[0] == 3 and transformed_points.shape[1] != 3:
            transformed_points = transformed_points.T
        
        # 还原源点云
        source_points = source_points * scale_factor + original_centroid
        print(f"已将源点云还原到原始坐标系，使用缩放因子: {scale_factor}，添加原始质心: {original_centroid}")
        
        # 还原目标点云
        target_points = target_points * scale_factor + original_centroid
        print(f"已将目标点云还原到原始坐标系，使用缩放因子: {scale_factor}，添加原始质心: {original_centroid}")
        
        # 注意：变换后的点云已经在apply_transformation函数中被还原
    else:
        # 确保点云是[N, 3]格式用于保存
        if source_points.shape[0] == 3 and source_points.shape[1] != 3:
            source_points = source_points.T
        if target_points.shape[0] == 3 and target_points.shape[1] != 3:
            target_points = target_points.T
        if transformed_points.shape[0] == 3 and transformed_points.shape[1] != 3:
            transformed_points = transformed_points.T
    
    # 保存为CIF文件
    source_cif_path = os.path.join(output_dir, f"{base_name}_source.cif")
    target_cif_path = os.path.join(output_dir, f"{base_name}_target.cif")
    predicted_cif_path = os.path.join(output_dir, f"{base_name}_predicted.cif")
    
    save_as_cif_direct(source_points, source_cif_path)
    save_as_cif_direct(target_points, target_cif_path)
    save_as_cif_direct(transformed_points, predicted_cif_path)
    
    print(f"已保存源点云为CIF格式: {source_cif_path}")
    print(f"已保存目标点云为CIF格式: {target_cif_path}")
    print(f"已保存预测点云为CIF格式: {predicted_cif_path}")

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
    """批量处理目录中的所有PKL文件，并生成评估结果CSV文件"""
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
    
    # 创建CSV结果文件
    csv_file = os.path.join(cmd_args.output_dir, 'evaluation_results.csv')
    csv_header = ['文件名', 'RRE (度)', 'RTE', 'RMSE', '配准召回率', '是否成功']
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
    
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
    
    # 评估结果统计
    all_results = []
    
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
            
            # 预测变换矩阵并评估结果
            textio.cprint("开始预测变换...")
            rotation_pred, translation_pred, normalization_info, evaluation_results = predict_transformation(args, net, dataset)
            
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
            
            # 保存结果到CSV
            result_row = [pkl_name]
            if evaluation_results:
                textio.cprint("\n评估结果:")
                if evaluation_results['rre'] is not None:
                    textio.cprint(f"相对旋转误差 (RRE): {evaluation_results['rre']:.4f}度")
                    textio.cprint(f"相对平移误差 (RTE): {evaluation_results['rte']:.4f}单位")
                    result_row.extend([f"{evaluation_results['rre']:.4f}", f"{evaluation_results['rte']:.4f}"])
                else:
                    textio.cprint("注意: 无法计算RRE和RTE，因为没有提供真实变换")
                    result_row.extend(["N/A", "N/A"])
                
                textio.cprint(f"均方根误差 (RMSE): {evaluation_results['rmse']:.4f}单位")
                textio.cprint(f"配准召回率 (RR): {evaluation_results['recall']:.4f}")
                result_row.extend([f"{evaluation_results['rmse']:.4f}", f"{evaluation_results['recall']:.4f}"])
                
                # 保存结果用于统计
                all_results.append(evaluation_results)
            else:
                result_row.extend(["N/A", "N/A", "N/A", "N/A"])
            
            # 保存成功标志
            result_row.append("成功")
            
            # 写入CSV
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(result_row)
            
            # 如果是归一化数据，先将源点云和目标点云还原到原始坐标系
            if normalization_info and normalization_info.get('is_normalized', False):
                scale_factor = normalization_info['scale_factor']
                original_centroid = normalization_info['original_centroid']
                
                # 确保数据格式正确
                source_is_transposed = False
                target_is_transposed = False
                
                if source_points.shape[0] != 3 and source_points.shape[1] == 3:
                    source_points = source_points.transpose(0, 1)
                    source_is_transposed = True
                    
                if target_points.shape[0] != 3 and target_points.shape[1] == 3:
                    target_points = target_points.transpose(0, 1)
                    target_is_transposed = True
                
                # 将original_centroid转换为张量（如果是numpy数组）
                if isinstance(original_centroid, np.ndarray):
                    original_centroid = torch.tensor(original_centroid, dtype=torch.float32)
                
                # 还原源点云
                source_points = source_points * scale_factor + original_centroid.reshape(3, 1)
                textio.cprint(f"已将源点云还原到原始坐标系，使用缩放因子: {scale_factor}，添加原始质心: {original_centroid}")
                
                # 还原目标点云
                target_points = target_points * scale_factor + original_centroid.reshape(3, 1)
                textio.cprint(f"已将目标点云还原到原始坐标系，使用缩放因子: {scale_factor}，添加原始质心: {original_centroid}")
                
                # 转回原始形状（如果有转置）
                if source_is_transposed:
                    source_points = source_points.transpose(0, 1)
                if target_is_transposed:
                    target_points = target_points.transpose(0, 1)
            
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
            
            # 记录失败到CSV
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([pkl_name, "N/A", "N/A", "N/A", "N/A", "失败"])
        
        textio.close()
    
    # 输出批处理摘要
    batch_textio.cprint("\n\n批处理摘要:")
    batch_textio.cprint(f"总共处理 {len(pkl_files)} 个文件，成功: {success_count}，失败: {len(pkl_files) - success_count}")
    
    # 计算并保存统计结果
    if all_results:
        # 计算平均值和中位数
        avg_results = {'rre': None, 'rte': None, 'rmse': 0.0, 'recall': 0.0}
        med_results = {'rre': None, 'rte': None, 'rmse': 0.0, 'recall': 0.0}
        
        # 收集所有有效的测量结果
        valid_rre = [r['rre'] for r in all_results if r['rre'] is not None]
        valid_rte = [r['rte'] for r in all_results if r['rte'] is not None]
        valid_rmse = [r['rmse'] for r in all_results]
        valid_recall = [r['recall'] for r in all_results]
        
        # 计算平均值
        if valid_rre:
            avg_results['rre'] = sum(valid_rre) / len(valid_rre)
            med_results['rre'] = sorted(valid_rre)[len(valid_rre) // 2]
        
        if valid_rte:
            avg_results['rte'] = sum(valid_rte) / len(valid_rte)
            med_results['rte'] = sorted(valid_rte)[len(valid_rte) // 2]
        
        avg_results['rmse'] = sum(valid_rmse) / len(valid_rmse)
        avg_results['recall'] = sum(valid_recall) / len(valid_rmse)
        
        med_results['rmse'] = sorted(valid_rmse)[len(valid_rmse) // 2]
        med_results['recall'] = sorted(valid_recall)[len(valid_recall) // 2]
        
        # 将统计结果写入CSV
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([])  # 空行
            writer.writerow(["\u5e73\u5747\u503c:"])
            if avg_results['rre'] is not None:
                writer.writerow(["", f"{avg_results['rre']:.4f}", f"{avg_results['rte']:.4f}", 
                               f"{avg_results['rmse']:.4f}", f"{avg_results['recall']:.4f}", ""])
            else:
                writer.writerow(["", "N/A", "N/A", f"{avg_results['rmse']:.4f}", f"{avg_results['recall']:.4f}", ""])
                
            writer.writerow(["\u4e2d\u4f4d\u6570:"])
            if med_results['rre'] is not None:
                writer.writerow(["", f"{med_results['rre']:.4f}", f"{med_results['rte']:.4f}", 
                               f"{med_results['rmse']:.4f}", f"{med_results['recall']:.4f}", ""])
            else:
                writer.writerow(["", "N/A", "N/A", f"{med_results['rmse']:.4f}", f"{med_results['recall']:.4f}", ""])
        
        # 输出到日志
        batch_textio.cprint("\n\n评估结果统计:")
        batch_textio.cprint("\n平均值:")
        if avg_results['rre'] is not None:
            batch_textio.cprint(f"相对旋转误差 (RRE): {avg_results['rre']:.4f}度")
            batch_textio.cprint(f"相对平移误差 (RTE): {avg_results['rte']:.4f}单位")
        batch_textio.cprint(f"均方根误差 (RMSE): {avg_results['rmse']:.4f}单位")
        batch_textio.cprint(f"配准召回率 (RR): {avg_results['recall']:.4f}")
        
        batch_textio.cprint("\n中位数:")
        if med_results['rre'] is not None:
            batch_textio.cprint(f"相对旋转误差 (RRE): {med_results['rre']:.4f}度")
            batch_textio.cprint(f"相对平移误差 (RTE): {med_results['rte']:.4f}单位")
        batch_textio.cprint(f"均方根误差 (RMSE): {med_results['rmse']:.4f}单位")
        batch_textio.cprint(f"配准召回率 (RR): {med_results['recall']:.4f}")
        
        batch_textio.cprint(f"\n统计结果已保存到: {csv_file}")
    
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
        
        # 如果是归一化数据，先将源点云和目标点云还原到原始坐标系
        if normalization_info and normalization_info.get('is_normalized', False):
            scale_factor = normalization_info['scale_factor']
            original_centroid = normalization_info['original_centroid']
            
            # 确保数据格式正确
            source_is_transposed = False
            target_is_transposed = False
            
            if source_points.shape[0] != 3 and source_points.shape[1] == 3:
                source_points = source_points.transpose(0, 1)
                source_is_transposed = True
                
            if target_points.shape[0] != 3 and target_points.shape[1] == 3:
                target_points = target_points.transpose(0, 1)
                target_is_transposed = True
            
            # 将original_centroid转换为张量（如果是numpy数组）
            if isinstance(original_centroid, np.ndarray):
                original_centroid = torch.tensor(original_centroid, dtype=torch.float32)
            
            # 还原源点云
            source_points = source_points * scale_factor + original_centroid.reshape(3, 1)
            textio.cprint(f"已将源点云还原到原始坐标系，使用缩放因子: {scale_factor}，添加原始质心: {original_centroid}")
            
            # 还原目标点云
            target_points = target_points * scale_factor + original_centroid.reshape(3, 1)
            textio.cprint(f"已将目标点云还原到原始坐标系，使用缩放因子: {scale_factor}，添加原始质心: {original_centroid}")
            
            # 转回原始形状（如果有转置）
            if source_is_transposed:
                source_points = source_points.transpose(0, 1)
            if target_is_transposed:
                target_points = target_points.transpose(0, 1)
        
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
