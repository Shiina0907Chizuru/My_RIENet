#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import RIENET
from util import npmat2euler, transform_point_cloud
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
import yaml
from easydict import EasyDict
from common.math import se3
from common.math_torch import se3
import pickle
import glob
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

class SinglePklDataset(Dataset):
    def __init__(self, pkl_path):
        # 加载数据
        self.pkl_path = pkl_path
        # 加载PKL文件
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 提取数据
        self.source = data['source'].clone().detach() if isinstance(data['source'], torch.Tensor) else torch.tensor(data['source'], dtype=torch.float32)
        self.target = data['target'].clone().detach() if isinstance(data['target'], torch.Tensor) else torch.tensor(data['target'], dtype=torch.float32)
        self.rotation = data['rotation'].clone().detach() if isinstance(data['rotation'], torch.Tensor) else torch.tensor(data['rotation'], dtype=torch.float32)
        self.translation = data['translation'].clone().detach() if isinstance(data['translation'], torch.Tensor) else torch.tensor(data['translation'], dtype=torch.float32)
        
        # 保存原始CIF信息（如果存在）
        self.atom_types = data.get('atom_types', [])
        self.atom_labels = data.get('atom_labels', [])
        self.cif_content = data.get('cif_content', [])
        self.atom_lines = data.get('atom_lines', [])
        
        # 获取网格信息（如果存在）
        self.grid_shape = data.get('grid_shape', np.array([64, 64, 64], dtype=np.int32))
        self.x_origin = data.get('x_origin', np.array([0.0], dtype=np.float32))
        self.y_origin = data.get('y_origin', np.array([0.0], dtype=np.float32))
        self.z_origin = data.get('z_origin', np.array([0.0], dtype=np.float32))
        self.x_voxel = data.get('x_voxel', np.array([0.05], dtype=np.float32))
        self.nstart = data.get('nstart', np.array([32], dtype=np.int32))
        
        print(f"已加载文件: {pkl_path}")
        print(f"点云形状: source={self.source.shape}, target={self.target.shape}")
        print(f"旋转矩阵形状: {self.rotation.shape}, 平移向量形状: {self.translation.shape}")
        
        # 检查是否有CIF相关数据
        if self.atom_types and self.atom_labels and self.cif_content and self.atom_lines:
            print(f"检测到CIF相关数据: {len(self.atom_types)}个原子")
        else:
            print("未检测到CIF相关数据，将无法生成CIF格式输出文件")

    def __len__(self):
        return 1  # 只有一个样本

    def __getitem__(self, idx):
        # 返回一个元组
        return (self.source, self.target, self.rotation, self.translation, 
                self.grid_shape, self.x_origin, self.y_origin, self.z_origin, 
                self.x_voxel, self.nstart)

def test_one_epoch(args, net, test_loader):
    net.eval()

    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []
    
    for src, target, rotation_ab, translation_ab, grid_shape, x_origin, y_origin, z_origin, x_voxel, nstart in test_loader:
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()

        # 在test_one_epoch的数中添加
        if src.shape[1] != 3:
            print(f"调整源点云格式: 从{src.shape}到", end="")
            src = src.transpose(1, 2)  # 从[B, N, 3]转置为[B, 3, N]
            print(f"{src.shape}")
        
        if target.shape[1] != 3:
            print(f"调整目标点云格式: 从{target.shape}到", end="")
            target = target.transpose(1, 2)  # 从[B, N, 3]转置为[B, 3, N]
            print(f"{target.shape}")

        # 打印点云形状信息、k值和其他关键参数
        print(f"点云形状: source={src.shape}, target={target.shape}")
        print(f"网格形状: {grid_shape}")    
        print(f"配置k值: list_k1={args.list_k1[:3]}..., list_k2={args.list_k2[:3]}...")
        print(f"x_origin类型: {type(x_origin)}, 形状: {x_origin.shape if torch.is_tensor(x_origin) else '非张量'}")
        print(f"y_origin类型: {type(y_origin)}, 形状: {y_origin.shape if torch.is_tensor(y_origin) else '非张量'}")
        print(f"z_origin类型: {type(z_origin)}, 形状: {z_origin.shape if torch.is_tensor(z_origin) else '非张量'}")
        print(f"x_voxel类型: {type(x_voxel)}, 形状: {x_voxel.shape if torch.is_tensor(x_voxel) else '非张量'}")
        print(f"nstart类型: {type(nstart)}, 形状: {nstart.shape if torch.is_tensor(nstart) else '非张量'}")

        batch_size = src.size(0)
        num_examples += batch_size
        
        rotation_ab_pred, translation_ab_pred, \
            loss1, loss2, loss3, loss4 = net(src, target, grid_shape, x_origin, y_origin, z_origin, x_voxel, nstart)
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        loss = loss1.sum() + loss2.sum() + loss3.sum() + loss4.sum()
        total_loss += loss.item()

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    return total_loss * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred

def save_as_cif(coordinates, atom_types, atom_labels, original_cif, atom_lines, output_path):
    """将点云坐标保存为CIF格式"""
    if not atom_types or not atom_labels or not original_cif or not atom_lines:
        print(f"警告：无法生成CIF文件，缺少必要的原子信息")
        return False
    
    if len(coordinates) != len(atom_types):
        print(f"警告：坐标数量({len(coordinates)})与原子类型数量({len(atom_types)})不匹配")
        return False
    
    # 创建CIF文件的新内容
    new_cif_content = original_cif.copy()
    
    # 更新原子坐标
    for i, (line_idx, coord) in enumerate(zip(atom_lines, coordinates)):
        line = original_cif[line_idx].strip()
        parts = line.split()
        
        # 替换坐标部分 (通常是第10-12列)
        if len(parts) >= 13:
            x, y, z = coord
            parts[10] = f"{x:.6f}"
            parts[11] = f"{y:.6f}"
            parts[12] = f"{z:.6f}"
            
            # 重建行
            new_line = " ".join(parts)
            new_cif_content[line_idx] = new_line + "\n"
    
    # 写入新的CIF文件
    with open(output_path, 'w') as f:
        f.writelines(new_cif_content)
    
    print(f"已保存CIF文件: {output_path}")
    return True

def save_as_cif_with_suffix(coordinates, atom_types, atom_labels, original_cif, atom_lines, output_path, suffix="_pred"):
    """将点云坐标保存为CIF格式，并添加自定义后缀"""
    # 获取文件名和扩展名
    base_path, ext = os.path.splitext(output_path)
    
    # 如果已经有后缀，就不重复添加
    if base_path.endswith(suffix):
        new_path = output_path
    else:
        new_path = base_path + suffix + ext
    
    return save_as_cif(coordinates, atom_types, atom_labels, original_cif, atom_lines, new_path)

def visualize_point_clouds(source, target, predicted=None, title="点云可视化"):
    """可视化点云"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 确保点云格式为[N, 3]
    if source.shape[0] == 3 and source.shape[1] != 3:
        source = source.T
    if target.shape[0] == 3 and target.shape[1] != 3:
        target = target.T
    if predicted is not None and predicted.shape[0] == 3 and predicted.shape[1] != 3:
        predicted = predicted.T
    
    # 绘制源点云 (蓝色)
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], c='blue', marker='o', label='源点云', alpha=0.5)
    
    # 绘制目标点云 (红色)
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='red', marker='^', label='目标点云', alpha=0.5)
    
    # 如果有预测点云，绘制预测点云 (绿色)
    if predicted is not None:
        ax.scatter(predicted[:, 0], predicted[:, 1], predicted[:, 2], c='green', marker='x', label='预测点云', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    return fig

def test(args, net, test_loader, textio):
    with torch.no_grad():
        test_loss, test_rotations_ab, test_translations_ab, \
        test_rotations_ab_pred, \
        test_translations_ab_pred = test_one_epoch(args, net, test_loader)
        print("test_rotations_ab", np.array(test_rotations_ab), np.array(test_rotations_ab_pred))
        print("test_translations_ab", np.array(test_translations_ab), np.array(test_translations_ab_pred))

    pred_transforms = torch.from_numpy(np.concatenate([test_rotations_ab_pred, test_translations_ab_pred.reshape(-1,3,1)], axis=-1))
    gt_transforms = torch.from_numpy(np.concatenate([test_rotations_ab, test_translations_ab.reshape(-1,3,1)], axis=-1))
    concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = (torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi).detach().cpu().numpy()
    residual_transmag = concatenated[:, :, 3].norm(dim=-1).detach().cpu().numpy()

    deg_mean = np.mean(residual_rotdeg)
    deg_rmse = np.sqrt(np.mean(residual_rotdeg**2))

    trans_mean = np.mean(residual_transmag)
    trans_rmse = np.sqrt(np.mean(residual_transmag**2))

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_eulers_ab = npmat2euler(test_rotations_ab)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - test_eulers_ab) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - test_eulers_ab))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)

    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

    textio.cprint('==单个PKL测试结果==')
    textio.cprint('A----------->B')
    textio.cprint('Loss: %f, rot_MSE: %f, rot_RMSE: %f, '\
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f, deg_mean: %f, deg_rmse: %f, trans_mean: %f, trans_rmse: %f: '\
                  % (test_loss, test_r_mse_ab, test_r_rmse_ab, test_r_mae_ab, 
                     test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab,
                     deg_mean, deg_rmse, trans_mean, trans_rmse))
    
    # 单独打印旋转和平移的比较结果，便于分析
    textio.cprint('\n真实旋转矩阵:')
    for rot in test_rotations_ab:
        textio.cprint(np.array2string(rot, precision=4, suppress_small=True))
        
    textio.cprint('\n预测旋转矩阵:')
    for rot in test_rotations_ab_pred:
        textio.cprint(np.array2string(rot, precision=4, suppress_small=True))
        
    textio.cprint('\n真实平移向量:')
    for trans in test_translations_ab:
        textio.cprint(np.array2string(trans, precision=4, suppress_small=True))
        
    textio.cprint('\n预测平移向量:')
    for trans in test_translations_ab_pred:
        textio.cprint(np.array2string(trans, precision=4, suppress_small=True))
    
    # 返回主要指标和预测结果，方便后续处理
    return {
        'test_loss': test_loss,
        'rot_rmse': test_r_rmse_ab,
        'trans_rmse': test_t_rmse_ab,
        'deg_mean': deg_mean,
        'trans_mean': trans_mean,
        'rotations_ab': test_rotations_ab,
        'translations_ab': test_translations_ab,
        'rotations_ab_pred': test_rotations_ab_pred,
        'translations_ab_pred': test_translations_ab_pred
    }

def save_results_as_cif(dataset, results, output_dir, source_cif_path=None):
    """将测试结果保存为CIF格式"""
    # 检查是否提供了源CIF文件
    if source_cif_path:
        # 从源CIF文件解析原子信息
        try:
            from cif_to_point_cloud import parse_cif
            atoms, atom_types, atom_labels, cif_content, atom_lines = parse_cif(source_cif_path)
            print(f"成功从源CIF文件解析得到{len(atoms)}个原子")
        except Exception as e:
            print(f"解析源CIF文件失败: {e}")
            return
    # 否则检查数据集是否包含CIF相关数据
    elif (not hasattr(dataset, 'atom_types') or not dataset.atom_types or
          not hasattr(dataset, 'atom_labels') or not dataset.atom_labels or
          not hasattr(dataset, 'cif_content') or not dataset.cif_content or
          not hasattr(dataset, 'atom_lines') or not dataset.atom_lines):
        print("警告: 缺少CIF相关数据，无法生成CIF输出文件")
        return
    else:
        # 使用数据集中的CIF相关数据
        atoms = None  # 不需要，我们直接使用点云数据
        atom_types = dataset.atom_types
        atom_labels = dataset.atom_labels
        cif_content = dataset.cif_content
        atom_lines = dataset.atom_lines
    
    # 获取源文件名（不带路径和扩展名）
    if source_cif_path:
        base_name = os.path.splitext(os.path.basename(source_cif_path))[0]
    else:
        base_name = os.path.splitext(os.path.basename(dataset.pkl_path))[0]
    
    # 创建保存CIF文件的路径
    source_cif_path_out = os.path.join(output_dir, f"{base_name}_source.cif")
    target_cif_path = os.path.join(output_dir, f"{base_name}_target.cif")
    predicted_cif_path = os.path.join(output_dir, f"{base_name}_predicted.cif")
    
    # 获取点云数据
    source = dataset.source.cpu().numpy()
    target = dataset.target.cpu().numpy()
    
    # 确保点云格式为[N, 3]
    if source.shape[0] == 3 and source.shape[1] != 3:
        source = source.T
    if target.shape[0] == 3 and target.shape[1] != 3:
        target = target.T
    
    # 检查点云和原子数量是否匹配
    if len(source) != len(atom_types):
        print(f"警告: 点云点数({len(source)})与原子数量({len(atom_types)})不匹配")
        # 尝试对齐数据长度 - 截取较短的长度
        min_length = min(len(source), len(atom_types))
        source = source[:min_length]
        atom_types = atom_types[:min_length]
        atom_labels = atom_labels[:min_length]
        atom_lines = atom_lines[:min_length]
        print(f"已截取数据至长度: {min_length}")
    
    # 使用预测的旋转和平移矩阵变换源点云
    rotation_pred = results['rotations_ab_pred'][0]  # 使用第一个预测（通常只有一个）
    translation_pred = results['translations_ab_pred'][0]
    
    # 转换为numpy数组
    source_tensor = torch.tensor(source, dtype=torch.float32).transpose(0, 1).unsqueeze(0)  # [1, 3, N]
    rotation_tensor = torch.tensor(rotation_pred, dtype=torch.float32).unsqueeze(0)  # [1, 3, 3]
    translation_tensor = torch.tensor(translation_pred, dtype=torch.float32).unsqueeze(0)  # [1, 3]
    
    # 应用变换
    predicted_tensor = transform_point_cloud(source_tensor, rotation_tensor, translation_tensor)
    predicted = predicted_tensor.squeeze(0).transpose(0, 1).cpu().numpy()  # [N, 3]
    
    # 保存源点云的CIF文件
    save_as_cif(source, atom_types, atom_labels, 
                cif_content, atom_lines, source_cif_path_out)
    
    # 保存目标点云的CIF文件
    save_as_cif(target, atom_types, atom_labels,
                cif_content, atom_lines, target_cif_path)
    
    # 保存预测点云的CIF文件
    save_as_cif(predicted, atom_types, atom_labels,
                cif_content, atom_lines, predicted_cif_path)
    
    print(f"CIF格式结果已保存至 {output_dir} 目录")
    
    # 生成可视化图像
    fig = visualize_point_clouds(source, target, predicted, title=f"测试结果: {base_name}")
    fig_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"可视化图像已保存至: {fig_path}")

def parse_args_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return EasyDict(config)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试单个PKL文件')
    parser.add_argument('--config', type=str, default='config/train-ca.yaml', help='配置文件路径')
    parser.add_argument('--model_path', type=str, default='', help='模型路径')
    parser.add_argument('--pkl_path', type=str, required=True, help='要测试的单个PKL文件路径')
    parser.add_argument('--output_dir', type=str, default='test_results', help='输出结果的目录')
    parser.add_argument('--save_cif', action='store_true', help='是否保存CIF格式的结果')
    parser.add_argument('--source_cif', type=str, default='', help='源CIF文件路径，用于生成CIF格式结果')
    
    cmd_args = parser.parse_args()
    
    # 加载YAML配置
    args = parse_args_from_yaml(cmd_args.config)
    
    # 命令行参数覆盖YAML配置
    if cmd_args.model_path:
        args.model_path = cmd_args.model_path
    if not args.model_path:
        args.model_path = 'checkpoints/' + args.exp_name + '/models/model.best.t7'
    
    # 创建输出目录
    if not os.path.exists(cmd_args.output_dir):
        os.makedirs(cmd_args.output_dir)
    
    # 设置日志输出
    pkl_name = os.path.basename(cmd_args.pkl_path).replace('.pkl', '')
    log_file = os.path.join(cmd_args.output_dir, f'test_{pkl_name}.log')
    textio = IOStream(log_file)
    
    # 设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # 输出基本信息
    textio.cprint(f"配置: {args}")
    textio.cprint(f"测试文件: {cmd_args.pkl_path}")
    textio.cprint(f"模型路径: {args.model_path}")
    if cmd_args.source_cif:
        textio.cprint(f"源CIF文件: {cmd_args.source_cif}")
    
    # 加载模型
    if args.model == 'RIENET':
        net = RIENET(args).cuda()
        
        if not os.path.exists(args.model_path):
            textio.cprint(f"错误: 找不到预训练模型 {args.model_path}")
            return
        
        checkpoint = torch.load(args.model_path)
        textio.cprint(f"加载模型 epoch {checkpoint['epoch']}, 最佳结果: {checkpoint['best_result']}")
        net.load_state_dict(checkpoint['model'], strict=False)
        
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            textio.cprint(f"使用 {torch.cuda.device_count()} GPUs!")
    else:
        textio.cprint(f"错误: 不支持的模型类型 {args.model}")
        return
    
    # 创建数据集和加载器
    test_data = SinglePklDataset(cmd_args.pkl_path)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # 执行测试
    textio.cprint("开始测试...")
    results = test(args, net, test_loader, textio)
    
    # 如果指定了保存CIF格式，则保存结果为CIF文件
    if cmd_args.save_cif:
        textio.cprint("正在生成CIF格式结果文件...")
        save_results_as_cif(test_data, results, cmd_args.output_dir, cmd_args.source_cif)
    
    textio.cprint("\n测试完成！结果已保存到 " + log_file)
    textio.close()

if __name__ == '__main__':
    main()
