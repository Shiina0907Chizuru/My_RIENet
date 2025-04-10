import mrcfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import io
import os
from collections import namedtuple
from Bio.PDB import  MMCIFParser, PDBParser
from Bio.PDB.Atom import DisorderedAtom
from os.path import join, isfile, isdir, basename, normpath
from os import listdir
from Bio import PDB
import einops
import random
from itertools import product
import pickle
MRCObject = namedtuple("MRCObject", ["grid", "voxel_size", "global_origin", "nstart"])
import cupy as cp

def edcs_loss(rho_em, rho_mo):
    rho_em = rho_em* rho_mo
    print("rho_mo",rho_mo.shape)
    rho_mo_mean = torch.sum( rho_em, dim=(1, 2, 3), keepdim=True)
    rho_mo = rho_mo* rho_mo
    rho_em_mean = torch.sum(    rho_mo, dim=(1, 2, 3), keepdim=True)
    print("rho_mo",rho_em_mean)
    '''

    rho_em_mean = torch.mean(rho_em, dim=(1, 2, 3), keepdim=True)  # 形状 (b, 1, 1, 1)
    rho_mo_mean = torch.mean(rho_mo, dim=(1, 2, 3), keepdim=True)  # 形状 (b, 1, 1, 1)

    # 减去均值
    rho_em_centered = rho_em - rho_em_mean  # 形状 (b, n, n, n)
    rho_mo_centered = rho_mo - rho_mo_mean  # 形状 (b, n, n, n)

    # 分子部分
    numerator = torch.sum(rho_em_centered * rho_mo_centered, dim=(1, 2, 3))  # 形状 (b,)

    # 分母部分
    denominator_em = torch.sum(rho_em_centered ** 2, dim=(1, 2, 3))  # 形状 (b,)
    denominator_mo = torch.sum(rho_mo_centered ** 2, dim=(1, 2, 3))  # 形状 (b,)
    denominator = torch.sqrt(denominator_em * denominator_mo)  # 形状 (b,)
    '''




    # E_DCS
    #edcs = 1 - numerator / denominator  # 形状 (b,)
    edcs =  rho_mo_mean # 形状 (b,)
    # 返回 batch 平均损失
    return torch.mean(edcs)
def normalize_grid(grid):
    # 计算98百分位的值
    minval=np.min(grid)
    p98_value = np.percentile(grid, 99.99)
    # 归一化处理，0.98为max
    grid = (grid- minval) / (p98_value- minval)
    # 将大于98百分位的值设为1
    grid[grid > 1] = 1
    contains_nan = np.isnan(grid).any()

    return grid, contains_nan
def load_Map(input_map_path)-> MRCObject:
    # Read MRC file
    mrc = mrcfile.open(input_map_path, permissive=True)
    grid= mrc.data.copy()
    voxel_size_o = float(mrc.voxel_size.x)
    voxel_size = np.asarray(mrc.voxel_size.tolist(), dtype=np.float32)

    origin = np.array(mrc.header.origin.tolist(), dtype=np.float32)
    nstart = np.asarray([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)

    mapcrs = np.asarray([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], dtype=int)


    mrc.close()

    # Reorder
    sort = np.asarray([0, 1, 2], dtype=np.int64)
    for i in range(3):
        sort[mapcrs[i] - 1] = i
    nstart = np.asarray([nstart[i] for i in sort])

    grid = np.transpose(grid, axes=2 - sort[::-1])
    global_origin = origin
    # Move offsets from nstart to origin if origin is zero (MRC2000)
    if np.all(origin == 0):
        global_origin = origin + nstart * voxel_size




    return MRCObject(grid, voxel_size_o, global_origin,nstart)

def load_pkl_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data_dict = pickle.load(f)


    grid_shape = data_dict['grid']
    x_origin = data_dict[' x_origin']
    y_origin = data_dict[' y_origin']
    z_origin = data_dict[' z_origin']
    x_voxel = data_dict['x_voxel']
    y_voxel=  data_dict['y_voxel']
    z_voxel = data_dict['z_voxel']
    src = data_dict['source']
    tgt=  data_dict['target']
    R = data_dict['rotation']
    T = data_dict['translation']
    nstart = data_dict['nstart']

    return grid_shape,x_origin,y_origin,z_origin,x_voxel,y_voxel,z_voxel,src,tgt,R,T,nstart

def get_index(coord, origin, voxel_size):
    return (coord - origin) / voxel_size


def ca_mask_pdb(grid_shape, x_origin, y_origin, z_origin, x_voxel, y_voxel, z_voxel, src):
    data = np.zeros(grid_shape, dtype=np.float32)  # 使用 float32 类型
    resolution = 3.5
    k = (np.pi / resolution) ** 2
    C = (k / np.pi) ** 1.5
    bw2 = 9 # bandwidth squared
    bw = np.sqrt(bw2)

    # 提前计算源坐标的索引
    coords_indices = np.array([
        [int(get_index(coord[0], x_origin, x_voxel)),
         int(get_index(coord[1], y_origin, y_voxel)),
         int(get_index(coord[2], z_origin, z_voxel))]
        for coord in src
    ])

    # 预计算 coord_lower 和 coord_upper
    coord_lower = np.floor(coords_indices - bw).astype(np.int32)
    coord_upper = np.ceil(coords_indices + bw).astype(np.int32)


    # 使用 np.indices 代替嵌套循环，提高效率
    indices = np.indices(grid_shape)

    for i, coord in enumerate(src):
        iz, jy, kx = coords_indices[i]
        if 0 <= iz < data.shape[0] and 0 <= jy < data.shape[1] and 0 <= kx < data.shape[2]:
            lower = coord_lower[i]
            upper = coord_upper[i]

        # 通过 np.indices 生成整个网格，并计算距离平方
            z_idx, y_idx, x_idx = indices[:, lower[0]:upper[0] + 1, lower[1]:upper[1] + 1, lower[2]:upper[2] + 1]
            dist_sq = (z_idx - iz) ** 2 + (y_idx - jy) ** 2 + (x_idx - kx) ** 2

            # 使用掩码筛选符合距离条件的点
            mask = dist_sq <= bw2
            prob = np.exp(-k * dist_sq) * mask

            # 将符合条件的概率值写入data
            data[z_idx[mask], y_idx[mask], x_idx[mask]] = np.maximum(data[z_idx[mask], y_idx[mask], x_idx[mask]],
                                                                     prob[mask])

    return data
def save_mrc(grid, voxel_size, origin,nstart, filename):
    (z, y, x) = grid.shape
    o = mrcfile.new(filename, overwrite=True)
    o.header["cella"].x = x * voxel_size
    o.header["cella"].y = y * voxel_size
    o.header["cella"].z = z * voxel_size
    o.header["origin"].x = origin[0]
    o.header["origin"].y = origin[1]
    o.header["origin"].z = origin[2]
    (o.header.nxstart, o.header.nystart, o.header.nzstart)=nstart
    out_box = np.reshape(grid, (z, y, x))
    o.set_data(out_box.astype(np.float32))
    o.update_header_stats()
    o.flush()
    o.close()


def get_index_gpu(coord, origin, voxel_size):
    return (coord - origin) / voxel_size


def ca_mask_pdb_gpu(grid_shape, x_origin, y_origin, z_origin, x_voxel, y_voxel, z_voxel, src):
    data = cp.zeros(grid_shape, dtype=cp.float32)  # 使用 GPU 数组
    resolution = 3.5
    k = (cp.pi / resolution) ** 2
    C = (k / cp.pi) ** 1.5
    bw2 = 12  # bandwidth squared
    bw = cp.sqrt(bw2)

    for coord in src:
        x, y, z = coord
        iz = int(get_index_gpu(z, z_origin, z_voxel))
        jy = int(get_index_gpu(y, y_origin, y_voxel))
        kx = int(get_index_gpu(x, x_origin, x_voxel))

        # 确保坐标不超出边界
        if 0 <= iz < data.shape[0] and 0 <= jy < data.shape[1] and 0 <= kx < data.shape[2]:
            #coord_lower = cp.floor([iz, jy, kx] - bw).astype(cp.int32)
            coord_lower = cp.floor(cp.array([iz, jy, kx]) - bw).astype(cp.int32)
            #coord_upper = cp.ceil([iz, jy, kx] + bw).astype(cp.int32)
            # 将 [iz, jy, kx] 转换为 cupy 数组，并与 bw 相加
            coord_upper = cp.ceil(cp.array([iz, jy, kx]) + bw).astype(cp.int32)

            for z in range(int(coord_lower[0]), int(coord_upper[0])):
                for y in range(int(coord_lower[1]), int(coord_upper[1])):
                    for x in range(int(coord_lower[2]), int(coord_upper[2])):
                        if 0 <= z < data.shape[0] and 0 <= y < data.shape[1] and 0 <= x < data.shape[2]:
                            dist_sq = (z - iz) ** 2 + (y - jy) ** 2 + (x - kx) ** 2
                            if dist_sq <= bw2:
                                prob = cp.exp(-k * dist_sq)
                                data[z, y, x] = cp.maximum(data[z, y, x], prob)

    return data.get()  # 从 GPU 数组返回 NumPy 数组
# 示例：使用该损失函数
if __name__ == "__main__":
    pkl_file = "/xiangyux/point/RIENet-main/data/ca_map_train/PDB-5jzhA_point.pkl"
    grid_shape, x_origin, y_origin, z_origin, x_voxel, y_voxel, z_voxel,src,tgt,R,t,nstart= load_pkl_data(pkl_file)
    #src = np.dot(R, src.T).T + t

    src_mask =  ca_mask_pdb( grid_shape, x_origin, y_origin, z_origin, x_voxel, y_voxel, z_voxel,src)
    tgt_mask =  ca_mask_pdb( grid_shape, x_origin, y_origin, z_origin, x_voxel, y_voxel, z_voxel,tgt)
    #out_all_map = "/xiangyux/point/RIENet-main/data/ca_zheng_train/tgt.map"
    #out_chain_map = "/xiangyux/point/RIENet-main/data/ca_zheng_train/src.map"
    #save_mrc( tgt_mask, x_voxel, (x_origin, y_origin, z_origin), nstart, out_all_map)
    #save_mrc(src_mask, x_voxel, (x_origin, y_origin, z_origin), nstart, out_chain_map)
    srcc = torch.tensor(  src_mask, dtype=torch.float32)
    srcc = srcc.unsqueeze(0)

    tgt= torch.tensor(tgt_mask ,dtype=torch.float32)
    tgt = tgt.unsqueeze(0)

    loss_fn = edcs_loss(  tgt,     srcc)
    print("loss_fn",loss_fn.item())


