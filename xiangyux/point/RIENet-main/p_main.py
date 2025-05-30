#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import sys
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from model import RIENET
from util import npmat2euler
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import yaml
from easydict import EasyDict
from common.math import se3
from common.math_torch import se3
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import glob
class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    if not args.eval:
        os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
        os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
        os.system('cp utils.py checkpoints' + '/' + args.exp_name + '/' + 'utils.py.backup')

def test_one_epoch(args, net, test_loader):
    net.eval()

    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []
    
    for src, target, rotation_ab, translation_ab,grid_shape,x_origin,y_origin,z_origin,x_voxel,nstart in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()

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

def train_one_epoch(args, net, train_loader, opt):
    net.train()
    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []
    #source, target, rotation, translation,gird_shape,x_origin,y_origin,z_origin,x_voxel,nstart
    for src, target, rotation_ab, translation_ab,grid_shape,x_origin,y_origin,z_origin,x_voxel,nstart in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()

        #常数参数不需要放入gpu，但是需要用这些计算cc损失
        '''
        grid_shape = torch.tensor(grid_shape, dtype=torch.long).to(device)
        x_origin = torch.tensor(x_origin, dtype=torch.float32).to(device)
        y_origin = torch.tensor(y_origin, dtype=torch.float32).to(device)
        z_origin = torch.tensor(z_origin, dtype=torch.float32).to(device)
        x_voxel = torch.tensor(x_voxel, dtype=torch.float32).to(device)
        nstart = torch.tensor(nstart, dtype=torch.long).to(device)
        '''

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size

        rotation_ab_pred, translation_ab_pred,\
            loss1, loss2, loss3,loss4 = net(src, target,grid_shape,x_origin,y_origin,z_origin,x_voxel,nstart)
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        loss = loss1.sum() + loss2.sum() + loss3.sum() +loss4.sum()

        print('Loss1: %f, Loss2: %f, Loss3: %f, Loss4: %f'
                  % (loss1.sum(), loss2.sum(), loss3.sum(),loss4.sum()))
        loss.backward()
        opt.step()
        total_loss += loss.item()
        torch.cuda.empty_cache()

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    return total_loss * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred

def test(args, net, test_loader, boardio, textio):
    with torch.no_grad():
        test_loss, test_rotations_ab, test_translations_ab, \
        test_rotations_ab_pred, \
        test_translations_ab_pred = test_one_epoch(args, net, test_loader)
        print("test_rotations_ab",np.array(test_rotations_ab),np.array(test_rotations_ab_pred))
        print("test_translations_ab", np.array(test_translations_ab), np.array(test_translations_ab_pred))

    pred_transforms = torch.from_numpy(np.concatenate([test_rotations_ab_pred,test_translations_ab_pred.reshape(-1,3,1)], axis=-1))
    gt_transforms = torch.from_numpy(np.concatenate([test_rotations_ab,test_translations_ab.reshape(-1,3,1)], axis=-1))
    concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = (torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi).detach().cpu().numpy()
    residual_transmag = concatenated[:, :, 3].norm(dim=-1).detach().cpu().numpy()

    deg_mean = np.mean(residual_rotdeg) #/.////
    deg_rmse = np.sqrt(np.mean(residual_rotdeg**2))

    trans_mean = np.mean(residual_transmag) #/.////
    trans_rmse = np.sqrt(np.mean(residual_transmag**2))

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_eulers_ab = npmat2euler(test_rotations_ab)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - test_eulers_ab) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - test_eulers_ab))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    # from sklearn.metrics import r2_score
    # r_ab_r2_score = r2_score(test_eulers_ab, test_rotations_ab_pred_euler)
    # t_ab_r2_score = r2_score(test_translations_ab, test_translations_ab_pred)

    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))


    textio.cprint('==FINAL TEST==')
    textio.cprint('A--------->B')
    textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f,'
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f, deg_mean: %f, deg_rmse: %f, trans_mean: %f, trans_rmse: %f: '
                  % (-1, test_loss,
                     test_r_mse_ab, test_r_rmse_ab,
                     test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab,
                     deg_mean,deg_rmse,trans_mean,trans_rmse))

def train(args, net, train_loader, test_loader, boardio, textio):
    checkpoint = None
    if args.resume:
        textio.cprint("start resume from checkpoint...........")
        if args.model_path == '':
            model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",model_path)
        else:
            model_path = args.model_path
            print(model_path)
        if not os.path.exists(model_path):
            print("can't find pretrained model")
            return
        checkpoint = torch.load(model_path)
        args.start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model'], strict=False)
        textio.cprint("end resume from checkpoint!!!!!!!!!!!!!!")
    best_test_r_mse_ab = np.inf
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = MultiStepLR(opt,
                            milestones=[int(i) for i in args.lr_step],
                            gamma=0.3)

    if checkpoint is not None:
        best_test_r_mse_ab = checkpoint['best_result']
        print(best_test_r_mse_ab)
        opt.load_state_dict(checkpoint['optimizer'])
    best_test_loss = np.inf

    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf

    best_deg_mean = np.inf
    best_deg_rmse = np.inf
    best_trans_mean = np.inf
    best_trans_rmse = np.inf
    best_r = []
    best_t = []
    r = []
    t = []
    for epoch in range(args.epochs):
        scheduler.step()
        train_loss, train_rotations_ab, train_translations_ab, \
        train_rotations_ab_pred, train_translations_ab_pred = train_one_epoch(args, net, train_loader, opt)
        #print("!!!!!!!!!!!!!!!!!!", train_rotations_ab, train_rotations_ab_pred)
        #print("!!!!!!!!!!!!!!!!!!", train_translations_ab,  train_translations_ab_pred)
        
        with torch.no_grad():
            test_loss, test_rotations_ab, test_translations_ab, \
            test_rotations_ab_pred, \
            test_translations_ab_pred = test_one_epoch(args, net, test_loader)
            r = test_rotations_ab_pred
            t = test_translations_ab_pred

        pred_transforms = torch.from_numpy(np.concatenate([test_rotations_ab_pred,test_translations_ab_pred.reshape(-1,3,1)], axis=-1))
        gt_transforms = torch.from_numpy(np.concatenate([test_rotations_ab,test_translations_ab.reshape(-1,3,1)], axis=-1))
        concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = (torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi).detach().cpu().numpy()
        residual_transmag = concatenated[:, :, 3].norm(dim=-1).detach().cpu().numpy()

        deg_mean = np.mean(residual_rotdeg) #/.////
        deg_rmse = np.sqrt(np.mean(residual_rotdeg**2))

        trans_mean = np.mean(residual_transmag) #/.////
        trans_rmse = np.sqrt(np.mean(residual_transmag**2))

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_eulers_ab = npmat2euler(train_rotations_ab)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - train_eulers_ab) ** 2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - train_eulers_ab))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

        test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
        test_eulers_ab = npmat2euler(test_rotations_ab)
        test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - test_eulers_ab) ** 2)
        test_r_rmse_ab = np.sqrt(test_r_mse_ab)
        test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - test_eulers_ab))
        test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
        test_t_rmse_ab = np.sqrt(test_t_mse_ab)
        test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))


        if best_test_loss >= test_loss:
            best_test_loss = test_loss
    
            best_test_r_mse_ab = test_r_mse_ab
            best_test_r_rmse_ab = test_r_rmse_ab
            best_test_r_mae_ab = test_r_mae_ab

            best_test_t_mse_ab = test_t_mse_ab
            best_test_t_rmse_ab = test_t_rmse_ab
            best_test_t_mae_ab = test_t_mae_ab

            best_deg_mean = deg_mean
            best_deg_rmse = deg_rmse
            best_trans_mean = trans_mean
            best_trans_rmse = trans_rmse
            best_r = r
            best_t = t
            print("best_r",best_r)
            print("best_t", best_t)
            if torch.cuda.device_count() > 1:
                state = {'model':net.module.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'best_result':best_test_r_mse_ab}
                torch.save(state, 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                state = {'model':net.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'best_result':best_test_r_mse_ab}
                torch.save(state, 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, train_loss, train_r_mse_ab,
                         train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f , deg_mean: %f, deg_rmse: %f, trans_mean: %f, trans_rmse: %f: '
                      % (epoch, test_loss, test_r_mse_ab,
                         test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab, deg_mean,deg_rmse,trans_mean,trans_rmse))

        print('==BEST TEST==!!!!!!',best_r)
        print('==BEST TEST==', best_t)

        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f , deg_mean: %f, deg_rmse: %f, trans_mean: %f, trans_rmse: %f: '
                      % (epoch, best_test_loss, best_test_r_mse_ab, best_test_r_rmse_ab,
                         best_test_r_mae_ab, best_test_t_mse_ab, best_test_t_rmse_ab, best_test_t_mae_ab, best_deg_mean,best_deg_rmse,best_trans_mean,best_trans_rmse))
        textio.cprint('==best==')
        boardio.add_scalar('A->B/train/loss', train_loss, epoch)
        boardio.add_scalar('A->B/train/rotation/MSE', train_r_mse_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/RMSE', train_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/MAE', train_r_mae_ab, epoch)
        boardio.add_scalar('A->B/train/translation/MSE', train_t_mse_ab, epoch)
        boardio.add_scalar('A->B/train/translation/RMSE', train_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/translation/MAE', train_t_mae_ab, epoch)

        ############TEST
        boardio.add_scalar('A->B/test/loss', test_loss, epoch)
        boardio.add_scalar('A->B/test/rotation/MSE', test_r_mse_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/RMSE', test_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/MAE', test_r_mae_ab, epoch)
        boardio.add_scalar('A->B/test/translation/MSE', test_t_mse_ab, epoch)
        boardio.add_scalar('A->B/test/translation/RMSE', test_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/translation/MAE', test_t_mae_ab, epoch)

        ############BEST TEST
        boardio.add_scalar('A->B/best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MSE', best_test_r_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/RMSE', best_test_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MAE', best_test_r_mae_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/MSE', best_test_t_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/RMSE', best_test_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/MAE', best_test_t_mae_ab, epoch)

        if torch.cuda.device_count() > 1:
            state ={'model':net.module.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'best_result':best_test_r_mse_ab}
            torch.save(state, 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            state ={'model':net.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'best_result':best_test_r_mse_ab}
            torch.save(state, 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()

def parse_args_from_yaml(yaml_path):
    with open(yaml_path, 'r',encoding='utf-8') as fd:
        args = yaml.safe_load(fd)
        args = EasyDict(d=args)
    return args


class PointCloudDataset(Dataset):
    def __init__(self, data_path):
        # 加载数据
        self.source, self.target, self.rotation, self.translation = load_dataset(data_path)

    def __len__(self):
        # 返回数据集的大小
        return len(self.source)

    def __getitem__(self, idx):
        # 获取索引 idx 对应的数据
        source = self.source
        target = self.target
        rotation = self.rotation
        translation = self.translation
        return source, target, rotation, translation

'''

# 加载数据的函数
def load_dataset(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)

    # 确保数据字典有 'source', 'target', 'rotation', 'translation'
    source = np.array(data_dict['source'], dtype='float32')
    source = source.T
    target = np.array(data_dict['target'], dtype='float32')
    target =target.T
    rotation = np.array(data_dict['rotation'], dtype='float32')
    translation = np.array(data_dict['translation'], dtype='float32')

    return source, target, rotation, translation
'''


def load_dataset(folder_path):
    # 获取文件夹中所有的 .pkl 文件路径
    pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))

    # 如果没有找到 .pkl 文件，抛出错误
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in directory: {folder_path}")

    # 逐个读取 .pkl 文件并返回数据
    for file_path in pkl_files:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)

        # 从数据字典中提取所需数据
        source = np.array(data_dict['source'], dtype='float32')
        target = np.array(data_dict['target'], dtype='float32')
        rotation = np.array(data_dict['rotation'], dtype='float32')
        translation = np.array(data_dict['translation'], dtype='float32')

        # 返回每个文件的数据
        yield source.T, target.T, rotation, translation
class PointCloud_ca_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # 初始化：获取文件夹中的所有 .pkl 文件
        self.data_dir = data_dir
        self.transform = transform
        self.data_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        if not self.data_files:
            raise FileNotFoundError(f"No .pkl files found in directory: {data_dir}")

    def __len__(self):
        # 返回文件夹中的 .pkl 文件个数
        return len(self.data_files)

    def __getitem__(self, idx):
        # 获取文件路径
        data_file = self.data_files[idx]

        # 打开并加载 pkl 文件中的数据
        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f)

        # 从数据字典中提取所需的数据
        source = np.array(data_dict['source'], dtype='float32')
        target = np.array(data_dict['target'], dtype='float32')
        rotation = np.array(data_dict['rotation'], dtype='float32')
        translation = np.array(data_dict['translation'], dtype='float32')

        gird_shape = np.array(data_dict['grid'], dtype='int')

        x_origin = np.array(data_dict[' x_origin'], dtype='float32')
        y_origin = np.array(data_dict[' y_origin'], dtype='float32')
        z_origin = np.array(data_dict[' z_origin'], dtype='float32')
        x_voxel  = np.array(data_dict['x_voxel'], dtype='float32')
        nstart = np.array(  data_dict['x_voxel'], dtype='int')


        source = torch.tensor(source.T, dtype=torch.float32)  # 转置后转换为 tensor
        target = torch.tensor(target.T, dtype=torch.float32)  # 转置后转换为 tensor
        rotation = torch.tensor(rotation, dtype=torch.float32)
        translation = torch.tensor(translation, dtype=torch.float32)

        #gird_shape = torch.tensor(gird_shape, dtype=torch.int)

        #x_origin = torch.tensor(x_origin, dtype=torch.float32)
        #y_origin = torch.tensor(y_origin, dtype=torch.float32)
        #z_origin = torch.tensor(z_origin, dtype=torch.float32)

        #x_voxel = torch.tensor(x_voxel, dtype=torch.float32)

        #nstart = torch.tensor(nstart, dtype=torch.int)

        # 返回一个元组（source, target, rotation, translation）
        return source, target, rotation, translation,gird_shape,x_origin,y_origin,z_origin,x_voxel,nstart
def main():
    args = parse_args_from_yaml(sys.argv[1])
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    if args.dataset == 'ca_point':
        print("load dateset")
        train_path = "/xiangyux/point/RIENet-main/data/ca_map_train"
        train_data = PointCloud_ca_Dataset(train_path)

        train_loader = DataLoader(train_data, args.batch_size, drop_last=True, shuffle=True)
        #test_path = "/xiangyux/point/RIENet-main/data/ca_test"
        test_path = "/xiangyux/point/RIENet-main/data/ca_map_train"
        test_data = PointCloud_ca_Dataset( test_path)

        test_loader = DataLoader(test_data, args.batch_size, drop_last=False, shuffle=False)


    else:
        raise Exception("not implemented")

    if args.model == 'RIENET':
        print("start!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        net = RIENET(args).cuda()
        if args.eval:
            if args.model_path == '':
                model_path = 'pretrained' + '/' + args.exp_name + '/model.best.t7'
                print("model_path!!!!!!!!!", model_path)
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            
            checkpoint = torch.load(model_path)
            print("model_path!!!!!!!!!",model_path)
            print(checkpoint['epoch'],checkpoint['best_result'])
            net.load_state_dict(checkpoint['model'], strict=False)
            textio.cprint("end resume from checkpoint!!!!!!!!!!!!!!")
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')
    if args.eval:
        test(args, net, test_loader, boardio, textio)
    else:
        train(args, net, train_loader, test_loader, boardio, textio)


    print('FINISH')
    #print("model_path!!!!!!!!!", model_path)
    boardio.close()


if __name__ == '__main__':
    main()
