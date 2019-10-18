#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 上午10:21
# @Author  : ruyueshuo
# @File    : dataset.py
# @Software: PyCharm

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy import signal

from options import Options,config


def add_feature(df):
    df['III'] = df['II'] - df['I']
    df['aVR'] = (df['II'] + df['I']) / 2
    df['aVL'] = (df['I'] - df['II']) / 2
    df['aVF'] = (df['II'] - df['I']) / 2
    return df

def resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig

def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise

def verflip(sig):
    '''
    信号竖直翻转
    :param sig:
    :return:
    '''
    return sig[::-1, :]

def shift(sig, interval=20):
    '''
    上下平移
    :param sig:
    :return:
    '''
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig


def min_max(data):
    min = data.min()
    max = data.max()
    return (data - min) / (max - min)


def transform(sig, train=False):
    # 前置不可或缺的步骤
    sig = resample(sig, config.target_point_num)
    # # 数据增强
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = verflip(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    # 后置不可或缺的步骤
    sig = sig.transpose()
    # 归一化
    # sig = min_max(sig)
    # print(sig)
    # print("shape of sig:", sig.shape)
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, config, train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load(config.train_data)
        # d2 = torch.load(config.test_dir)
        self.config = config
        self.train = train
        self.data = dd['train'] if train else dd['val']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1. / np.log(dd['wc'])

    def __getitem__(self, index):
        fid = self.data[index]
        file_path = os.path.join(self.config.train_dir, fid)
        if not os.path.exists(file_path):
            file_path = os.path.join(self.config.tst_dir, fid)
        df = pd.read_csv(file_path, sep=' ')
        df = add_feature(df).values
        # df = pd.read_csv(file_path, sep=' ').values
        x = transform(df, self.train)
        target = np.zeros(self.config.num_classes)
        target[self.file2idx[fid]] = 1
        target = torch.tensor(target, dtype=torch.float32)
        return x, target

    def __len__(self):
        return len(self.data)


def load_data(opt):
    from torch.utils.data import DataLoader
    train_dataset = ECGDataset(config=opt, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    val_dataset = ECGDataset(config=opt, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=0)
    print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))

    return train_dataset, train_dataloader, val_dataset, val_dataloader


def plot_ect(data):
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.show()
    pass


if __name__ == '__main__':
    d = ECGDataset(config.train_data)
    print(d[0][0].size()[0])
    print(d[0][1].size())
    print(d[0][0][0].size())
    for i in range(d[0][0].size()[0]):

        plot_ect(np.array(d[0][0][i]))
