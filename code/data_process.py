#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 上午10:21
# @Author  : ruyueshuo
# @File    : data_process.py
# @Software: PyCharm
import os, torch
import numpy as np
from options import config

# 保证每次划分数据一致
np.random.seed(config.seed)


def name2index(path):
    '''
    把类别名称转换为index索引
    :param path: 文件路径
    :return: 字典
    '''
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    return name2indx


def split_data(file2idx, val_ratio=0.1):
    '''
    划分数据集,val需保证每类至少有1个样本
    :param file2idx:
    :param val_ratio:验证集占总数据的比例
    :return:训练集，验证集路径
    '''
    data1 = set(os.listdir(config.train_dir))
    data2 = set(os.listdir(config.test_dir))
    data = data1.union(data2)
    val = set()
    idx2file = [[] for _ in range(config.num_classes)]
    for file, list_idx in file2idx.items():
        for idx in list_idx:
            idx2file[idx].append(file)
    for item in idx2file:
        # print(len(item), item)
        num = int(len(item) * val_ratio)
        val = val.union(item[:num])
    train = data.difference(val)
    return list(train), list(val)


# def split_data_kfold(file2idx, val_ratio=0.1):
#     '''
#     划分数据集,val需保证每类至少有1个样本
#     :param file2idx:
#     :param val_ratio:验证集占总数据的比例
#     :return:训练集，验证集路径
#     '''
#     data = set(os.listdir(config.train_dir))
#     val = set()
#     idx2file = [[] for _ in range(config.num_classes)]
#     for file, list_idx in file2idx.items():
#         for idx in list_idx:
#             idx2file[idx].append(file)
#     for item in idx2file:
#         # print(len(item), item)
#         num = int(len(item) * val_ratio)
#         val = val.union(item[:num])
#     train = data.difference(val)
#     return list(train), list(val)


def file2index(path, name2idx):
    '''
    获取文件id对应的标签类别
    :param path:文件路径
    :return:文件id对应label列表的字段
    '''
    file2index = dict()
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]
        labels = [name2idx[name] for name in arr[3:]]
        # print(id, labels)
        file2index[id] = labels
    return file2index


def count_labels(data, file2idx):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0] * config.num_classes
    for ii, fp in enumerate(data):
        print(ii, fp)
        # if fp[-3:] == 'txt':
        for i in file2idx[fp]:
            print(i)
            cc[i] += 1
    return np.array(cc)


def train(name2idx, idx2name):
    train_label_path = add()
    file2idxtrn = file2index(train_label_path, name2idx)
    test_label = os.path.join('data', 'hf_round1_subB_noDup_rename.txt')
    file2idxtst = file2index(test_label, name2idx)
    file2idx = dict(file2idxtrn, **file2idxtst)
    train, val = split_data(file2idx)
    wc=count_labels(train,file2idx)
    print(wc)
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx,'wc':wc}
    torch.save(dd, config.train_data)

def add():
    root = r'data'
    train_label = os.path.join(root, 'hf_round1_label.txt')
    test_label = os.path.join(root, 'hefei_round1_ansA_20191008.txt')
    with open(train_label, encoding="utf-8") as trn_f:
        trn = trn_f.read()
    with open(test_label, encoding="utf-8") as tst_f:
        tst = tst_f.read()
    # print(trn[0])
    train_label = trn + tst
    train_label_path = os.path.join('user_data', 'trn_label.txt')
    with open(train_label_path, 'w') as f:  # 设置文件对象
        f.write(train_label)
    return train_label_path

if __name__ == '__main__':
    root = r'data'
    train_label = os.path.join(root, 'hf_round1_label.txt')
    test_label = os.path.join(root, 'hefei_round1_ansA_20191008.txt')
    with open(train_label, encoding="utf-8") as trn_f:
        trn = trn_f.read()
    with open(test_label, encoding="utf-8") as tst_f:
        tst = tst_f.read()
    print(trn[0])
    f = trn + tst
    # for line in open(train_label, encoding='utf-8'):
    #     arr = line.strip().split('\t')
    pass
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    train(name2idx, idx2name)
