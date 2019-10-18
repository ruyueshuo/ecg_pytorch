#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 上午10:21
# @Author  : ruyueshuo
# @File    : options.py
# @Software: PyCharm
import os
import argparse


class Options(object):
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # root = r'data'


        # Base
        self.parser.add_argument('--root', default=r'./data', help='root of data')
        self.parser.add_argument('--model_name', default='resnet18', help='name of model')
        self.parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
        self.parser.add_argument('--stage_epoch', default=[16, 32, 64], help='在第几个epoch进行到下一个state,调整lr')
        self.parser.add_argument('--num_classes', type=int, default=55, help='label的类别数.')
        self.parser.add_argument('--max_epoch', type=int, default=256, help='最大训练多少代')
        self.parser.add_argument('--target_point_num', type=int, default=2048, help='采样长度')
        self.parser.add_argument('--ckpt', type=str, default='user_data')
        self.parser.add_argument('--sub_dir', type=str, default='prediction_result')
        self.parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
        self.parser.add_argument('--current_w', type=str, default='current_w.pth')
        self.parser.add_argument('--best_w', type=str, default='best_w.pth')
        self.parser.add_argument('--lr_decay', type=float, default=10., help='decay speed of learning rate')
        self.parser.add_argument('--seed', type=int, default=2019, help='random seed')

        # self.parser.add_argument("command", metavar="<command>", help="train or test")
        # self.parser.add_argument("--ckpt", type=str, help="path of model weight file")
        self.parser.add_argument("--ex", type=str, help="experience name")
        self.parser.add_argument("--resume", action='store_true', default=False)

        self.isTrain = True
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        self.opt.train_dir = os.path.join(self.opt.root, 'hf_round1_train')
        self.opt.tst_dir = os.path.join(self.opt.root, 'hf_round1_testA')
        self.opt.test_dir = os.path.join(self.opt.root, 'hefei_round1_testB_noDup_rename')
        self.opt.train_label = os.path.join(self.opt.root, 'hf_round1_label.txt')
        self.opt.test_label = os.path.join(self.opt.root, 'hf_round1_subB_noDup_rename.txt')
        self.opt.arrythmia = os.path.join(self.opt.root, 'hf_round1_arrythmia.txt')
        self.opt.train_data = os.path.join(self.opt.ckpt, 'train.pth')

        return self.opt


class Config:
    # for data_process.py
    #root = r'D:\ECG'
    root = r'data'
    train_dir = os.path.join(root, 'hf_round1_train')
    tst_dir = os.path.join(root, 'hf_round1_testA')
    test_dir = os.path.join(root, 'hf_round1_testA')
    train_label = os.path.join(root, 'hf_round1_label.txt')
    test_label = os.path.join(root, 'hf_round1_subB_noDup_rename.txt')
    arrythmia = os.path.join(root, 'hf_round1_arrythmia.txt')
    train_data = os.path.join('user_data', 'train.pth')

    # for train
    #训练的模型名称
    model_name = 'resnet18'
    # model_name2 = 'densenet121'
    # model_name = 'Net'
    #在第几个epoch进行到下一个state,调整lr
    stage_epoch = [16,32,64]
    #训练时的batch大小
    batch_size = 64
    #label的类别数
    num_classes = 55
    #最大训练多少个epoch
    max_epoch = 256
    #目标的采样长度
    target_point_num = 2048
    #保存模型的文件夹
    ckpt = 'user_data/'
    #保存提交文件的文件夹
    sub_dir = 'perdiction_result/'
    #初始的学习率
    lr = 1e-3
    #保存模型当前epoch的权重
    current_w = 'current_w.pth'
    #保存最佳的权重
    best_w = 'best_w.pth'
    # 学习率衰减 lr/=lr_decay
    lr_decay = 10

    seed = 2019

    #for test
    temp_dir=os.path.join(root,'temp')


config = Config()
