#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/24 上午10:55
# @Author  : ruyueshuo
# @File    : main.py
# @Software: PyCharm
# LIBRARIES

from __future__ import print_function

from options import Options
from dataset import load_data
from model import Ecg
from data_process import train,name2index,config
##
# def main():
""" Training
"""
# torch.manual_seed(config.seed)
# torch.cuda.manual_seed(config.seed)
##
name2idx = name2index(config.arrythmia)
idx2name = {idx: name for name, idx in name2idx.items()}
train(name2idx, idx2name)
# ARGUMENTS
opt = Options().parse()

##
# LOAD DATA
train_dataset, train_dataloader, val_dataset, val_dataloader = load_data(opt)

##
# LOAD MODEL
model = Ecg(opt, train_dataset, train_dataloader, val_dataset, val_dataloader)

##
# TRAIN MODEL
model.train()

# TEST
model.test()

# if __name__ == '__main__':
#     main()

