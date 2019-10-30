#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/18 上午10:05
# @Author  : FengDa
# @File    : test.py
# @Software: PyCharm
import pandas as pd
from options import Options
from dataset import load_data
from model import Ecg
from data_process import train,name2index
##

# torch.manual_seed(config.seed)
# torch.cuda.manual_seed(config.seed)
##

# name2idx = name2index(config.arrythmia)
# idx2name = {idx: name for name, idx in name2idx.items()}
# print(idx2name)
# train(name2idx, idx2name)
# ARGUMENTS
opt = Options().parse()
##
# LOAD DATA
# train_dataset, train_dataloader, val_dataset, val_dataloader = load_data(opt)

##
# LOAD MODEL
model = Ecg(opt, None, None, None, None)

##
# TRAIN MODEL
# model.train()
print("test started.")
# TEST
model.test()
print("test finished.")

