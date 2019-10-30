#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/10 上午9:30
# @Author  : ruyueshuo
# @File    : net.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.densenet import densenet121
from models.resnet import resnet18


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.rnet = resnet18()
        self.dnet = densenet121()

    def forward(self, x):

        rout = self.rnet(x)
        dout = self.dnet(x)

        out = torch.add(rout, dout) / 2

        return out