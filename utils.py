# -*- coding: utf-8 -*-
'''
@time: 2019/9/12 15:16

@ author: javis
'''
import torch
import numpy as np
import time,os, math
from sklearn.metrics import f1_score
from torch import nn
from scipy import stats


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

#计算F1score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    # print("y_true:", y_true)
    # print("y_pre:", y_pre)
    # print("shape of y_true:", y_true.shape)
    # print("shape of y_pre:", y_pre.shape)
    return f1_score(y_true, y_pre)

#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()


def time_feature(d):
    """
    获取时域特征
    :param data: np.array.时域信号
    :return: timeFeatList: 时域特征
    """
    feats = []
    for i in range(d.shape[0]):
        temp = []
        for j in range(d.shape[1]):
            data = d[i, j, :]
            dfMean = np.mean(data)  # 均值
            dfVar = np.var(data)  # 方差
            dfStd = np.std(data)  # 标准差
            dfRMS = math.sqrt(pow(dfMean, 2) + pow(dfStd, 2))  # 均方根
            dataSorted = np.sort(np.abs(data))
            dfPeak = np.mean(dataSorted[-10:])
            dfSkew = stats.skew(data)  # 偏度
            dfKurt = stats.kurtosis(data)  # 峭度
            # dfBoxingFactor = dfRMS / np.mean(np.abs(data))
            dfPeakFactor = dfPeak / dfRMS  # 峰值因子
            dfPulseFactor = dfPeak / np.mean(np.abs(data))  # 脉冲因子

            timeFeats = [dfMean, dfVar, dfRMS, dfPeak, dfSkew, dfKurt, dfPeakFactor, dfPulseFactor]
            # print(timeFeats)
            # break
            temp.append(timeFeats)
        feats.append(temp)
    feats = np.array(feats)
    feats = feats.reshape((d.shape[0], -1))
    return feats
