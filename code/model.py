#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 上午10:21
# @Author  : ruyueshuo
# @File    : model.py
# @Software: PyCharm

import torch, time, os, shutil
import pandas as pd
from tensorboard_logger import Logger
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

import models, utils
from dataset import ECGDataset, add_feature, transform
from data_process import name2index
# from options import config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.manual_seed(config.seed)
# torch.cuda.manual_seed(config.seed)


class Ecg(object):

    @staticmethod
    def name():
        """Return name of the class.
        """
        return 'Ecg'

    def __init__(self, opt, train_dataset, train_dataloader, val_dataset, val_dataloader):
        super(Ecg, self).__init__()
        ##
        # Initalize variables.
        self.opt = opt
        self.trn_dataloader = train_dataloader
        self.trn_dataset = train_dataset
        self.val_dataloader = val_dataloader
        self.val_dataset = val_dataset
        # self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        # self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.show_interval = 10
        self.threshold = 0.5
        self.show_interval = 100

        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed(self.opt.seed)

    # 保存当前模型的权重，并且更新最佳的模型权重
    def save_ckpt(self, state, is_best, model_save_dir):
        current_w = os.path.join(model_save_dir, self.opt.current_w)
        best_w = os.path.join(model_save_dir, self.opt.best_w)
        torch.save(state, current_w)
        if is_best:
            shutil.copyfile(current_w, best_w)

    def train_epoch(self, model, optimizer, criterion):
        model.train()
        f1_meter, loss_meter, it_count = 0, 0, 0
        for inputs, target in tqdm(self.trn_dataloader):
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            output = model(inputs)
            # print("output:", output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_meter += loss.item()
            it_count += 1
            # print("output: \t target:".format(output, target))
            # print("shape of output:", output.size())
            # print("shape of target:", target.size())
            f1 = utils.calc_f1(target, torch.sigmoid(output))
            # print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
            f1_meter += f1
            if it_count != 0 and it_count % self.show_interval == 0:
                print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
        return loss_meter / it_count, f1_meter / it_count

    def val_epoch(self, model, criterion, threshold=0.5):
        model.eval()
        f1_meter, loss_meter, it_count = 0, 0, 0
        with torch.no_grad():
            for inputs, target in tqdm(self.val_dataloader):
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                output = model(inputs)
                loss = criterion(output, target)
                loss_meter += loss.item()
                it_count += 1
                output = torch.sigmoid(output)
                f1 = utils.calc_f1(target, output, threshold)
                f1_meter += f1
        return loss_meter / it_count, f1_meter / it_count

    def train(self):
        # model
        model = getattr(models, self.opt.model_name)()
        # model1 = getattr(models, self.opt.model_name1)()
        # model2 = getattr(models, self.opt.model_name2)()
        # if self.opt.ckpt and not self.opt.resume:
        #     state = torch.load(self.opt.ckpt, map_location='cpu')
        #     model.load_state_dict(state['state_dict'])
        #     # model1.load_state_dict(state['state_dict'])
        #     # model2.load_state_dict(state['state_dict'])
        #     print('train with pretrained weight val_f1', state['f1'])
        model = model.to(self.device)
        # model1 = model1.to(device)
        # model2 = model2.to(device)
        # data
        # train_dataset = ECGDataset(data_path=self.opt.train_data, train=True)
        # train_dataloader = DataLoader(train_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)
        # val_dataset = ECGDataset(data_path=self.opt.train_data, train=False)
        # val_dataloader = DataLoader(val_dataset, batch_size=self.opt.batch_size, num_workers=0)
        # print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
        # optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=self.opt.lr)
        w = torch.tensor(self.trn_dataset.wc, dtype=torch.float).to(self.device)
        criterion = utils.WeightedMultilabel(w)
        # 模型保存文件夹
        # model_save_dir = '%s/%s_%s' % (self.opt.ckpt, self.opt.model_name, time.strftime("%Y%m%d%H%M"))
        model_save_dir = self.opt.ckpt
        if self.opt.ex:
            model_save_dir += self.opt.ex
        best_f1 = -1
        lr = self.opt.lr
        start_epoch = 1
        stage = 1
        # 从上一个断点，继续训练
        if self.opt.resume:
            if os.path.exists(self.opt.ckpt):  # 这里是存放权重的目录
                model_save_dir = self.opt.ckpt
                current_w = torch.load(os.path.join(self.opt.ckpt, self.opt.current_w))
                best_w = torch.load(os.path.join(model_save_dir, self.opt.best_w))
                best_f1 = best_w['loss']
                start_epoch = current_w['epoch'] + 1
                lr = current_w['lr']
                stage = current_w['stage']
                model.load_state_dict(current_w['state_dict'])
                # 如果中断点恰好为转换stage的点
                if start_epoch - 1 in self.opt.stage_epoch:
                    stage += 1
                    lr /= self.opt.lr_decay
                    utils.adjust_learning_rate(optimizer, lr)
                    model.load_state_dict(best_w['state_dict'])
                print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
        logger = Logger(logdir=model_save_dir, flush_secs=2)
        # =========>开始训练<=========
        for epoch in range(start_epoch, self.opt.max_epoch + 1):
            print("epoch:", epoch)
            since = time.time()
            train_loss, train_f1 = self.train_epoch(model, optimizer, criterion)
            val_loss, val_f1 = self.val_epoch(model, criterion)
            print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
                  % (epoch, stage, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
            logger.log_value('train_loss', train_loss, step=epoch)
            logger.log_value('train_f1', train_f1, step=epoch)
            logger.log_value('val_loss', val_loss, step=epoch)
            logger.log_value('val_f1', val_f1, step=epoch)
            state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                     'stage': stage}
            self.save_ckpt(state, best_f1 < val_f1, model_save_dir)
            best_f1 = max(best_f1, val_f1)
            if epoch in self.opt.stage_epoch:
                stage += 1
                lr /= self.opt.lr_decay
                best_w = os.path.join(model_save_dir, self.opt.best_w)
                model.load_state_dict(torch.load(best_w)['state_dict'])
                print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
                utils.adjust_learning_rate(optimizer, lr)

    # 用于测试加载模型
    def val(self):
        list_threhold = [0.5]
        model = getattr(models, self.opt.model_name)()
        if self.opt.ckpt: model.load_state_dict(torch.load(self.opt.ckpt, map_location='cpu')['state_dict'])
        model = model.to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        # val_dataset = ECGDataset(data_path=self.opt.train_data, train=False)
        # val_dataloader = DataLoader(val_dataset, batch_size=self.opt.batch_size, num_workers=0)
        for threshold in list_threhold:
            val_loss, val_f1 = self.val_epoch(model, criterion, threshold)
            print('threshold %.2f val_loss:%0.3e val_f1:%.3f\n' % (threshold, val_loss, val_f1))

    # 提交结果使用
    def test(self):
        name2idx = name2index(self.opt.arrythmia)
        idx2name = {idx: name for name, idx in name2idx.items()}
        utils.mkdirs(self.opt.sub_dir)
        # model
        model_save_dir = self.opt.ckpt
        best_w = torch.load(os.path.join(model_save_dir, self.opt.best_w))
        model = getattr(models, self.opt.model_name)()
        model.load_state_dict(torch.load(best_w, map_location='cpu')['state_dict'])
        model = model.to(self.device)
        model.eval()
        sub_file = '%s/result.txt' % (self.opt.sub_dir)
        fout = open(sub_file, 'w', encoding='utf-8')
        with torch.no_grad():
            for line in open(self.opt.test_label, encoding='utf-8'):
                fout.write(line.strip('\n'))
                id = line.split('\t')[0]
                file_path = os.path.join(self.opt.test_dir, id)
                # df = pd.read_csv(file_path, sep=' ').values
                df = pd.read_csv(file_path, sep=' ')
                df = add_feature(df).values
                x = transform(df).unsqueeze(0).to(self.device)
                output = torch.sigmoid(model(x)).squeeze().cpu().numpy()
                ixs = [i for i, out in enumerate(output) if out > 0.5]
                for i in ixs:
                    fout.write("\t" + idx2name[i])
                fout.write('\n')
        fout.close()
# # 保存当前模型的权重，并且更新最佳的模型权重
# def save_ckpt(state, is_best, model_save_dir):
#     current_w = os.path.join(model_save_dir, config.current_w)
#     best_w = os.path.join(model_save_dir, config.best_w)
#     torch.save(state, current_w)
#     if is_best:
#         shutil.copyfile(current_w, best_w)
#
#
# def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
#     model.train()
#     f1_meter, loss_meter, it_count = 0, 0, 0
#     for inputs, target in tqdm(train_dataloader):
#         inputs = inputs.to(device)
#         target = target.to(device)
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         # forward
#         output = model(inputs)
#         # print("output:", output)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         loss_meter += loss.item()
#         it_count += 1
#         # print("output: \t target:".format(output, target))
#         # print("shape of output:", output.size())
#         # print("shape of target:", target.size())
#         f1 = utils.calc_f1(target, torch.sigmoid(output))
#         # print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
#         f1_meter += f1
#         if it_count != 0 and it_count % show_interval == 0:
#             print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
#     return loss_meter / it_count, f1_meter / it_count
#
#
# def val_epoch(model, criterion, val_dataloader, threshold=0.5):
#     model.eval()
#     f1_meter, loss_meter, it_count = 0, 0, 0
#     with torch.no_grad():
#         for inputs, target in tqdm(val_dataloader):
#             inputs = inputs.to(device)
#             target = target.to(device)
#             output = model(inputs)
#             loss = criterion(output, target)
#             loss_meter += loss.item()
#             it_count += 1
#             output = torch.sigmoid(output)
#             f1 = utils.calc_f1(target, output, threshold)
#             f1_meter += f1
#     return loss_meter / it_count, f1_meter / it_count
#
#
# def train(args):
#     # model
#     model = getattr(models, config.model_name)()
#     # model1 = getattr(models, config.model_name1)()
#     # model2 = getattr(models, config.model_name2)()
#     if args.ckpt and not args.resume:
#         state = torch.load(args.ckpt, map_location='cpu')
#         model.load_state_dict(state['state_dict'])
#         # model1.load_state_dict(state['state_dict'])
#         # model2.load_state_dict(state['state_dict'])
#         print('train with pretrained weight val_f1', state['f1'])
#     model = model.to(device)
#     # model1 = model1.to(device)
#     # model2 = model2.to(device)
#     # data
#     train_dataset = ECGDataset(data_path=config.train_data, train=True)
#     train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
#     val_dataset = ECGDataset(data_path=config.train_data, train=False)
#     val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)
#     print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
#     # optimizer and loss
#     optimizer = optim.Adam(model.parameters(), lr=config.lr)
#     w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
#     criterion = utils.WeightedMultilabel(w)
#     # 模型保存文件夹
#     model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"))
#     if args.ex:
#         model_save_dir += args.ex
#     best_f1 = -1
#     lr = config.lr
#     start_epoch = 1
#     stage = 1
#     # 从上一个断点，继续训练
#     if args.resume:
#         if os.path.exists(args.ckpt):  # 这里是存放权重的目录
#             model_save_dir = args.ckpt
#             current_w = torch.load(os.path.join(args.ckpt, config.current_w))
#             best_w = torch.load(os.path.join(model_save_dir, config.best_w))
#             best_f1 = best_w['loss']
#             start_epoch = current_w['epoch'] + 1
#             lr = current_w['lr']
#             stage = current_w['stage']
#             model.load_state_dict(current_w['state_dict'])
#             # 如果中断点恰好为转换stage的点
#             if start_epoch - 1 in config.stage_epoch:
#                 stage += 1
#                 lr /= config.lr_decay
#                 utils.adjust_learning_rate(optimizer, lr)
#                 model.load_state_dict(best_w['state_dict'])
#             print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
#     logger = Logger(logdir=model_save_dir, flush_secs=2)
#     # =========>开始训练<=========
#     for epoch in range(start_epoch, config.max_epoch + 1):
#         print("epoch:", epoch)
#         since = time.time()
#         train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
#         val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)
#         print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
#               % (epoch, stage, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
#         logger.log_value('train_loss', train_loss, step=epoch)
#         logger.log_value('train_f1', train_f1, step=epoch)
#         logger.log_value('val_loss', val_loss, step=epoch)
#         logger.log_value('val_f1', val_f1, step=epoch)
#         state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
#                  'stage': stage}
#         save_ckpt(state, best_f1 < val_f1, model_save_dir)
#         best_f1 = max(best_f1, val_f1)
#         if epoch in config.stage_epoch:
#             stage += 1
#             lr /= config.lr_decay
#             best_w = os.path.join(model_save_dir, config.best_w)
#             model.load_state_dict(torch.load(best_w)['state_dict'])
#             print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
#             utils.adjust_learning_rate(optimizer, lr)
#
#
# # 用于测试加载模型
# def val(args):
#     list_threhold = [0.5]
#     model = getattr(models, config.model_name)()
#     if args.ckpt: model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
#     model = model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     val_dataset = ECGDataset(data_path=config.train_data, train=False)
#     val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)
#     for threshold in list_threhold:
#         val_loss, val_f1 = val_epoch(model, criterion, val_dataloader, threshold)
#         print('threshold %.2f val_loss:%0.3e val_f1:%.3f\n' % (threshold, val_loss, val_f1))
#
#
# # 提交结果使用
# def test(args):
#     name2idx = name2index(config.arrythmia)
#     idx2name = {idx: name for name, idx in name2idx.items()}
#     utils.mkdirs(config.sub_dir)
#     # model
#     model = getattr(models, config.model_name)()
#     model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
#     model = model.to(device)
#     model.eval()
#     sub_file = '%s/subA_%s.txt' % (config.sub_dir, time.strftime("%Y%m%d%H%M"))
#     fout = open(sub_file, 'w', encoding='utf-8')
#     with torch.no_grad():
#         for line in open(config.test_label, encoding='utf-8'):
#             fout.write(line.strip('\n'))
#             id = line.split('\t')[0]
#             file_path = os.path.join(config.test_dir, id)
#             # df = pd.read_csv(file_path, sep=' ').values
#             df = pd.read_csv(file_path, sep=' ')
#             df = add_feature(df).values
#             x = transform(df).unsqueeze(0).to(device)
#             output = torch.sigmoid(model(x)).squeeze().cpu().numpy()
#             ixs = [i for i, out in enumerate(output) if out > 0.5]
#             for i in ixs:
#                 fout.write("\t" + idx2name[i])
#             fout.write('\n')
#     fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("command", metavar="<command>", help="train or infer")
    # parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    # parser.add_argument("--ex", type=str, help="experience name")
    # parser.add_argument("--resume", action='store_true', default=False)
    # args = parser.parse_args()
    # if args.command == "train":
    #     train(args)
    # if args.command == "test":
    #     test(args)
    # if (args.command == "val"):
    #     val(args)
