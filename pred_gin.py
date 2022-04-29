import argparse
import torch
import random
import numpy as np
from torch import optim
import torch.nn as nn
import os
import time
from model.contrastive_gin import simclr
from torch_geometric.loader import DataLoader
import torch_geometric
from data_provider.pretrain_dataset import GraphTextDataset
import warnings
import matplotlib.pyplot as plt
from utils.lr import PolynomialDecayLR
import math

fix_seed = 19990605
random.seed(fix_seed)
torch.manual_seed(fix_seed)

parser = argparse.ArgumentParser(description='Graph and Text Unsupervised Pretraining')

# train mode
parser.add_argument('--graph_self', action='store_true', help='use graph self-supervise or not', default=False)

# data preprocess and data loader
parser.add_argument('--data_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--graph_aug1', type=str, default='dnodes',
                    help='augment type, options:[dnodes, pedges, subgraph, mask_nodes, random2, random3, random4]')
parser.add_argument('--graph_aug2', type=str, default='pedges',
                    help='augment type, options:[dnodes, pedges, subgraph, mask_nodes, random2, random3, random4]')
parser.add_argument('--text_max_len', type=int, default=128, help='input text max length')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# Graph model options (Gin)
parser.add_argument('--gin_num_features', type=int, default=9, help='graph input feature dim')
parser.add_argument('--gin_hidden_dim', type=int, default=32)
parser.add_argument('--gin_num_layers', type=int, default=5)

# Text model options
parser.add_argument('--bert_pretrain', type=bool, default=True, help='use pretrained bert')
parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')


# optimization
parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

# gpu setting
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

if args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else args.devices
    device = torch.device('cuda:{}'.format(args.gpu))
    print('Use GPU: cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')
    print('Use CPU')

if args.use_multi_gpu and args.use_gpu:  # 使用多GPU
    multi_gpu_flag = True
else:
    multi_gpu_flag = False

print('Args in experiment:')
print(args)

setting = '{}_{}_selfg{}_aug1{}_aug2{}_maxlen{}'.format(
            args.graph_model,
            args.text_model,
            args.graph_self,
            args.graph_aug1,
            args.graph_aug2,
            args.text_max_len,
)
path = os.path.join(args.checkpoints, setting)

if not os.path.exists(path):
    os.makedirs(path)

time_now = time.time()

# dataset
data_set = GraphTextDataset(
    root=args.data_path,
    text_max_len=args.text_max_len,
    graph_aug1=args.graph_aug1,
    graph_aug2=args.graph_aug2,
)

if multi_gpu_flag:                                                   # 如果使用多GPU
    # dataloader
    train_loader = torch_geometric.loader.DataListLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1
    )
    # model
    model = simclr(args).to(device)
    model = torch_geometric.nn.DataParallel(model, device_ids=args.device_ids)
else:                                                                  # 如果不使用多GPU
    # dataloader
    train_loader = torch_geometric.loader.DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1
    )
    # model
    model = simclr(args).to(device)

# optimizer
model_optim = optim.Adam(model.parameters(), lr=args.lr)
for epoch in range(args.train_epochs):
    iter_count = 0
    train_steps = len(train_loader)
    # print(train_steps)  64/batch_size
    train_loss = []
    model.train()
    epoch_time = time.time()
    for i, (aug1, aug2, text1, mask1, text2, mask2) in enumerate(train_loader):
        iter_count += 1
        model_optim.zero_grad()
        if not multi_gpu_flag:  # 使用多GPU时，无需手动将batch移动到GPU上
            aug1 = aug1.to(device)
            aug2 = aug2.to(device)
        text1, mask1, text2, mask2 = text1.to(device), mask1.to(device), text2.to(device), mask2.to(device)

        graph1_rep = model.encode_graph(aug1)
        graph2_rep = model.encode_graph(aug2)
        text1_rep = model.encode_text(text1, mask1)
        text2_rep = model.encode_text(text2, mask2)

        _, _, loss11 = model(graph1_rep, text1_rep)
        _, _, loss12 = model(graph1_rep, text2_rep)
        _, _, loss21 = model(graph2_rep, text1_rep)
        _, _, loss22 = model(graph2_rep, text2_rep)

        if args.graph_self:
            _, _, loss_graph_self = model.graph_self(graph1_rep, graph2_rep)
            loss = (loss11 + loss12 + loss21 + loss22 + loss_graph_self) / 5.0
        else:
            loss = (loss11 + loss12 + loss21 + loss22) / 4.0

        train_loss.append(loss.item())
        loss.backward()
        model_optim.step()

        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    train_loss = np.average(train_loss)
    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), path + "epoch-" + str(epoch + 1) + ".pt")


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from model import Graphormer
from data import GraphDataModule, get_dataset

from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Graphormer.add_model_specific_args(parser)
    parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.max_steps = args.tot_updates + 1
    if not args.test and not args.validate:
        print(args)
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    dm = GraphDataModule.from_argparse_args(args)

    # ------------
    # model
    # ------------
    if args.checkpoint_path != '':
        model = Graphormer.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
        )
    else:
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
        )
    if not args.test and not args.validate:
        print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    # ------------
    # training
    # ------------
    metric = 'valid_' + get_dataset(dm.dataset_name)['metric']
    dirpath = args.default_root_dir + f'/lightning_logs/checkpoints'
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        dirpath=dirpath,
        filename=dm.dataset_name + '-{epoch:03d}-{' + metric + ':.4f}',
        save_top_k=100,
        mode=get_dataset(dm.dataset_name)['metric_mode'],
        save_last=True,
    )
    if not args.test and not args.validate and os.path.exists(dirpath + '/last.ckpt'):
        args.resume_from_checkpoint = dirpath + '/last.ckpt'
        print('args.resume_from_checkpoint', args.resume_from_checkpoint)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

    if args.test:
        result = trainer.test(model, datamodule=dm)
        pprint(result)
    elif args.validate:
        result = trainer.validate(model, datamodule=dm)
        pprint(result)
    else:
        trainer.fit(model, datamodule=dm)








