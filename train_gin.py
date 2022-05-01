import argparse
import torch
import random
import numpy as np
from torch import optim
import torch.nn as nn
import os
import time
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
import torch_geometric
from model.contrastive_gin import GINSimclr
from data_provider.pretrain_datamodule import GINPretrainDataModule


def main(args):
    pl.seed_everything(args.seed)

    # data
    dm = GINPretrainDataModule.from_argparse_args(args)

    # model
    model = GINSimclr(
        temperature=args.temperature,
        gin_hidden_dim=args.gin_hidden_dim,
        gin_num_layers=args.gin_num_layers,
        gin_num_features=args.gin_num_features,
        bert_hidden_dim=args.bert_hidden_dim,
        bert_pretrain=args.bert_pretrain,
        graph_self=args.graph_self,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print('total params:', sum(p.numel() for p in model.parameters()))

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="gin/checkpoint/", every_n_epochs=5))
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, strategy="ddp")

    trainer.fit(model, datamodule=dm)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--default_root_dir', type=str, default='./checkpoints/', help='location of model checkpoints')
    # parser.add_argument('--max_epochs', type=int, default=500)

    # GPU
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    # parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser = Trainer.add_argparse_args(parser)
    parser = GINSimclr.add_model_specific_args(parser)  # add model args
    parser = GINPretrainDataModule.add_argparse_args(parser)  # add data args
    args = parser.parse_args()

    # gpu setting
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # if args.use_gpu and args.use_multi_gpu:
    #     args.devices = args.devices.replace(' ', '')
    #     device_ids = args.devices.split(',')
    #     args.device_ids = [int(id_) for id_ in device_ids]
    #     args.gpu = args.device_ids[0]
    #
    # if args.use_gpu:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else args.devices
    #     device = torch.device('cuda:{}'.format(args.gpu))
    #     print('Use GPU: cuda:{}'.format(args.gpu))
    # else:
    #     device = torch.device('cpu')
    #     print('Use CPU')

    if args.use_multi_gpu and args.use_gpu:  # 使用多GPU
        args.multi_gpu_flag = True
    else:
        args.multi_gpu_flag = False

    print('Args in experiment:')
    print(args)

    main(args)








