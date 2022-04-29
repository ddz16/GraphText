import argparse
import torch
import random
import numpy as np
from torch import optim
import torch.nn as nn
import os
import time
from model.contrastive_graphformer import GraphormerSimclr
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from torch_geometric.loader import DataLoader
import torch_geometric
from torch.utils.data import DataLoader
from data_provider.pretrain_datamodule import GraphormerPretrainDataModule
from data_provider.collator import collator_text
from functools import partial
import warnings
import matplotlib.pyplot as plt

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
parser.add_argument('--max_node', type=int, default=128, help='max node nums')
parser.add_argument('--multi_hop_max_dist', type=int, default=11, help='max dist')
parser.add_argument('--spatial_pos_max', type=int, default=11, help='max spatial position nums')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# Graph model options (Graphformer)
parser.add_argument('--graphformer_pretrain', type=bool, default=True, help='use pretrained graphformer')
parser.add_argument('--num_atoms', type=int, default=7, help='atom nums')
parser.add_argument('--num_in_degree', type=int, default=7, help='in degree nums')
parser.add_argument('--num_out_degree', type=int, default=7, help='out degree nums')
parser.add_argument('--num_edges', type=int, default=512, help='edge nums')
parser.add_argument('--num_spatial', type=int, default=8, help='spatial nums')
parser.add_argument('--num_edge_dis', type=int, default=2, help='edge distance nums')
parser.add_argument('--edge_type', type=str, default='multi_hop', help='edge type')
parser.add_argument('--multi_hop_max_dist', type=int, default=11, help='max dist')
parser.add_argument('--num_encoder_layers', type=int, default=25, help='encoder layer num')
parser.add_argument('--embedding_dim', type=int, default=768, help='d_model')
parser.add_argument('--ffn_embedding_dim', type=int, default=768, help='d_ff')
parser.add_argument('--num_attention_heads', type=int, default=32, help='head nums')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate in FFN')
parser.add_argument('--attention_dropout', type=float, default=0.1, help='dropout rate in MSA')
parser.add_argument('--layerdrop', type=float, default=0.1, help='dropout rate')
parser.add_argument('--encoder_normalize_before', type=bool, default=False, help='')
parser.add_argument('--pre_layernorm', type=bool, default=False, help='post-LN or pre-LN')
parser.add_argument('--apply_graphormer_init', type=bool, default=False, help='use param init')
parser.add_argument('--activation_fn', type=str, default='gelu', help='activation function')
parser.add_argument('--embed_scale', type=float, default=None, help='embedding scale size')
parser.add_argument('--freeze_embeddings', type=bool, default=False, help='freeze embeddings')
parser.add_argument('--n_trans_layers_to_freeze', type=int, default=0, help='freeze layers num')
parser.add_argument('--export', type=bool, default=False, help='export')
parser.add_argument('--traceable', type=bool, default=False, help='True: all the inner states; False: Only last state')
parser.add_argument('--q_noise', type=float, default=0.0, help='noise size')
parser.add_argument('--qn_block_size', type=int, default=8, help='qn block size')

# Text model options
parser.add_argument('--bert_pretrain', type=bool, default=True, help='use pretrained bert')
parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')


# optimization
parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')

# GPU
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
# parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

parser = Trainer.add_argparse_args(parser)
parser = GraphormerSimclr.add_model_specific_args(parser)  # add model args
parser = GraphormerPretrainModule.add_argparse_args(parser)  # add data args
args = parser.parse_args()
pl.seed_everything(args.seed)

# gpu setting
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_multi_gpu and args.use_gpu:  # 使用多GPU
    multi_gpu_flag = True
else:
    multi_gpu_flag = False

print('Args in experiment:')
print(args)


print('Args in experiment:')
print(args)

# data
dm = GraphormerPretrainDataModule.from_argparse_args(args)

# model
model = GraphormerSimclr(
    temperature=args.temperature,
    bert_hidden_dim=args.bert_hidden_dim,
    bert_pretrain=args.bert_pretrain,
    graph_self=args.graph_self,
    warmup_updates=args.warmup_updates,
    tot_updates=args.tot_updates,
    peak_lr=args.peak_lr,
    end_lr=args.end_lr,
    weight_decay=args.weight_decay,
    graphormer_pretrain=args.graphormer_pretrain,
    num_atoms=args.num_atoms,
    num_in_degree=args.num_in_degree,
    num_out_degree=args.num_out_degree,
    num_edges=args.num_edges,
    num_spatial=args.num_spatial,
    num_edge_dis=args.num_edge_dis,
    edge_type=args.edge_type,
    multi_hop_max_dist=args.multi_hop_max_dist,
    num_encoder_layers=args.num_encoder_layers,
    graph_embed_dim=args.graph_embed_dim,
    graph_ffn_embed_dim=args.graph_ffn_embed_dim,
    graph_attention_heads=args.graph_attention_heads,
    dropout=args.dropout,
    attention_dropout=args.attention_dropout,
    activation_dropout=args.activation_dropout,
    encoder_normalize_before=args.encoder_normalize_before,
    pre_layernorm=args.pre_layernorm,
    apply_graphormer_init=args.apply_graphormer_init,
    activation_fn=args.activation_fn,
)

print('total params:', sum(p.numel() for p in model.parameters()))


callbacks = []
callbacks.append(plc.ModelCheckpoint(dirpath="gin/checkpoint/", every_n_epochs=5))
trainer = Trainer.from_argparse_args(args, callbacks=callbacks)

trainer.fit(model, datamodule=dm)





