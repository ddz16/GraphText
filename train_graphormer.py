import argparse
import torch
import random
import numpy as np
from torch import optim
import torch.nn as nn
import os
import time
from model.contrastive_graphformer import simclr
from torch_geometric.loader import DataLoader
import torch_geometric
from torch.utils.data import DataLoader
from data_provider.pretrain_dataset import GraphformerTextDataset
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
data_set = GraphformerTextDataset(
    root=args.data_path,
    text_max_len=args.text_max_len,
    graph_aug1=args.graph_aug1,
    graph_aug2=args.graph_aug2,
)
# dataloader
train_loader = DataLoader(
    data_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    collate_fn=partial(collator_text,
                       max_node=args.max_node,
                       multi_hop_max_dist=args.multi_hop_max_dist,
                       spatial_pos_max=args.spatial_pos_max),
)

# model
model = simclr(args).to(device)
if multi_gpu_flag:
    model = torch.nn.DataParallel(model, device_ids=args.device_ids)


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
        graph1 = aug1.to(device)
        graph2 = aug2.to(device)
        text1, mask1, text2, mask2 = text1.to(device), mask1.to(device), text2.to(device), mask2.to(device)

        graph1_rep = model.encode_graph(graph1)
        graph2_rep = model.encode_graph(graph2)
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





