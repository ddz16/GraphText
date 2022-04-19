import argparse
import torch
import random
import numpy as np
from torch import optim
import torch.nn as nn
import os
import time
from model.contrastive import simclr
from torch_geometric.loader import DataLoader
import torch_geometric
from data_provider.MyDataset import GraphTextDataset
import warnings
import matplotlib.pyplot as plt

fix_seed = 19990605
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Graph and Text Unsupervised Pretraining')

# choose model
parser.add_argument('--graph_model', type=str, default='GIN',
                    help='model name, options: [GIN, Graphformer]')
parser.add_argument('--text_model', type=str, default='Bert',
                    help='model name, options: [Bert]')

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
parser.add_argument('--gin_num_features', type=int, default=32, help='graph input feature dim')
parser.add_argument('--gin_hidden_dim', type=int, default=32)
parser.add_argument('--gin_num_layers', type=int, default=5)


# Graph model options (Graphformer)
# parser.add_argument('--graphformer_pretrain', type=bool, default=True, help='use pretrained graphformer')
# parser.add_argument('--gnn_num_features', type=int, default=7, help='graph input feature dim')
# parser.add_argument('--gnn_hidden_dim', type=int, default=7)
# parser.add_argument('--gnn_num_layers', type=int, default=7)
#
# parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
# parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
# parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
# parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
# parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
# parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
# parser.add_argument('--factor', type=int, default=1, help='attn factor')
# parser.add_argument('--distil', action='store_false',
#                     help='whether to use distilling in encoder, using this argument means not using distilling',
#                     default=True)
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
# parser.add_argument('--embed', type=str, default='timeF',
#                     help='time features encoding, options:[timeF, fixed, learned]')
# parser.add_argument('--activation', type=str, default='gelu', help='activation')
# parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
# parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
# parser.add_argument('--temperature', type=float, default=1.0, help='the temperature of NT_XentLoss')
# parser.add_argument('--loss_weight', type=float, default=1, help='the weight of MI loss ')

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

if args.use_multi_gpu and args.use_gpu:  # 如果使用多GPU
    # data
    data_set = GraphTextDataset(
        root=args.data_path,
        text_max_len=args.text_max_len,
        graph_aug1=args.graph_aug1,
        graph_aug2=args.graph_aug2,
    )
    train_loader = torch_geometric.loader.DataListLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )
    # build model
    model = simclr(args).to(device)
    model = torch_geometric.nn.DataParallel(model, device_ids=args.device_ids)
else:  # 如果不使用多GPU
    # data
    data_set = GraphTextDataset(
        root=args.data_path,
        text_max_len=args.text_max_len,
        graph_aug1=args.graph_aug1,
        graph_aug2=args.graph_aug2,
    )
    train_loader = torch_geometric.loader.DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )
    # build model
    model = simclr(args).to(device)

# optimizer
model_optim = optim.Adam(model.parameters(), lr=args.lr)
for epoch in range(args.train_epochs):
    iter_count = 0
    train_steps = len(train_loader)
    train_loss = []
    model.train()
    epoch_time = time.time()
    for i, (aug1, aug2, text1, mask1, text2, mask2) in enumerate(train_loader):
        iter_count += 1
        model_optim.zero_grad()
        graph1 = aug1.to(device)
        graph2 = aug2.to(device)
        text1, mask1, text2, mask2 = text1.to(device), mask1.to(device), text2.to(device), mask2.to(device)

        _, _, loss11 = model(graph1, text1, mask1)
        _, _, loss12 = model(graph1, text2, mask2)
        _, _, loss21 = model(graph2, text1, mask1)
        _, _, loss22 = model(graph2, text2, mask2)
        if args.graph_self:
            _, _, loss_graph_self = model.graph_self(graph1, graph2)
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

torch.cuda.empty_cache()



