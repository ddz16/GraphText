# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collator import collator
from wrapper import MyGraphPropPredDataset, MyPygPCQM4MDataset, MyZINCDataset

from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
import torch_geometric
from torch.utils.data import DataLoader
from functools import partial
from pretrain_dataset import GINPretrainDataset, GraphformerPretrainDataset
from data_provider.collator import collator_text


class GINPretrainDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = '../data/',
        text_max_len: int = 128,
        graph_aug1: str = 'dnodes',
        graph_aug2: str = 'pedges',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.multi_gpu_flag = args.multi_gpu_flag
        self.dataset = GINPretrainDataset(root, text_max_len, graph_aug1, graph_aug2)

    def setup(self, stage: str = None):
        self.train_dataset = self.dataset

    def train_dataloader(self):
        if not self.multi_gpu_flag:
            loader = torch_geometric.loader.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            loader = torch_geometric.loader.DataListLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        print('len(train_dataloader)', len(loader))
        return loader


class GraphormerPretrainDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = '../data/',
        text_max_len: int = 128,
        graph_aug1: str = 'dnodes',
        graph_aug2: str = 'pedges',
        multi_gpu_flag: bool = False,
        max_node: int = 100,
        multi_hop_max_dist: int = 5,
        spatial_pos_max: int = 1024,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.multi_gpu_flag = multi_gpu_flag
        self.dataset = GraphformerPretrainDataset(root, text_max_len, graph_aug1, graph_aug2)
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def setup(self, stage: str = None):
        self.train_dataset = self.dataset

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(collator_text,
                               max_node=self.max_node,
                               multi_hop_max_dist=self.multi_hop_max_dist,
                               spatial_pos_max=self.spatial_pos_max),
        )
        print('len(train_dataloader)', len(loader))
        return loader


