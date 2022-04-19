import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from utils.GraphAug import drop_nodes, permute_edges, subgraph, mask_nodes
from copy import deepcopy
import numpy as np
import os
import random
from transformers import BertTokenizer, BertForPreTraining


class GraphTextDataset(Dataset):
    def __init__(self, root='../data/', text_max_len=128, graph_aug1='dnodes', graph_aug2='pedges'):
        super(GraphTextDataset, self).__init__(root)
        self.graph_aug1 = graph_aug1
        self.graph_aug2 = graph_aug2
        self.text_max_len = text_max_len
        self.graph_name_list = os.listdir(root + 'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root + 'text/')
        self.text_name_list.sort()

    def len(self):
        return len(self.graph_name_list)

    def get(self, idx):
        graph_name, text_name = self.graph_name_list[idx], self.text_name_list[idx]
        # load and process graph
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)
        data_aug1 = self.augment(data_graph, self.graph_aug1)
        data_aug2 = self.augment(data_graph, self.graph_aug2)

        # load and process text
        text_path = os.path.join(self.root, 'text', text_name)
        with open(text_path, 'r', encoding='utf-8') as f:
            text_list = f.readlines()
        if len(text_list) < 2:
            two_text_list = [text_list[0], text_list[0][-self.text_max_len:]]
        else:
            two_text_list = random.sample(text_list, 2)
        text1, mask1 = self.tokenizer_text(two_text_list[0])
        text2, mask2 = self.tokenizer_text(two_text_list[1])

        return data_aug1, data_aug2, text1, mask1, text2, mask2

    def augment(self, data, graph_aug):
        if graph_aug == 'dnodes':
            data_aug = drop_nodes(deepcopy(data))
        elif graph_aug == 'pedges':
            data_aug = permute_edges(deepcopy(data))
        elif graph_aug == 'subgraph':
            data_aug = subgraph(deepcopy(data))
        elif graph_aug == 'mask_nodes':
            data_aug = mask_nodes(deepcopy(data))
        elif graph_aug == 'random2':  # choose one from two augmentations
            n = np.random.randint(2)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
        elif graph_aug == 'random3':  # choose one from three augmentations
            n = np.random.randint(3)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
        elif graph_aug == 'random4':  # choose one from four augmentations
            n = np.random.randint(4)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            elif n == 3:
                data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False
        else:
            data_aug = deepcopy(data)
            data_aug.x = torch.ones((data.edge_index.max()+1, 1))

        return data_aug

    def tokenizer_text(self, text):
        tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        sentence_token = tokenizer.encode_plus(text=text,
                                               truncation=True,
                                               padding='max_length',
                                               add_special_tokens=False,
                                               max_length=self.text_max_len,
                                               return_tensors='pt',
                                               return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask


if __name__ == '__main__':
    mydataset = GraphTextDataset()
    train_loader = DataLoader(
        mydataset,
        batch_size=16,
    )
    for i, data in enumerate(train_loader):
        print(data)