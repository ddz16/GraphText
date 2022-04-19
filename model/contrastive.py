import torch
import torch.nn as nn
from model.gin import GraphEncoder
from model.bert import TextEncoder
import torch.nn.functional as F
import numpy as np
from transformers import BertForPreTraining


class simclr(torch.nn.Module):
    def __init__(self, configs):
        super(simclr, self).__init__()
        
        self.temperature = configs.temperature
        if self.temperature == 0:  # learned temperature
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.embedding_dim = configs.gin_hidden_dim * configs.gin_num_layers
        if configs.graph_model == 'GIN':
            self.gnn = GraphEncoder(configs.gin_num_features, configs.gin_hidden_dim, configs.gin_num_layers)

        self.bert = TextEncoder(pretrained=configs.bert_pretrain, avg=False)

        self.proj_head_graph = nn.Sequential(
          nn.Linear(self.embedding_dim, self.embedding_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.proj_head_text = nn.Sequential(
          nn.Linear(configs.bert_hidden_dim, configs.bert_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(configs.bert_hidden_dim, self.embedding_dim)
        )

        # self.init_emb()

    def init_emb(self):
        """ Initialize weights of Linear layers """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def encode_graph(self, graph, edge_index, batch):
        y, _ = self.gnn(graph, edge_index, batch)
        y = self.proj_head_graph(y)
        return y

    def encode_text(self, input_ids, attention_mask):
        y = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        y = self.proj_head_text(y)
        return y

    def forward(self, graph_data, input_ids, attention_mask):
        graph, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch

        batch_size = graph.size(0)
        device = graph.device

        features_graph = self.encode_graph(graph, edge_index, batch)
        features_text = self.encode_text(input_ids, attention_mask)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        if self.temperature == 0:
            logit_scale = self.logit_scale.exp()
            logits_per_graph = logit_scale * features_graph @ features_text.t()
        else:
            logits_per_graph = features_graph @ features_text.t() / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        return logits_per_graph, logits_per_text, loss

    def graph_self(self, graph_data1, graph_data2):
        graph1, edge_index1, batch1 = graph_data1.x, graph_data1.edge_index, graph_data1.batch
        graph2, edge_index2, batch2 = graph_data2.x, graph_data2.edge_index, graph_data2.batch
        batch_size = graph1.size(0)
        device = graph1.device

        features_graph1 = self.encode_graph(graph1, edge_index1, batch1)
        features_graph2 = self.encode_graph(graph2, edge_index2, batch2)

        # normalized features
        features_graph1 = F.normalize(features_graph1, dim=-1)
        features_graph2 = F.normalize(features_graph2, dim=-1)

        # cosine similarity as logits
        if self.temperature == 0:
            logit_scale = self.logit_scale.exp()
            logits_per_graph1 = logit_scale * features_graph1 @ features_graph2.t()
        else:
            logits_per_graph1 = features_graph1 @ features_graph2.t() / self.temperature
        logits_per_graph2 = logits_per_graph1.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=device)  # 大小为B
        loss_graph1 = F.cross_entropy(logits_per_graph1, labels)
        loss_graph2 = F.cross_entropy(logits_per_graph2, labels)
        loss = (loss_graph1 + loss_graph2) / 2

        return logits_per_graph1, logits_per_graph2, loss
