import torch
import torch.nn as nn
from model.gin_graph_encoder import GraphEncoder
from model.bert import TextEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim


class GINSimclr(pl.LightningModule):
    def __init__(
            self,
            temperature,
            gin_hidden_dim,
            gin_num_layers,
            gin_num_features,
            bert_hidden_dim,
            bert_pretrain,
            graph_self,
            lr,
            weight_decay,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.temperature = temperature
        self.gin_hidden_dim = gin_hidden_dim
        self.gin_num_layers = gin_num_layers
        self.gin_num_features = gin_num_features
        self.bert_hidden_dim = bert_hidden_dim
        self.bert_pretrain = bert_pretrain
        self.graph_self = graph_self

        self.lr = lr
        self.weight_decay = weight_decay

        self.embedding_dim = self.gin_hidden_dim * self.gin_num_layers

        self.graph_encoder = GraphEncoder(self.gin_num_features, self.gin_hidden_dim, self.gin_num_layers)
        self.text_encoder = TextEncoder(pretrained=self.bert_pretrain)

        self.graph_proj_head = nn.Sequential(
          nn.Linear(self.embedding_dim, self.embedding_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.text_proj_head = nn.Sequential(
          nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.bert_hidden_dim, self.embedding_dim)
        )

    def forward(self, features_graph, features_text):
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        logits_per_graph = features_graph @ features_text.t() / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        return logits_per_graph, logits_per_text, loss

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        aug1, aug2, text1, mask1, text2, mask2 = batch

        graph1_rep, _ = self.graph_encoder(aug1.x, aug1.edge_index, aug1.batch)
        graph1_rep = self.graph_proj_head(graph1_rep)

        graph2_rep, _ = self.graph_encoder(aug2.x, aug2.edge_index, aug2.batch)
        graph2_rep = self.graph_proj_head(graph2_rep)

        text1_rep = self.text_encoder(text1, mask1)
        text1_rep = self.text_proj_head(text1_rep)

        text2_rep = self.text_encoder(text2, mask2)
        text2_rep = self.text_proj_head(text2_rep)

        _, _, loss11 = self.forward(graph1_rep, text1_rep)
        _, _, loss12 = self.forward(graph1_rep, text2_rep)
        _, _, loss21 = self.forward(graph2_rep, text1_rep)
        _, _, loss22 = self.forward(graph2_rep, text2_rep)

        if self.graph_self:
            _, _, loss_graph_self = self.forward(graph1_rep, graph2_rep)
            loss = (loss11 + loss12 + loss21 + loss22 + loss_graph_self) / 5.0
        else:
            loss = (loss11 + loss12 + loss21 + loss22) / 4.0

        self.log("train_loss", loss)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        parser.add_argument('--graph_self', action='store_true', help='use graph self-supervise or not', default=False)
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=32)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--gin_num_features', type=int, default=9, help='graph input feature dim')
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_pretrain', type=bool, default=True, help='use pretrained bert')
        # optimization
        parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
        return parent_parser

