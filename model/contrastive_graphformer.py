import torch
import torch.nn as nn
from model.graphormer.graphormer_graph_encoder import GraphormerGraphEncoder
from model.bert import TextEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.lr import PolynomialDecayLR
from torch import optim


class GraphormerSimclr(pl.LightningModule):
    def __init__(
            self,
            temperature,
            bert_hidden_dim,
            bert_pretrain,
            graph_self,
            warmup_updates,
            tot_updates,
            peak_lr,
            end_lr,
            weight_decay,
            graphormer_pretrain,
            num_atoms: int,
            num_in_degree: int,
            num_out_degree: int,
            num_edges: int,
            num_spatial: int,
            num_edge_dis: int,
            edge_type: str,
            multi_hop_max_dist: int,
            num_encoder_layers: int = 12,
            graph_embed_dim: int = 768,
            graph_ffn_embed_dim: int = 768,
            graph_attention_heads: int = 32,
            dropout: float = 0.0,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            encoder_normalize_before: bool = True,
            pre_layernorm: bool = False,
            apply_graphormer_init: bool = True,
            activation_fn: str = "gelu",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.temperature = temperature
        self.bert_hidden_dim = bert_hidden_dim
        self.bert_pretrain = bert_pretrain
        self.graph_self = graph_self

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay

        self.graphormer_pretrain = graphormer_pretrain
        self.num_atoms = num_atoms
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.num_edges = num_edges
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.num_encoder_layers = num_encoder_layers
        self.graph_embed_dim = graph_embed_dim
        self.graph_ffn_embed_dim = graph_ffn_embed_dim
        self.graph_attention_heads = graph_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_normalize_before = encoder_normalize_before
        self.pre_layernorm = pre_layernorm
        self.apply_graphormer_init = apply_graphormer_init
        self.activation_fn = activation_fn

        self.graph_encoder = GraphormerGraphEncoder(
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            num_encoder_layers=num_encoder_layers,
            embedding_dim=graph_embed_dim,
            ffn_embedding_dim=graph_ffn_embed_dim,
            num_attention_heads=graph_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            encoder_normalize_before=encoder_normalize_before,
            pre_layernorm=pre_layernorm,
            apply_graphormer_init=apply_graphormer_init,
            activation_fn=activation_fn,
        )
        if self.graphormer_pretrain:
            载入预训练

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
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        aug1, aug2, text1, mask1, text2, mask2 = batch

        _, graph1_rep = self.graph_encoder(aug1)
        graph1_rep = self.graph_proj_head(graph1_rep)

        _, graph2_rep = self.graph_encoder(aug2)
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
        # Graphormer
        parser.add_argument('--graphormer_pretrain', type=bool, default=True, help='use pretrained graphormer')
        parser.add_argument('--num_atoms', type=int, default=7, help='atom nums')
        parser.add_argument('--num_in_degree', type=int, default=7, help='in degree nums')
        parser.add_argument('--num_out_degree', type=int, default=7, help='out degree nums')
        parser.add_argument('--num_edges', type=int, default=512, help='edge nums')
        parser.add_argument('--num_spatial', type=int, default=8, help='spatial nums')
        parser.add_argument('--num_edge_dis', type=int, default=2, help='edge distance nums')
        parser.add_argument('--edge_type', type=str, default='multi_hop', help='edge type')
        parser.add_argument('--multi_hop_max_dist', type=int, default=11, help='max dist')
        parser.add_argument('--num_encoder_layers', type=int, default=12, help='encoder layer num')
        parser.add_argument('--graph_embed_dim', type=int, default=768, help='d_model')
        parser.add_argument('--graph_ffn_embed_dim', type=int, default=768, help='d_ff')
        parser.add_argument('--graph_attention_heads', type=int, default=32, help='head nums')
        parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate in FFN')
        parser.add_argument('--attention_dropout', type=float, default=0.1, help='dropout rate in MSA')
        parser.add_argument('--activation_dropout', type=float, default=0.1, help='dropout rate')
        parser.add_argument('--encoder_normalize_before', type=bool, default=True, help='')
        parser.add_argument('--pre_layernorm', type=bool, default=False, help='post-LN or pre-LN')
        parser.add_argument('--apply_graphormer_init', type=bool, default=True, help='use param init')
        parser.add_argument('--activation_fn', type=str, default='gelu', help='activation function')
        # parser.add_argument('--embed_scale', type=float, default=None, help='embedding scale size')
        # parser.add_argument('--freeze_embeddings', type=bool, default=False, help='freeze embeddings')
        # parser.add_argument('--n_trans_layers_to_freeze', type=int, default=0, help='freeze layers num')
        # parser.add_argument('--export', type=bool, default=False, help='export')
        # parser.add_argument('--traceable', type=bool, default=False,
        #                     help='True: all the inner states; False: Only last state')
        # parser.add_argument('--q_noise', type=float, default=0.0, help='noise size')
        # parser.add_argument('--qn_block_size', type=int, default=8, help='qn block size')

        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_pretrain', type=bool, default=True, help='use pretrained bert')
        # optimization
        parser.add_argument('--warmup_updates', type=int, default=60000)
        parser.add_argument('--tot_updates', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
        return parent_parser

