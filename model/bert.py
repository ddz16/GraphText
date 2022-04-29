import torchvision.models as models
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class TextEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(TextEncoder, self).__init__()
        if pretrained:  # if use pretrained bert model
            self.bert_model = BertModel.from_pretrained('/data2/zyj/20200705v1/GraphText/bert_pretrained/')
        else:
            config = BertConfig(vocab_size=31090, )
            self.bert_model = BertModel(config)

        self.hidden_size = self.bert_model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        output = self.bert_model(input_ids, attention_mask)['pooler_output']  # b,d
        return output


if __name__ == '__main__':
    model = TextEncoder()