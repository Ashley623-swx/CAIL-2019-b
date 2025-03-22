import json
import torch
import torch.nn as nn
from transformers import AutoModel
from config import Config
import torch.nn.functional as F



class MyModel(nn.Module):
    def __init__(self, Config):
        super(MyModel, self).__init__()
        self.plm = AutoModel.from_pretrained(Config.plm_path)
        self.dropout = nn.Dropout(Config.dropout)
        self.fc = nn.Linear(self.plm.config.hidden_size, Config.embedding_dim)
        for param in self.plm.parameters():
            param.requires_grad = True

    def forward(self, inputs_A, attention_mask_A, inputs_B, attention_mask_B, inputs_C, attention_mask_C):
        # 获取 [CLS] 
        emb_A = self.plm(input_ids=inputs_A, attention_mask=attention_mask_A).last_hidden_state[:, 0, :]
        emb_B = self.plm(input_ids=inputs_B, attention_mask=attention_mask_B).last_hidden_state[:, 0, :]
        emb_C = self.plm(input_ids=inputs_C, attention_mask=attention_mask_C).last_hidden_state[:, 0, :]

        # dropout
        emb_A = self.dropout(emb_A)
        emb_B = self.dropout(emb_B)
        emb_C = self.dropout(emb_C)

        # 全连接层
        emb_A = self.fc(emb_A)
        emb_B = self.fc(emb_B)
        emb_C = self.fc(emb_C)

        # 余弦相似度
        sim_AB = torch.cosine_similarity(emb_A, emb_B, dim=1)
        sim_AC = torch.cosine_similarity(emb_A, emb_C, dim=1)
        return sim_AB, sim_AC
