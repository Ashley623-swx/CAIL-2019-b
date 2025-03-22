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
        # 新增全连接层，将预训练模型输出的 hidden_size 映射到指定的 embedding_dim
        self.fc = nn.Linear(self.plm.config.hidden_size, Config.embedding_dim)
        # 固定预训练模型参数（这里可以选择是否微调预训练模型）
        for param in self.plm.parameters():
            param.requires_grad = True

    def forward(self, inputs_A, attention_mask_A, inputs_B, attention_mask_B, inputs_C, attention_mask_C):
        # 从预训练模型中获取 [CLS] 标记对应的向量
        emb_A = self.plm(input_ids=inputs_A, attention_mask=attention_mask_A).last_hidden_state[:, 0, :]
        emb_B = self.plm(input_ids=inputs_B, attention_mask=attention_mask_B).last_hidden_state[:, 0, :]
        emb_C = self.plm(input_ids=inputs_C, attention_mask=attention_mask_C).last_hidden_state[:, 0, :]

        # dropout 处理
        emb_A = self.dropout(emb_A)
        emb_B = self.dropout(emb_B)
        emb_C = self.dropout(emb_C)

        # 经过全连接层映射到新的维度
        emb_A = self.fc(emb_A)
        emb_B = self.fc(emb_B)
        emb_C = self.fc(emb_C)
        # 如果需要，可以在 fc 后添加激活函数，例如 ReLU
        emb_A = F.relu(emb_A)
        emb_B = F.relu(emb_B)
        emb_C = F.relu(emb_C)

        # 计算余弦相似度
        sim_AB = torch.cosine_similarity(emb_A, emb_B, dim=1)
        sim_AC = torch.cosine_similarity(emb_A, emb_C, dim=1)
        return sim_AB, sim_AC