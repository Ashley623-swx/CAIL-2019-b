import json
import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertModel
from config import Config
import torch.nn.functional as F
import os
import re



class MyModel(nn.Module):
    def __init__(self, Config):
        super(MyModel, self).__init__()
        # 使用BertConfig和BertModel替代AutoModel
        config_path = os.path.join(Config.plm_path, 'bert_config.json')
        model_path = os.path.join(Config.plm_path, 'pytorch_model.bin')
        
        if os.path.exists(config_path) and os.path.exists(model_path):
            bert_config = BertConfig.from_json_file(config_path)
            self.plm = BertModel(bert_config)
            
            # 加载预训练权重并处理键名不匹配问题
            pretrained_dict = torch.load(model_path)
            model_dict = self.plm.state_dict()
            
            # 创建键名映射关系
            new_dict = {}
            for k, v in pretrained_dict.items():
                # 移除bert.前缀和cls.相关权重
                if k.startswith('bert.'):
                    new_key = k[5:]  # 去除"bert."前缀
                    if new_key in model_dict:
                        new_dict[new_key] = v
            
            # 更新模型权重
            model_dict.update(new_dict)
            self.plm.load_state_dict(model_dict)
            print(f"成功加载预训练模型权重")
        else:
            raise FileNotFoundError(f"无法找到模型文件: {config_path} 或 {model_path}")
            
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
