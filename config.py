from transformers import AutoTokenizer
import os


class Config():
    # 路径配置
    train_path = r'CAIL-2019-SCM-data/train.json'
    eval_path = r'CAIL-2019-SCM-data/valid.json'

    # 通用配置
    plm_path = os.path.abspath('ms')  # 使用绝对路径
    tokenizer_path = os.path.abspath('ms')  # 使用绝对路径
    max_length = 512
    batch_size = 7
    epoch = 50
    learning_rate = 1e-5
    weight_decay = 5e-3
    schedule = 'CosineAnnealingLR'
    device = 'cuda:0'  # 使用第一个GPU

    # RCNN 相关配置
    dropout = 0.2  
    embedding_dim = 256
