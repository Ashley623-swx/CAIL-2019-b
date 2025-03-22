import json
from random import shuffle
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def read_json(path):
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:  # 逐行读取
            if not line.strip():
                continue
            data.append(json.loads(line))
    shuffle(data)
    return data


class MyDataset(Dataset):
    def __init__(self, data, Config):
        tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_path)
        self.inputs_A = tokenizer([x['A'] for x in data],
                                  truncation=True, max_length=Config.max_length,
                                  padding='max_length', return_tensors='pt')
        self.inputs_B = tokenizer([x['B'] for x in data],
                                  truncation=True, max_length=Config.max_length,
                                  padding='max_length', return_tensors='pt')
        self.inputs_C = tokenizer([x['C'] for x in data],
                                  truncation=True, max_length=Config.max_length,
                                  padding='max_length', return_tensors='pt')
        self.labels = torch.tensor([0 if x['label'] == 'B' else 1 for x in data], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs_A = self.inputs_A['input_ids'][idx]
        attention_mask_A = self.inputs_A['attention_mask'][idx]
        inputs_B = self.inputs_B['input_ids'][idx]
        attention_mask_B = self.inputs_B['attention_mask'][idx]
        inputs_C = self.inputs_C['input_ids'][idx]
        attention_mask_C = self.inputs_C['attention_mask'][idx]
        label = self.labels[idx]
        return {'inputs_A': inputs_A, 'attention_mask_A': attention_mask_A,
                'inputs_B': inputs_B, 'attention_mask_B': attention_mask_B,
                'inputs_C': inputs_C, 'attention_mask_C': attention_mask_C,
                'label': label}


def get_train_eval_DataLoader(Config):
    train = read_json(Config.train_path)
    eval = read_json(Config.eval_path)
    train_dl = DataLoader(MyDataset(train, Config), batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    print('训练数据加载完成')
    eval_dl = DataLoader(MyDataset(eval, Config), batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    print('开发验证数据加载完成')
    return train_dl, eval_dl




if __name__ == '__main__':
    # 测试数据集加载部分是否好使
    from ..config import Config
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_path)
    train_dl, dev_dl = get_train_eval_DataLoader(Config)
    for batch in train_dl:
        print(batch['inputs_A'].shape, batch['inputs_B'].shape, batch['inputs_C'].shape, batch['labels'])
        break
