import json
from collections import Counter

with open(r'C:\Users\Administrator\PycharmProjects\pythonProject1\CAIL-2019-SCM\CAIL-2019-SCM-data\train.json', encoding='utf-8') as f:
    train_data = []
    for line in f:  # 逐行读取
        # 跳过空行
        if not line.strip():
            continue
        train_data.append(json.loads(line))


print(f'训练样本总数为：{len(train_data)}')
c = Counter([x['label'] for x in train_data])  # 各个类别的样本数量
for ele, freq in c.items():
    print(ele,freq)