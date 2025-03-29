import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import gc
import warnings
warnings.filterwarnings('ignore')

from config import Config
from model import MyModel
from utils.data_loader import get_train_eval_DataLoader
from utils.strategy import fetch_scheduler, set_seed
from train import train_one_epoch
from valid import valid_one_epoch


set_seed(2025)
device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
model = MyModel(Config)
optimizer = AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
criterion = nn.BCEWithLogitsLoss()
scheduler = fetch_scheduler(optimizer=optimizer, schedule=Config.schedule)
train_dl, valid_dl = get_train_eval_DataLoader(Config)

model.to(device)
best_model_state = copy.deepcopy(model.state_dict())
best_valid_loss = np.inf
best_valid_accuracy = 0.0

start_time = time.time()

for epoch in range(1, Config.epoch + 1):
    train_loss, train_accuracy = train_one_epoch(model, optimizer, scheduler, criterion, train_dl, device, epoch)
    valid_loss, valid_accuracy = valid_one_epoch(model, criterion, valid_dl, device, epoch)
    if valid_loss <= best_valid_loss:
        print(f'best valid loss has improved ({best_valid_loss}---->{valid_loss})')
        best_valid_loss = valid_loss
        best_valid_accuracy = valid_accuracy
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, './results/saved_checkpoint.pth')
        print('A new best model state has saved')

end_time = time.time()
print('Training Finish !!!!!!!!')
print(f'best valid loss == {best_valid_loss}, best valid accuracy == {best_valid_accuracy}')
time_cost = end_time - start_time
print(f'training cost time == {time_cost}s')









