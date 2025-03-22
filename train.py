# 训练函数
from tqdm import tqdm

def train_one_epoch(model, optimizer, scheduler, criterion, train_loader, device, epoch):
    model.train()
    num_examples = 0
    total_loss = 0.0
    total_correct = 0

    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, batch in bar:
        inputs_A = batch['inputs_A'].to(device)
        inputs_B = batch['inputs_B'].to(device)
        inputs_C = batch['inputs_C'].to(device)
        attention_mask_A = batch['attention_mask_A'].to(device)
        attention_mask_B = batch['attention_mask_B'].to(device)
        attention_mask_C = batch['attention_mask_C'].to(device)
        labels = batch['label'].to(device)

        # 获取模型输出的相似度
        sim_AB, sim_AC = model(inputs_A, attention_mask_A, inputs_B, attention_mask_B, inputs_C, attention_mask_C)
        # 将标签 0 转换为 -1，标签 1 保持为 1
        logits = sim_AC - sim_AB
        loss = criterion(logits, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = (logits > 0).long()  # 当 sim_AC > sim_AB 时预测为 1（选择 C），否则为 0（选择 B）
        correct = (preds == labels).sum().item()
        total_correct += correct
        num_examples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / num_examples
        accuracy = total_correct / num_examples
        bar.set_postfix(epoch=epoch, train_loss=avg_loss, train_accuracy=accuracy)

    if scheduler is not None:
        scheduler.step()
    return avg_loss, accuracy


