import torch
from tqdm import tqdm

@torch.no_grad()
def valid_one_epoch(model, criterion, valid_dl, device, epoch):
    model.eval()
    num_examples = 0
    total_correct = 0
    total_loss = 0.0

    bar = tqdm(enumerate(valid_dl), total=len(valid_dl))
    for i, batch in bar:
        inputs_A = batch['inputs_A'].to(device)
        inputs_B = batch['inputs_B'].to(device)
        inputs_C = batch['inputs_C'].to(device)
        attention_mask_A = batch['attention_mask_A'].to(device)
        attention_mask_B = batch['attention_mask_B'].to(device)
        attention_mask_C = batch['attention_mask_C'].to(device)
        labels = batch['label'].to(device)

        sim_AB, sim_AC = model(inputs_A, attention_mask_A, inputs_B, attention_mask_B, inputs_C, attention_mask_C)
        logits = sim_AC - sim_AB
        loss = criterion(logits, labels.float())

        preds = (logits > 0).long()
        correct = (preds == labels).sum().item()
        total_correct += correct
        num_examples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / num_examples
        accuracy = total_correct / num_examples
        bar.set_postfix(epoch=epoch, valid_loss=avg_loss, valid_accuracy=accuracy)

    return avg_loss, accuracy