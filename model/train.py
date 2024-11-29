import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device, scheduler):
    model.train()
    losses = []
    correct = 0
    total = 0

    loop = tqdm(loader, leave=False)
    for batch in loop:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        preds = torch.sigmoid(outputs) >= 0.5
        correct += (preds == labels.byte()).sum().item()
        total += labels.size(0)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loop.set_description(f'Train Loss: {loss.item():.4f}')

    scheduler.step()
    return correct / total, np.mean(losses)

def eval_model(model, loader, criterion, device):
    model.eval()
    losses = []
    correct = 0
    total = 0

    loop = tqdm(loader, leave=False)
    with torch.no_grad():
        for batch in loop:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            preds = torch.sigmoid(outputs) >= 0.5
            correct += (preds == labels.byte()).sum().item()
            total += labels.size(0)

            loop.set_description(f'Val Loss: {loss.item():.4f}')

    return correct / total, np.mean(losses)

def get_predictions(model, data_loader, device):
    model.eval()
    predictions = []
    prediction_probs = []
    real_values = []

    loop = tqdm(data_loader, leave=False)
    with torch.no_grad():
        for batch in loop:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            outputs = model(images, input_ids, attention_mask)
            probs = torch.sigmoid(outputs)
            preds = probs >= 0.5

            predictions.extend(preds.cpu().numpy())
            prediction_probs.extend(probs.cpu().numpy())
            real_values.extend(labels.cpu().numpy())

    return predictions, prediction_probs, real_values
