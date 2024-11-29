import pandas as pd
import numpy as np
import torch
import time
from transformers import BertTokenizer
from torchvision import transforms
from torch.optim import lr_scheduler
from torch import nn
from sklearn.metrics import accuracy_score

from dataset import get_data_loaders
from model import MultimodalModel
from train import train_epoch, eval_model, get_predictions

import torch
torch.backends.cudnn.enabled = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-4
MAX_TEXT_LENGTH = 128

train_data_path = '../../processed_data/NLST_train.csv'
val_data_path = '../../processed_data/NLST_val.csv'

train_df = pd.read_csv(train_data_path)
val_df = pd.read_csv(val_data_path)

print(f"Total training rows in NLST_train.csv: {len(train_df)}")
print(f"Total validation rows in NLST_val.csv: {len(val_df)}")

train_df = train_df.dropna(subset=['images', 'text', 'cancer', 'cvd'])
val_df = val_df.dropna(subset=['images', 'text', 'cancer', 'cvd'])

print(f"Total training rows after dropping NA: {len(train_df)}")
print(f"Total validation rows after dropping NA: {len(val_df)}")

train_df['cvd'] = train_df['cvd'].astype(int)
val_df['cvd'] = val_df['cvd'].astype(int)

def convert_cancer_label(row):
    if row['cancer'] == 0:
        return 'lung cancer: negative'
    elif row['cancer'] == 1:
        return 'lung cancer: positive'
    else:
        return 'lung cancer: unknown'

# train_df['cancer_text'] = train_df.apply(convert_cancer_label, axis=1)
# val_df['cancer_text'] = val_df.apply(convert_cancer_label, axis=1)

train_df['cancer_text'] = "lung cancer: negative"
val_df['cancer_text'] = "lung cancer: negative"

train_df['full_text'] = train_df['text'] + ' ' + train_df['cancer_text']
val_df['full_text'] = val_df['text'] + ' ' + val_df['cancer_text']

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_loader, val_loader = get_data_loaders(train_df, val_df, tokenizer, image_transform, MAX_TEXT_LENGTH, BATCH_SIZE)

model = MultimodalModel()
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
best_acc = 0

log_file_path = 'training_log.txt'
with open(log_file_path, 'w') as log_file:
    log_file.write('Epoch,Train Loss,Train Acc,Val Loss,Val Acc,Epoch Time\n')

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        start = time.time()

        train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        val_acc, val_loss = eval_model(model, val_loader, criterion, device)

        end = time.time()
        epoch_mins, epoch_secs = divmod(end - start, 60)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}')
        print(f'Epoch Time: {int(epoch_mins)}m {int(epoch_secs)}s\n')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        log_file.write(f"{epoch + 1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{int(epoch_mins)}m {int(epoch_secs)}s\n")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_multimodal_model.pth')
            print(f'Best model saved with Val Acc: {best_acc:.4f}\n')
            log_file.write(f'Best model saved at epoch {epoch + 1} with Val Acc: {best_acc:.4f}\n')

print(f"Training complete. Epoch details saved to '{log_file_path}'.\n")

model.load_state_dict(torch.load('best_multimodal_model.pth'))
model = model.to(device)
print('Best model loaded.\n')

preds, probs, labels = get_predictions(model, val_loader, device)

accuracy = accuracy_score(labels, preds)
print(f'Accuracy on validation set: {accuracy:.4f}')
