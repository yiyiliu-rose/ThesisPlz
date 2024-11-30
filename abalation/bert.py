import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import datetime

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

train_data_path = '../NLST_train.csv'
val_data_path = '../NLST_val.csv'

train_df = pd.read_csv(train_data_path)
val_df = pd.read_csv(val_data_path)

print(f"Total training rows before dropping NA: {len(train_df)}")
print(f"Total validation rows before dropping NA: {len(val_df)}")

train_df = train_df.dropna(subset=['text'])
val_df = val_df.dropna(subset=['text'])

print(f"Total training rows after dropping NA: {len(train_df)}")
print(f"Total validation rows after dropping NA: {len(val_df)}")

train_df['cvd'] = train_df['cvd'].astype(int)
val_df['cvd'] = val_df['cvd'].astype(int)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)

class NLSTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def create_data_loader(df, tokenizer, max_len, batch_size, shuffle=True):
    ds = NLSTDataset(
        texts=df['text'].to_numpy(),
        labels=df['cvd'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=shuffle
    )

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=True)
test_data_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=False)

class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

model = BertClassifier(n_classes=1)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

loss_fn = nn.BCEWithLogitsLoss().to(device)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def train_epoch(
    model, 
    data_loader, 
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    n_examples
):
    model.train()
    losses = []
    correct_predictions = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        outputs = outputs.squeeze()
        
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())
        
        preds = torch.sigmoid(outputs) >= 0.5
        correct_predictions += torch.sum(preds == labels.byte())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            outputs = outputs.squeeze()
            
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            
            preds = torch.sigmoid(outputs) >= 0.5
            correct_predictions += torch.sum(preds == labels.byte())
    
    return correct_predictions.double() / n_examples, np.mean(losses)

history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
best_accuracy = 0

epoch_range = tqdm(range(EPOCHS), desc="Epochs")

for epoch in epoch_range:
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    
    start_time = time.time()
    
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_df)
    )
    
    val_acc, val_loss = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(val_df)
    )
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}')
    print(f'Epoch Time: {int(epoch_mins)}m {int(epoch_secs)}s\n')
    
    history['train_acc'].append(train_acc.cpu().numpy())
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc.cpu().numpy())
    history['val_loss'].append(val_loss)
    
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), 'best_model_state.bin')
        print(f'Best model updated with Val Acc: {best_accuracy:.4f}\n')

model.load_state_dict(torch.load('best_model_state.bin'))
print('Best model loaded.\n')

def get_predictions(model, data_loader):
    model.eval()
    texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    
    with torch.no_grad():
        for batch in data_loader:
            texts.extend(batch['text'])
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            outputs = outputs.squeeze()
            probs = torch.sigmoid(outputs)
            preds = probs >= 0.5
            
            predictions.extend(preds.cpu().numpy())
            prediction_probs.extend(probs.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
    
    return texts, predictions, prediction_probs, real_values

texts, preds, probs, labels = get_predictions(model, test_data_loader)

accuracy = accuracy_score(labels, preds)
print(f'Accuracy on test set: {accuracy:.4f}')

prediction_df = pd.DataFrame({
    'predictions': preds,
    'probabilities': probs,
    'labels': labels
})

# prediction_df.to_csv('nlst_predictions.csv', index=False, encoding='utf-8')
# print("Predictions saved to 'nlst_predictions.csv'")
