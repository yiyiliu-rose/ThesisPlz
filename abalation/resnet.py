import pandas as pd
import numpy as np
from io import BytesIO
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import datetime

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-4

train_data_path = '../NLST_train.csv'
val_data_path = '../NLST_val.csv'

train_df = pd.read_csv(train_data_path)
val_df = pd.read_csv(val_data_path)

print(f"Total training rows in NLST_train.csv: {len(train_df)}")
print(f"Total validation rows in NLST_val.csv: {len(val_df)}")

train_df = train_df.dropna(subset=['images'])
val_df = val_df.dropna(subset=['images'])

print(f"Total training rows after dropping NA: {len(train_df)}")
print(f"Total validation rows after dropping NA: {len(val_df)}")

train_df['cvd'] = train_df['cvd'].astype(int)
val_df['cvd'] = val_df['cvd'].astype(int)

def load_image(image_bytes):
    return np.load(BytesIO(image_bytes), allow_pickle=True)

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_data = row['images']
        if isinstance(image_data, str):
            try:
                image_bytes = ast.literal_eval(image_data)
            except Exception as e:
                print(f"Error parsing image data at index {idx}: {e}")
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                label = 0
                if self.transform:
                    image = self.transform(image)
                return image, label
        else:
            image_bytes = image_data
        try:
            image = load_image(image_bytes)
            image = np.transpose(image, (1, 2, 0))
            image = image.astype(np.uint8)
            if self.transform:
                image = self.transform(image)
            label = row['cvd']
            return image, label
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            label = 0
            if self.transform:
                image = self.transform(image)
            return image, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = ImageDataset(train_df, transform=transform)
val_dataset = ImageDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

def train_epoch(model, loader, criterion, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=False)
    for batch in loop:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1).float()
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        
        preds = torch.sigmoid(outputs) >= 0.5
        correct += (preds == labels.byte()).sum().item()
        total += labels.size(0)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    scheduler.step()
    return correct / total, np.mean(losses)

def eval_model(model, loader, criterion, device, n_examples):
    model.eval()
    losses = []
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=False)
    with torch.no_grad():
        for batch in loop:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            preds = torch.sigmoid(outputs) >= 0.5
            correct += (preds == labels.byte()).sum().item()
            total += labels.size(0)
    
    return correct / total, np.mean(losses)

history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
best_acc = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    start = time.time()
    
    train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scheduler, len(train_dataset))
    val_acc, val_loss = eval_model(model, val_loader, criterion, device, len(val_dataset))
    
    end = time.time()
    epoch_mins, epoch_secs = divmod(end - start, 60)
    
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}')
    print(f'Epoch Time: {int(epoch_mins)}m {int(epoch_secs)}s\n')
    
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model_resnet.pth')
        print(f'Best model saved with Val Acc: {best_acc:.4f}\n')

model.load_state_dict(torch.load('best_model_resnet.pth'))
print('Best model loaded.\n')

def get_predictions(model, data_loader):
    model.eval()
    predictions = []
    prediction_probs = []
    real_values = []
    
    loop = tqdm(data_loader, leave=False)
    with torch.no_grad():
        for batch in loop:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = probs >= 0.5
            
            predictions.extend(preds.cpu().numpy())
            prediction_probs.extend(probs.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
    
    return predictions, prediction_probs, real_values

preds, probs, labels = get_predictions(model, val_loader)

accuracy = accuracy_score(labels, preds)
print(f'Accuracy on validation set: {accuracy:.4f}')

prediction_df = pd.DataFrame({
    'predictions': preds,
    'probabilities': probs,
    'labels': labels
})

# prediction_df.to_csv('nlst_predictions.csv', index=False, encoding='utf-8')
# print("Predictions saved to 'nlst_predictions.csv'")
