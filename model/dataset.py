import pandas as pd
import numpy as np
from io import BytesIO
import ast
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer

class MultimodalDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None, max_length=128):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_data = row['images']
        if isinstance(image_data, str):
            try:
                image_bytes = ast.literal_eval(image_data)
            except Exception as e:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image_bytes = image_data
        try:
            image = np.load(BytesIO(image_bytes), allow_pickle=True)
            image = np.transpose(image, (1, 2, 0))
            image = image.astype(np.uint8)
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.transform:
                image = self.transform(image)

        full_text = row['full_text']
        encoding = self.tokenizer.encode_plus(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        label = row['cvd']

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.float)
        }

def get_data_loaders(train_df, val_df, tokenizer, image_transform, max_length, batch_size):
    train_dataset = MultimodalDataset(train_df, tokenizer, transform=image_transform, max_length=max_length)
    val_dataset = MultimodalDataset(val_df, tokenizer, transform=image_transform, max_length=max_length)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
