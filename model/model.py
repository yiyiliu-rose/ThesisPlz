import torch
import torch.nn as nn
import timm
from transformers import BertModel

class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.image_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.image_model.head = nn.Identity()
        
        self.text_model = BertModel.from_pretrained('bert-base-uncased')

        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.1)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_model(image)  # (batch_size, 768)
        image_features = image_features.unsqueeze(1)  # (batch_size, 1, 768)
        image_features = image_features.permute(1, 0, 2)  # (1, batch_size, 768)

        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state  # (batch_size, seq_len, 768)
        text_features = text_features.permute(1, 0, 2)  # (seq_len, batch_size, 768)

        attn_output, attn_weights = self.cross_attn(text_features, image_features, image_features)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, 768)
        pooled_output = attn_output.mean(dim=1)

        logits = self.classifier(pooled_output)
        return logits
