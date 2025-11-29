import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, in_dim, num_classes, nhead=4):
        super().__init__()
        self.embed = nn.Linear(in_dim, 128)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=nhead,
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embed(x).unsqueeze(1)
        x = self.enc(x)
        return self.fc(x[:, 0])
