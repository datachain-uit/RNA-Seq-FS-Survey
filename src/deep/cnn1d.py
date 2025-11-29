import torch.nn as nn
import torch

class CNN1D(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * in_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)             
        x = torch.relu(self.conv1(x))  
        x = torch.relu(self.conv2(x)) 
        x = x.flatten(1)
        return self.fc(x)
