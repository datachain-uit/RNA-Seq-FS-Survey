import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, output_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2)
        )
        self.adv = nn.Linear(128, 1)
        self.cls = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.net(x)
        return torch.sigmoid(self.adv(h)), self.cls(h)
