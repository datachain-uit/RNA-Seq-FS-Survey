import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy, pandas as pd, os

from .gan import Generator, Discriminator

def train_gan(X_train, y_train, X_val, y_val,
              input_dim, num_classes,
              device, latent_dim=64,
              epochs=50, lr=1e-3,
              batch_size=64, log_dir="./logs"):

    os.makedirs(log_dir, exist_ok=True)

    # loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False
    )

    G = Generator(latent_dim, num_classes, input_dim).to(device)
    D = Discriminator(input_dim, num_classes).to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    adv_loss = nn.BCELoss()
    aux_loss = nn.CrossEntropyLoss()

    best_acc = 0
    best_D_wts = copy.deepcopy(D.state_dict())
    history = []

    for epoch in range(1, epochs + 1):
        D.train(); G.train()
        d_loss_epoch, g_loss_epoch = 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            bs = xb.size(0)
            valid = torch.ones(bs, 1).to(device)
            fake = torch.zeros(bs, 1).to(device)

            # sample generator
            z = torch.randn(bs, latent_dim).to(device)
            y_gen = torch.randint(0, num_classes, (bs,)).to(device)
            gen_data = G(z, y_gen)

            # --- Train D ---
            opt_D.zero_grad()
            real_valid, real_cls = D(xb)
            fake_valid, fake_cls = D(gen_data.detach())

            d_loss = (
                adv_loss(real_valid, valid)
                + aux_loss(real_cls, yb)
                + adv_loss(fake_valid, fake)
                + aux_loss(fake_cls, y_gen)
            ) / 2
            d_loss.backward()
            opt_D.step()

            # --- Train G ---
            opt_G.zero_grad()
            fake_valid2, fake_cls2 = D(gen_data)
            g_loss = adv_loss(fake_valid2, valid) + aux_loss(fake_cls2, y_gen)
            g_loss.backward()
            opt_G.step()

            d_loss_epoch += d_loss.item()
            g_loss_epoch += g_loss.item()

        # Validation acc
        D.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                _, cls_out = D(xv)
                pred = cls_out.argmax(1)
                correct += (pred == yv).sum().item()
                total += yv.size(0)
        val_acc = correct / total

        if val_acc > best_acc:
            best_acc = val_acc
            best_D_wts = copy.deepcopy(D.state_dict())

        history.append({
            "Epoch": epoch,
            "D_Loss": d_loss_epoch / len(train_loader),
            "G_Loss": g_loss_epoch / len(train_loader),
            "ValAcc": val_acc
        })

        print(f"[GAN] Epoch {epoch:03d} | Acc={val_acc:.4f}")

    pd.DataFrame(history).to_csv(f"{log_dir}/GAN_training_log.csv", index=False)
    D.load_state_dict(best_D_wts)
    return D, best_acc
