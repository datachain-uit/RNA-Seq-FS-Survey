import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import copy, os

def train_dl(model, train_loader, val_loader,
             device, model_name="DL",
             epochs=50, lr=1e-3,
             log_dir="./logs"):

    os.makedirs(log_dir, exist_ok=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_wts = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())

        history.append({"Epoch": epoch, "TrainLoss": total_loss, "ValAcc": val_acc})
        print(f"[{model_name}] Epoch {epoch:02d}: Loss={total_loss:.4f} | ValAcc={val_acc:.4f}")

    pd.DataFrame(history).to_csv(f"{log_dir}/{model_name}_training_log.csv", index=False)
    model.load_state_dict(best_wts)
    return model, best_acc
