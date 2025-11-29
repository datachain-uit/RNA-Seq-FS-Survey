import os
import sys
sys.path.append("src")
sys.path.append(".")

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from metrics import evaluate_all_metrics
from config import (
    DATA_DIR, FS_RESULT_DIR, LOG_DIR,
    FS_METHOD
)

from deep.mlp import MLP
from deep.lstm import LSTMModel
from deep.gru import GRUModel
from deep.cnn1d import CNN1D
from deep.transformer import TransformerModel
from deep.vae import VAEModel
from deep.gan import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_val   = pd.read_csv(f"{DATA_DIR}/X_val.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
y_val   = pd.read_csv(f"{DATA_DIR}/y_val.csv").values.ravel()
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

mask_path = f"{FS_RESULT_DIR}/{FS_METHOD.lower()}_mask.npy"
print(f"Loading FS mask: {mask_path}")
mask = np.load(mask_path).astype(bool)

X_train_sel = X_train.loc[:, mask]
X_val_sel   = X_val.loc[:, mask]
X_test_sel  = X_test.loc[:, mask]

input_dim = X_train_sel.shape[1]
num_classes = len(np.unique(y_train))

def to_torch(df):
    return torch.tensor(df.values, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(
    to_torch(X_train_sel), torch.tensor(y_train, dtype=torch.long)
), batch_size=64, shuffle=True)

val_loader = DataLoader(TensorDataset(
    to_torch(X_val_sel), torch.tensor(y_val, dtype=torch.long)
), batch_size=64, shuffle=False)

test_loader = DataLoader(TensorDataset(
    to_torch(X_test_sel), torch.tensor(y_test, dtype=torch.long)
), batch_size=128, shuffle=False)

def train_and_log(model_name, model, train_loader, val_loader, epochs=50, lr=1e-3):
    os.makedirs(LOG_DIR, exist_ok=True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        batches = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()

            out = model(Xb)
            loss = criterion(out, yb)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batches += 1

        train_loss /= batches

        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                pred = model(Xv).argmax(1)
                correct += (pred == yv).sum().item()
                total += yv.size(0)

        val_acc = correct / total

        print(f"[{model_name}] Epoch {epoch:02d} | Loss={train_loss:.4f} | ValAcc={val_acc:.4f}")
        history.append({"Epoch": epoch, "TrainLoss": train_loss, "ValAcc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()

    log_path = f"{LOG_DIR}/{model_name}_training_log.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"Saved log: {log_path}")

    model.load_state_dict(best_state)
    return model, best_acc

def evaluate_model(model):
    y_true, y_pred, y_prob = [], [], []

    model.eval()
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            out = model(Xb)

            preds = out.argmax(1).cpu().numpy()
            probs = nn.Softmax(dim=1)(out).cpu().numpy()

            y_true.extend(yb.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    return np.array(y_true), np.array(y_pred), np.array(y_prob)

fs_metrics_path = f"{FS_RESULT_DIR}/fs_metrics_{FS_METHOD.lower()}.csv"
fs_row = pd.read_csv(fs_metrics_path).iloc[0]

cpu_util  = fs_row["CPUUtil(%)"]
wall_time = fs_row["WallTime(s)"]
peak_mem  = fs_row["PeakMem(MB)"]
base_mem  = fs_row["BaseMem(MB)"]
energy    = fs_row["Energy(J)"]
carbon    = fs_row["Carbon(gCO2e)"]
edp       = fs_row["EDP(J*s)"]
cache_hit = fs_row.get("CacheHit", np.nan)

deep_models = {
    "MLP": MLP(input_dim, num_classes),
    "LSTM": LSTMModel(input_dim, 64, num_classes),
    "GRU": GRUModel(input_dim, 64, num_classes),
    "CNN1D": CNN1D(input_dim, num_classes),
    "Transformer": TransformerModel(input_dim, num_classes),
    "VAE": VAEModel(input_dim, 32, num_classes),
}

results = []

for name, model in deep_models.items():
    print(f"\nTraining {name}...")
    start = time.time()

    model, _ = train_and_log(
        name, model, train_loader, val_loader,
        epochs=50, lr=1e-3
    )

    y_true, y_pred, y_prob = evaluate_model(model)

    metrics = evaluate_all_metrics(
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        cpu_util=cpu_util, wall_time=wall_time,
        peak_mem_mb=peak_mem, base_mem_mb=base_mem,
        num_evals=input_dim,
        missing_flags=None,
        labels_for_leak=y_test,
        cache_hit=cache_hit
    )

    metrics.update({
        "FS_Method": FS_METHOD,
        "Classifier": name,
        "NumFeatures": input_dim,
        "Energy(J)": energy,
        "Carbon(gCO2e)": carbon,
        "EDP(J*s)": edp,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    results.append(metrics)
    print(f"   ✔ {name}: Acc={metrics['Accuracy']:.4f} | F1={metrics['MacroF1']:.4f}")


out_path = f"{LOG_DIR}/classification_summary_deep.csv"
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"\nSaved DL results -> {out_path}")
