import torch
import numpy as np

def evaluate_dl_model(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            pred = out.argmax(1).cpu().numpy()
            prob = torch.softmax(out, dim=1).cpu().numpy()

            y_true.extend(yb.numpy())
            y_pred.extend(pred)
            y_prob.extend(prob)

    return np.array(y_true), np.array(y_pred), np.array(y_prob)
