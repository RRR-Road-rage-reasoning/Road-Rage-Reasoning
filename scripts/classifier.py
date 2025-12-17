import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import random

from models.timeconv_mlp import TimeConvMLP
from datasets.feature_datasets import FeatureDataset


# -------------------------------------------------
# Reproducibility settings
# -------------------------------------------------
def set_seed(seed=42):
    """
    Fix random seeds across Python, NumPy, and PyTorch
    to ensure deterministic and reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------
# Compute class-wise positive weights for imbalance
# -------------------------------------------------
def compute_pos_weight(labels):
    """
    Compute positive class weights for multi-label
    binary classification to mitigate class imbalance.

    Args:
        labels (Tensor): Ground-truth labels of shape [N, C].

    Returns:
        Tensor: Clipped positive weights for each class.
    """
    pos = labels.sum(dim=0)
    neg = len(labels) - pos
    return torch.clamp(neg / pos.clamp(min=1), max=5.0)


# -------------------------------------------------
# Threshold calibration via F1-score maximization
# -------------------------------------------------
def search_best_thresholds(model, loader, device):
    """
    Search for optimal decision thresholds for each label
    by maximizing the F1-score on the validation set.

    Args:
        model (nn.Module): Trained classification model.
        loader (DataLoader): Validation data loader.
        device (str): Computation device.

    Returns:
        list: Optimal threshold for each label.
    """
    model.eval()
    probs, labels = [], []

    with torch.no_grad():
        for feats, labs in loader:
            feats = feats.to(device)
            probs.append(torch.sigmoid(model(feats)).cpu())
            labels.append(labs)

    probs = torch.cat(probs)
    labels = torch.cat(labels)

    thresholds = []
    for i in range(3):
        best_f1, best_t = -1, 0.5
        for t in np.arange(0.1, 0.9, 0.02):
            pred = (probs[:, i] > t).int()
            f1 = (2 * (pred & labels[:, i]).sum()) / \
                 ((pred.sum() + labels[:, i].sum()).clamp(min=1))
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)

    return thresholds


# -------------------------------------------------
# Training and evaluation procedure
# -------------------------------------------------
def train_and_eval(feat_dir, epochs=50, batch_size=4, lr=1e-3, device='cuda'):
    """
    Train a lightweight temporal classifier on pre-extracted
    visual features and evaluate its performance.

    Args:
        feat_dir (str): Directory containing pre-extracted features.
        epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        lr (float): Learning rate.
        device (str): Computation device.
    """
    set_seed(42)

    dataset = FeatureDataset(feat_dir)

    # Fixed 80/20 train-validation split
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_total - n_train], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute pos_weight using only the training subset
    train_labels = torch.stack([dataset[i][1] for i in train_dataset.indices])
    pos_weight = compute_pos_weight(train_labels).to(device)

    model = TimeConvMLP().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---------------- Training ----------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {total_loss / len(train_loader):.4f}")

    # ---------------- Threshold calibration ----------------
    thresholds = search_best_thresholds(model, test_loader, device)
    print("Optimal thresholds:", thresholds)

    # ---------------- Evaluation ----------------
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for feats, labels in test_loader:
            feats = feats.to(device)
            probs = torch.sigmoid(model(feats)).cpu()
            preds = torch.zeros_like(probs)
            for i, t in enumerate(thresholds):
                preds[:, i] = (probs[:, i] > t).int()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ---------------- Model checkpointing ----------------
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, "timeconv_mlp.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved to: {save_path}")


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    feat_dir = "./features"   # Path to pre-extracted feature directory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_and_eval(feat_dir, device=device)
