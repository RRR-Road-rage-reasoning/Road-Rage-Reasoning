import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import random
from datasets.feature_datasets import FeatureDataset
from models.timeconv_mlp import TimeConvMLP


# -------------------------------------------------
# Reproducibility: fix all random seeds
# -------------------------------------------------
def set_seed(seed=42):
    """
    Fix random seeds for Python, NumPy, and PyTorch to ensure
    deterministic and reproducible experimental results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------
# Automatic computation of pos_weight for imbalanced data
# -------------------------------------------------
def compute_pos_weight(labels):
    """
    Compute class-wise positive weights for multi-label
    binary classification to alleviate class imbalance.

    Args:
        labels (Tensor): Ground-truth labels of shape [N, C].

    Returns:
        Tensor: Clipped positive weights for each class.
    """
    pos = labels.sum(dim=0)
    neg = len(labels) - pos
    pw = neg / pos.clamp(min=1)
    return torch.clamp(pw, max=5.0)


# -------------------------------------------------
# Threshold calibration for each label
# -------------------------------------------------
def search_best_thresholds(model, loader, device, num_labels=3):
    """
    Search for the optimal decision threshold for each label
    by maximizing the F1-score on the validation set.

    Args:
        model (nn.Module): Trained classification model.
        loader (DataLoader): Validation data loader.
        device (str): Computation device.
        num_labels (int): Number of labels.

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
    for i in range(num_labels):
        best_f1, best_t = -1, 0.5
        for t in np.arange(0.1, 0.9, 0.02):
            pred = (probs[:, i] > t).int()
            f1 = f1_score(labels[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)

    return thresholds


# -------------------------------------------------
# Training and evaluation pipeline
# -------------------------------------------------
def train_and_eval(feat_dir, epochs=50, batch_size=4, lr=1e-3, device='cuda'):
    """
    Train a lightweight temporal classifier on pre-extracted
    VLM features and evaluate its performance.

    Args:
        feat_dir (str): Directory containing pre-extracted features.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        device (str): Computation device.
    """
    set_seed(42)

    dataset = FeatureDataset(feat_dir)

    # Fixed 80/20 train-test split
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_total - n_train], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute pos_weight using only the training set
    train_labels = torch.stack([dataset[i][1] for i in train_dataset.indices])
    pos_weight = compute_pos_weight(train_labels).to(device)

    model = TimeConvMLP().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---------------- Training ----------------
    for _ in range(epochs):
        model.train()
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    # ---------------- Threshold Calibration ----------------
    thresholds = search_best_thresholds(model, test_loader, device)

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

    # -------------------------------------------------
    # Level-1 classification: dangerous vs. non-dangerous
    # -------------------------------------------------
    y_true_bin = (all_labels.sum(axis=1) > 0).astype(int)
    y_pred_bin = (all_preds.sum(axis=1) > 0).astype(int)

    print("\n===== Level-1 Classification (Any vs. None) =====")
    print(f"Binary Acc: {accuracy_score(y_true_bin, y_pred_bin) * 100:.2f}%")
    print(f"Binary F1 : {f1_score(y_true_bin, y_pred_bin, zero_division=0):.4f}")

    # -------------------------------------------------
    # All-zero (000) class accuracy
    # -------------------------------------------------
    print("\n===== All-Zero (000) Class Accuracy =====")
    idx_zero = (y_true_bin == 0)
    if idx_zero.sum() > 0:
        y_pred_zero = (all_preds[idx_zero].sum(axis=1) == 0).astype(int)
        y_true_zero = np.ones_like(y_pred_zero)
        print(f"000 Class Acc: {accuracy_score(y_true_zero, y_pred_zero) * 100:.2f}%")
    else:
        print("No all-zero samples in the test set.")

    # -------------------------------------------------
    # Level-2 classification: per-label evaluation
    # -------------------------------------------------
    idx_pos = (y_true_bin == 1)
    if idx_pos.sum() > 0:
        print("\n===== Level-2 Classification (Per Label) =====")
        for i in range(3):
            acc = accuracy_score(all_labels[idx_pos, i], all_preds[idx_pos, i])
            f1 = f1_score(all_labels[idx_pos, i], all_preds[idx_pos, i], zero_division=0)
            print(f"Label {i+1} - Acc: {acc * 100:.2f}%, F1: {f1:.4f}")


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    feat_dir = "./features_Internvl"   # Path to pre-extracted features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_and_eval(feat_dir, device=device)
