import os
import glob
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, feat_dir):
        self.paths = sorted(glob.glob(os.path.join(feat_dir, "*.pt")))
        self.labels = []

        for path in self.paths:
            name = os.path.basename(path)
            label_str = name.split('_')[1].split('.')[0]
            self.labels.append(
                torch.tensor([int(c) for c in label_str], dtype=torch.float32)
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        feat = torch.load(self.paths[idx], map_location="cpu").squeeze(0)
        return feat.float(), self.labels[idx]