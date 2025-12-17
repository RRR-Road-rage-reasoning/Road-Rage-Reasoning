import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class VideoFolderDataset(Dataset):
    """
    Each sub-folder represents one video.
    Frames are uniformly sampled to a fixed length.
    """

    def __init__(self, root_dir, num_frames=20, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform

        self.video_folders = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        folder = self.video_folders[idx]

        frame_paths = sorted(
            glob.glob(os.path.join(folder, "*.png")) +
            glob.glob(os.path.join(folder, "*.jpg"))
        )

        if len(frame_paths) == 0:
            raise RuntimeError(f"No frames found in {folder}")

        # -------- uniform sampling --------
        if len(frame_paths) >= self.num_frames:
            indices = np.linspace(0, len(frame_paths) - 1, self.num_frames).astype(int)
            frame_paths = [frame_paths[i] for i in indices]
        else:
            frame_paths += [frame_paths[-1]] * (self.num_frames - len(frame_paths))

        frames = [Image.open(p).convert("RGB") for p in frame_paths]

        if self.transform:
            frames = self.transform(frames)

        video_id = os.path.basename(folder)
        return frames, video_id
