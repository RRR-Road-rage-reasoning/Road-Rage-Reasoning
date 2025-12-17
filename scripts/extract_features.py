import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.video_folder_dataset import VideoFolderDataset
from datasets.transforms import ToTensorAndNormalize
from models.internvl_clip_vision import internvl_clip_6b


def main():
    """
    Extract video-level visual features using a frozen InternVL vision encoder.
    All absolute paths are replaced with public-friendly placeholders.
    """

    # Root directory containing decoded video frames
    root_dir = "./data/videos"

    # Directory to store extracted feature tensors
    save_dir = "./features/internvl"
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Dataset ----------------
    dataset = VideoFolderDataset(
        root_dir=root_dir,
        num_frames=20,
        transform=ToTensorAndNormalize()
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    # ---------------- Model ----------------
    model = internvl_clip_6b(img_size=224).to(device).eval()

    # ---------------- Feature Extraction ----------------
    with torch.no_grad():
        for frames, vid_ids in tqdm(loader, desc="Extracting features"):
            # Convert frame layout to [B, C, T, H, W]
            frames = frames.permute(0, 2, 1, 3, 4).to(device)

            outputs = model(frames)

            # Handle models that return auxiliary outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Temporal pooling for video-level representation
            if outputs.dim() == 3:
                outputs = outputs.mean(dim=1)

            for i, vid in enumerate(vid_ids):
                torch.save(
                    outputs[i].cpu(),
                    os.path.join(save_dir, f"{vid}.pt")
                )

    print("Feature extraction completed successfully.")


if __name__ == "__main__":
    main()
