import os
import glob
import torch
from PIL import Image
from torchvision import transforms

from models.internvl_clip_vision import internvl_clip_6b
from models.timeconv_mlp import TimeConvMLP


def main(video_dir):
    """
    End-to-end inference pipeline:
    video frames → InternVL visual features → TimeConvMLP → binary probabilities.

    All paths are public-friendly placeholders.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Image Transform ----------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ---------------- Models ----------------
    internvl = internvl_clip_6b(img_size=224).to(device).eval()

    mlp = TimeConvMLP().to(device)
    mlp.load_state_dict(
        torch.load(
            "./checkpoints/timeconv_mlp.pth",
            map_location=device
        )
    )
    mlp.eval()

    # ---------------- Load Frames ----------------
    frame_paths = sorted(glob.glob(os.path.join(video_dir, "*.png")))
    frames = [
        transform(Image.open(p).convert("RGB"))
        for p in frame_paths[:20]
    ]

    frames = (
        torch.stack(frames)
        .unsqueeze(0)
        .permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        .to(device)
    )

    # ---------------- Inference ----------------
    with torch.no_grad():
        features = internvl(frames)

        if isinstance(features, tuple):
            features = features[0]

        # Temporal average pooling
        features = features.mean(dim=1)

        logits = mlp(features)
        probs = torch.sigmoid(logits)

    print("Predicted probabilities:", probs.cpu().numpy())


if __name__ == "__main__":
    # Directory containing extracted video frames (PNG format)
    video_dir = "./data/sample_video_frames"
    main(video_dir)
