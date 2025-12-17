import torch
from torchvision import transforms


class ToTensorAndNormalize:
    """
    Resize -> ToTensor -> ImageNet normalization
    """

    def __init__(self, img_size=224):
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, frames):
        # frames: List[PIL.Image]
        return torch.stack([self.tf(f) for f in frames])  # [T, C, H, W]
