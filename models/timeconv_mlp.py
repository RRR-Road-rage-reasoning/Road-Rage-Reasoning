import torch
import torch.nn as nn


class TimeConvMLP(nn.Module):
    """
    TimeConvMLP

    A lightweight temporal classification head for video-level prediction.

    Input:
        x: Tensor of shape [B, T, C]
           where T = number of tokens (e.g., 5121),
                 C = feature dimension (e.g., 3200)

    Architecture:
        1. 1D temporal convolution over token dimension
        2. BatchNorm + ReLU
        3. Adaptive average pooling to fixed temporal length
        4. MLP classifier

    Output:
        logits: Tensor of shape [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int = 3200,
        hidden_dim: int = 128,
        output_dim: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        # Temporal convolution block
        self.time_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(64)   # Fix temporal dimension
        )

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(32 * 64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): shape [B, T, C]

        Returns:
            logits (Tensor): shape [B, output_dim]
        """
        # [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)

        # Temporal convolution
        x = self.time_conv(x)   # [B, 32, 64]

        # Flatten
        x = x.flatten(1)        # [B, 2048]

        # Classification
        logits = self.mlp(x)
        return logits


if __name__ == "__main__":
    # Simple sanity check
    B, T, C = 2, 5121, 3200
    dummy_input = torch.randn(B, T, C)
    model = TimeConvMLP()
    out = model(dummy_input)
    print("Output shape:", out.shape)  # Expected: [2, 3]
