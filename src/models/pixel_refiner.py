"""
PixelRefiner: Lightweight CNN for post-VAE artifact correction.

Following DiffBIR (Lin et al., 2023) and StableSR (Wang et al., IJCV 2024),
this module corrects VAE reconstruction artifacts in pixel space.
"""

import torch
import torch.nn as nn


class PixelRefiner(nn.Module):
    """Lightweight CNN that refines VAE-decoded images in pixel space.
    
    Takes the decoded SAM2-LatentDiff output concatenated with the original
    low-light image (6 channels total) and predicts a residual correction.
    Zero-initialized output layer ensures identity mapping at initialization.
    
    Architecture:
        Conv2d(6→ch) → [GELU → Conv2d(ch→ch)] × 4 → Conv2d(ch→3, zero-init)
    
    Args:
        channels: Hidden channel dimension (default: 48).
    """

    def __init__(self, channels: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv2d(channels, 3, 3, padding=1)

        # Zero-init: refiner starts as identity (decoded passes through unchanged)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, decoded: torch.Tensor, low_rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            decoded: (B, 3, H, W) VAE-decoded enhanced image in [0, 1].
            low_rgb: (B, 3, H, W) original low-light image in [0, 1].
        
        Returns:
            (B, 3, H, W) refined image in [0, 1].
        """
        x = torch.cat([decoded, low_rgb], dim=1)  # (B, 6, H, W)
        h = self.net(x)
        return (decoded + self.out(h)).clamp(0, 1)
