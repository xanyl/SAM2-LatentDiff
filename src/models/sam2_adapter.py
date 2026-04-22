"""
SAM2 Feature Adapter for Stable Diffusion Cross-Attention.

Projects SAM2 encoder features (256-D) to match SD's cross-attention
dimension (768-D), producing a sequence of 77 tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM2Adapter(nn.Module):
    """Projects SAM2 hierarchical encoder features to SD cross-attention space.
    
    Architecture:
        Conv2d(256→768, 1×1) → GELU → Conv2d(768→768, 1×1) → AdaptivePool → (B, 77, 768)
    
    Args:
        sam2_dim: Input feature dimension from SAM2 encoder (default: 256).
        sd_cross_dim: Stable Diffusion cross-attention dimension (default: 768).
        seq_len: Output sequence length matching SD's expected context (default: 77).
    """

    def __init__(self, sam2_dim: int = 256, sd_cross_dim: int = 768, seq_len: int = 77):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(sam2_dim, sd_cross_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(sd_cross_dim, sd_cross_dim, kernel_size=1),
        )
        self.seq_len = seq_len

    def forward(self, sam2_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sam2_features: (B, 256, H, W) from SAM2 encoder.
        
        Returns:
            (B, 77, 768) projected features for SD cross-attention.
        """
        x = self.proj(sam2_features)                      # (B, 768, H, W)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)        # (B, H*W, 768)
        if x.shape[1] > self.seq_len:
            x = F.adaptive_avg_pool1d(
                x.permute(0, 2, 1), self.seq_len
            ).permute(0, 2, 1)                             # (B, 77, 768)
        return x
