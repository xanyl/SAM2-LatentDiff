"""Loss functions for SAM2-LatentDiff two-stage training."""

import torch
import torch.nn.functional as F


def latent_loss(enhanced: torch.Tensor, gt: torch.Tensor, config: dict) -> torch.Tensor:
    """Stage 1 loss: L1 + weighted L2 in latent space.
    
    Args:
        enhanced: (B, 4, H, W) predicted enhanced latent.
        gt: (B, 4, H, W) ground-truth latent.
        config: Stage 1 config with loss weights.
    """
    loss_l1 = F.l1_loss(enhanced, gt)
    loss_l2 = F.mse_loss(enhanced, gt)
    return config["loss_l1_weight"] * loss_l1 + config["loss_l2_weight"] * loss_l2


def pixel_loss(refined: torch.Tensor, gt_rgb: torch.Tensor,
               lpips_fn, config: dict) -> torch.Tensor:
    """Stage 2 loss: Charbonnier + weighted L2 + weighted LPIPS in pixel space.
    
    Args:
        refined: (B, 3, H, W) refined output in [0, 1].
        gt_rgb: (B, 3, H, W) ground-truth RGB in [0, 1].
        lpips_fn: LPIPS loss function (AlexNet).
        config: Stage 2 config with loss weights.
    """
    # Charbonnier loss (smooth L1)
    loss_char = torch.sqrt((refined - gt_rgb) ** 2 + 1e-6).mean()
    # L2 loss
    loss_l2 = F.mse_loss(refined, gt_rgb)
    # LPIPS (expects [-1, 1] input)
    loss_lpips = lpips_fn(refined * 2 - 1, gt_rgb * 2 - 1).mean()

    return (config["loss_charbonnier_weight"] * loss_char
            + config["loss_l2_weight"] * loss_l2
            + config["loss_lpips_weight"] * loss_lpips)
