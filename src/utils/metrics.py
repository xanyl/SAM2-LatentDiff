"""Evaluation metrics: PSNR, SSIM, LPIPS."""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    """Compute PSNR between ground truth and prediction."""
    return float(peak_signal_noise_ratio(gt, pred, data_range=data_range))


def compute_ssim(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    """Compute SSIM between ground truth and prediction."""
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=data_range))


def compute_lpips(gt_tensor, pred_tensor, lpips_fn) -> float:
    """Compute LPIPS between tensors in [0, 1]."""
    return lpips_fn(pred_tensor * 2 - 1, gt_tensor * 2 - 1).item()


def print_metrics(psnr_vals, ssim_vals, lpips_vals, label=""):
    """Print formatted metrics summary."""
    prefix = f"  {label}" if label else "  "
    print(f"{prefix}PSNR:  {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f} dB")
    print(f"{prefix}SSIM:  {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}")
    print(f"{prefix}LPIPS: {np.mean(lpips_vals):.4f} ± {np.std(lpips_vals):.4f}")
