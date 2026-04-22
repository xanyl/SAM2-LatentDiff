#!/usr/bin/env python3
"""
Ablation Studies for SAM2-LatentDiff.

Ablation A: Replace SAM2 features with zero vectors.
Ablation B: Replace SAM2 features with random noise.

Usage:
    python scripts/run_ablation.py --config configs/default.yaml \
        --stage1_checkpoint checkpoints/stage1/best.pt \
        --stage2_checkpoint checkpoints/stage2/best.pt
"""

import argparse, os, sys, time
import numpy as np, torch, yaml, lpips
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.pipeline import SAM2LatentDiffPipeline
from src.data.dataset import build_datasets
from src.utils.metrics import compute_psnr, compute_ssim, compute_lpips, print_metrics


def run_ablation(pipeline, test_dataset, lpips_fn, mode="zero"):
    """Run a single ablation experiment.
    
    Args:
        mode: 'zero' for Ablation A, 'random' for Ablation B.
    """
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for idx in tqdm(range(len(test_dataset)), desc=f"Ablation ({mode})"):
        low, _, feat, gt_rgb, low_rgb = test_dataset[idx]
        low_t = low.unsqueeze(0).cuda()
        gt_rgb_t = gt_rgb.unsqueeze(0).cuda()
        low_rgb_t = low_rgb.unsqueeze(0).cuda()

        with torch.no_grad():
            x_in = torch.cat([low_t, low_t], dim=1)
            t_zero = torch.zeros(1, dtype=torch.long, device="cuda")

            if mode == "zero":
                ctx = torch.zeros(1, 77, 768, device="cuda")
            else:  # random
                ctx = torch.randn(1, 77, 768, device="cuda")

            with autocast():
                pred = pipeline.unet(x_in, t_zero, encoder_hidden_states=ctx).sample
            enhanced = low_t + pred
            decoded = pipeline.decode_latent(enhanced)
            refined = pipeline.refiner(decoded, low_rgb_t)

        gt_np = gt_rgb.permute(1, 2, 0).numpy()
        ref_np = refined[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)

        psnr_vals.append(compute_psnr(gt_np, ref_np))
        ssim_vals.append(compute_ssim(gt_np, ref_np))
        lpips_vals.append(compute_lpips(gt_rgb_t, refined, lpips_fn))

    return psnr_vals, ssim_vals, lpips_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--stage2_checkpoint", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config["paths"]["data_dir"]
    dataset_dir = args.dataset_dir or config["paths"]["dataset_dir"]
    output_dir = args.output_dir or os.path.join(config["paths"]["output_dir"], "ablations")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("SAM2-LatentDiff — Ablation Studies")
    print("=" * 60)

    pipeline = SAM2LatentDiffPipeline(config)
    pipeline.load_stage1(args.stage1_checkpoint)
    pipeline.load_vae(os.path.join(config["paths"]["checkpoint_dir"], "vae"))
    if args.stage2_checkpoint:
        pipeline.load_stage2(args.stage2_checkpoint)
    pipeline.unet.eval(); pipeline.sam2_adapter.eval(); pipeline.refiner.eval()

    _, test_dataset = build_datasets(config, data_dir, dataset_dir, stage=2)
    lpips_fn = lpips.LPIPS(net="alex").cuda()

    # Ablation A: No SAM2
    print("\n--- Ablation A: Zero features (no SAM2) ---")
    a_psnr, a_ssim, a_lpips = run_ablation(pipeline, test_dataset, lpips_fn, "zero")
    print_metrics(a_psnr, a_ssim, a_lpips, "  Ablation A: ")

    # Ablation B: Random conditioning
    print("\n--- Ablation B: Random noise conditioning ---")
    b_psnr, b_ssim, b_lpips = run_ablation(pipeline, test_dataset, lpips_fn, "random")
    print_metrics(b_psnr, b_ssim, b_lpips, "  Ablation B: ")

    # Save
    import pandas as pd
    df = pd.DataFrame({
        "Method": ["Ablation A (No SAM2)", "Ablation B (Random)"],
        "PSNR": [np.mean(a_psnr), np.mean(b_psnr)],
        "SSIM": [np.mean(a_ssim), np.mean(b_ssim)],
        "LPIPS": [np.mean(a_lpips), np.mean(b_lpips)],
    })
    df.to_csv(os.path.join(output_dir, "ablation_results.csv"), index=False)
    print(f"\nSaved: {os.path.join(output_dir, 'ablation_results.csv')}")


if __name__ == "__main__":
    main()
