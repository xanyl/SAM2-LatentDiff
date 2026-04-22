#!/usr/bin/env python3
"""
Evaluate SAM2-LatentDiff on LOL-v2 Real test set.

Computes PSNR, SSIM, LPIPS for both latent-only and refined outputs.
Generates visual comparison figures.

Usage:
    python scripts/evaluate.py --config configs/default.yaml \
        --stage1_checkpoint checkpoints/stage1/best.pt \
        --stage2_checkpoint checkpoints/stage2/best.pt
"""

import argparse, os, sys, time
import numpy as np, torch, yaml, lpips
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.pipeline import SAM2LatentDiffPipeline
from src.data.dataset import build_datasets
from src.utils.metrics import compute_psnr, compute_ssim, compute_lpips, print_metrics


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
    output_dir = args.output_dir or config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    print("=" * 60)
    print("SAM2-LatentDiff — Evaluation")
    print("=" * 60)

    # Build model
    pipeline = SAM2LatentDiffPipeline(config)
    pipeline.load_stage1(args.stage1_checkpoint)
    pipeline.load_vae(os.path.join(config["paths"]["checkpoint_dir"], "vae"))
    if args.stage2_checkpoint:
        pipeline.load_stage2(args.stage2_checkpoint)
    pipeline.unet.eval()
    pipeline.sam2_adapter.eval()
    pipeline.refiner.eval()

    # Dataset
    _, test_dataset = build_datasets(config, data_dir, dataset_dir, stage=2)
    lpips_fn = lpips.LPIPS(net="alex").cuda()

    # Evaluate
    lat_psnr, lat_ssim, lat_lpips = [], [], []
    ref_psnr, ref_ssim, ref_lpips = [], [], []
    times = []
    vis_idx = config["eval"]["vis_indices"]

    fig, axes = plt.subplots(len(vis_idx), 4, figsize=(24, 6 * len(vis_idx)))
    for j, t in enumerate(["Low-Light", "Latent Only", "Refined (Ours)", "Ground Truth"]):
        axes[0, j].set_title(t, fontsize=16, fontweight="bold")

    print(f"\nEvaluating on {len(test_dataset)} test images...")
    for idx in tqdm(range(len(test_dataset))):
        low, _, feat, gt_rgb, low_rgb = test_dataset[idx]
        low_t = low.unsqueeze(0).cuda()
        feat_t = feat.unsqueeze(0).cuda()
        low_rgb_t = low_rgb.unsqueeze(0).cuda()
        gt_rgb_t = gt_rgb.unsqueeze(0).cuda()

        t0 = time.time()
        with torch.no_grad():
            result = pipeline(low_t, feat_t, low_rgb_t)
        times.append(time.time() - t0)

        gt_np = gt_rgb.permute(1, 2, 0).numpy()
        dec_np = result["decoded"][0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        ref_np = result["refined"][0].cpu().permute(1, 2, 0).numpy().clip(0, 1)

        lat_psnr.append(compute_psnr(gt_np, dec_np))
        lat_ssim.append(compute_ssim(gt_np, dec_np))
        lat_lpips.append(compute_lpips(gt_rgb_t, result["decoded"], lpips_fn))

        ref_psnr.append(compute_psnr(gt_np, ref_np))
        ref_ssim.append(compute_ssim(gt_np, ref_np))
        ref_lpips.append(compute_lpips(gt_rgb_t, result["refined"], lpips_fn))

        if idx in vis_idx:
            i = vis_idx.index(idx)
            axes[i, 0].imshow(low_rgb.permute(1, 2, 0).numpy()); axes[i, 0].axis("off")
            axes[i, 1].imshow(dec_np); axes[i, 1].axis("off")
            axes[i, 2].imshow(ref_np); axes[i, 2].axis("off")
            axes[i, 2].set_ylabel(f"PSNR: {ref_psnr[-1]:.1f}", fontsize=12)
            axes[i, 3].imshow(gt_np); axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "qualitative_comparison.png"), dpi=200, bbox_inches="tight")
    print(f"\nSaved: qualitative_comparison.png")

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS — SAM2-LatentDiff (LOL-v2 Real Test)")
    print("=" * 60)
    print("\n  Latent Only (without refiner):")
    print_metrics(lat_psnr, lat_ssim, lat_lpips, "    ")
    print("\n  With PixelRefiner (final):")
    print_metrics(ref_psnr, ref_ssim, ref_lpips, "    ")
    print(f"\n  Inference time: {np.mean(times)*1000:.1f} ms/image")

    # Save CSV
    import pandas as pd
    pd.DataFrame({
        "latent_psnr": lat_psnr, "latent_ssim": lat_ssim, "latent_lpips": lat_lpips,
        "refined_psnr": ref_psnr, "refined_ssim": ref_ssim, "refined_lpips": ref_lpips,
    }).to_csv(os.path.join(output_dir, "per_image_metrics.csv"), index=False)

    summary = pd.DataFrame({
        "Method": ["Latent Only", "With PixelRefiner"],
        "PSNR": [np.mean(lat_psnr), np.mean(ref_psnr)],
        "SSIM": [np.mean(lat_ssim), np.mean(ref_ssim)],
        "LPIPS": [np.mean(lat_lpips), np.mean(ref_lpips)],
    })
    summary.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)
    print(f"\nSaved: per_image_metrics.csv, results_summary.csv")


if __name__ == "__main__":
    main()
