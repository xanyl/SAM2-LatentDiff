#!/usr/bin/env python3
"""
Generate all figures for the paper/report.

Usage:
    python scripts/visualize.py --results_dir outputs/
"""

import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/")
    parser.add_argument("--fig_dir", default=None)
    args = parser.parse_args()

    fig_dir = args.fig_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    TEAL, MB, RD, OR = "#0D9488", "#2539CB", "#CC3333", "#E8593C"
    DB, GR = "#1A237E", "#2C8C3C"

    # --- Figure 1: PSNR Bar Chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ["CUGD\n(Baseline)", "Latent\nOnly", "No SAM2\n(Abl A)", "Random\n(Abl B)", "SAM2-LatentDiff\n+ Refiner"]
    psnr = [21.0, 17.19, 18.08, 17.87, 18.61]
    colors = ["#995522", "#6666AA", RD, RD, TEAL]
    bars = ax.bar(methods, psnr, color=colors, edgecolor="white", linewidth=2, width=0.6)
    for bar, val in zip(bars, psnr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.2f}", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylabel("PSNR (dB) \u2191", fontsize=14, fontweight="bold")
    ax.set_title("PSNR Comparison on LOL-v2 Real Test Set", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(psnr) + 2.5)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "psnr_comparison.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "psnr_comparison.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: psnr_comparison.png/.pdf")

    # --- Figure 2: Multi-Metric ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ms = ["CUGD", "Latent\nOnly", "No SAM2", "Random", "Ours"]
    pv = [21.0, 17.19, 18.08, 17.87, 18.61]
    sv = [0.83, 0.5758, 0.6731, 0.6733, 0.7085]
    lv = [0.20, 0.2812, 0.3091, 0.3225, 0.2399]
    cm = ["#995522", "#6666AA", RD, "#DD5555", TEAL]
    for ax, vals, title, fmt in zip(axes, [pv, sv, lv], ["PSNR \u2191", "SSIM \u2191", "LPIPS \u2193"], [".2f", ".4f", ".4f"]):
        bars = ax.bar(ms, vals, color=cm, edgecolor="white", width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.01, f"{v:{fmt}}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.suptitle("Multi-Metric Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "multi_metric.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: multi_metric.png")

    # --- Figure 3: PixelRefiner Impact ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (ax, m, b, a, d) in enumerate(zip(axes,
        ["PSNR (dB) \u2191", "SSIM \u2191", "LPIPS \u2193"],
        [17.19, 0.5758, 0.2812], [18.61, 0.7085, 0.2399], ["+1.42", "+0.133", "\u22120.041"])):
        bars = ax.bar(["Before", "After"], [b, a], color=["#88552280", TEAL], edgecolor="white", width=0.5)
        for bar, val in zip(bars, [b, a]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(b,a)*0.02, f"{val}", ha="center", fontsize=13, fontweight="bold")
        ax.set_title(m, fontsize=13, fontweight="bold")
        ax.text(0.5, max(b,a)*0.5, d, ha="center", fontsize=16, fontweight="bold", color=GR,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E0F5E8", edgecolor=GR))
        ax.grid(axis="y", alpha=0.3); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.suptitle("PixelRefiner Impact (87K params)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "refiner_impact.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: refiner_impact.png")

    # --- Figure 4: Performance Context ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ml = ["ReDDiT (CVPR25)", "Retinexformer", "GSAD (NeurIPS23)", "Reti-Diff (ICLR25)", "CUGD (TCSVT25)", "Ours"]
    pl = [31.25, 27.18, 27.0, 23.0, 21.0, 18.61]
    cl = ["#777", "#777", "#777", "#777", "#995522", TEAL]
    bars = ax.barh(ml[::-1], pl[::-1], color=cl[::-1], edgecolor="white", height=0.5)
    for b, v in zip(bars, pl[::-1]):
        ax.text(b.get_width()+0.3, b.get_y()+b.get_height()/2, f"{v:.2f} dB", va="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("PSNR (dB)", fontsize=13)
    ax.set_title("LLIE Methods on LOL-v2 Real", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "performance_context.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: performance_context.png")

    print(f"\nAll figures saved to: {fig_dir}/")


if __name__ == "__main__":
    main()
