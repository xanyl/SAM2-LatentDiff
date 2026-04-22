#!/usr/bin/env python3
"""
Download LOL-v2 dataset and model checkpoints.

Usage:
    python scripts/download_data.py --output_dir data/
"""

import argparse, os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/")
    parser.add_argument("--checkpoints_dir", default="checkpoints/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    print("=" * 60)
    print("Downloading LOL-v2 Dataset and Checkpoints")
    print("=" * 60)

    # --- LOL-v2 Dataset ---
    print("\n[1/4] Downloading LOL-v2 Real...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="okhater/lolv2-real", repo_type="dataset",
        local_dir=os.path.join(args.output_dir, "LOLv2_Real")
    )
    print("  Done!")

    print("\n[2/4] Downloading LOL-v2 Synthetic...")
    snapshot_download(
        repo_id="okhater/lolv2-synthetic", repo_type="dataset",
        local_dir=os.path.join(args.output_dir, "LOLv2_Synthetic")
    )
    print("  Done!")

    # --- SAM2 Checkpoint ---
    print("\n[3/4] Downloading SAM2 Hiera-Large checkpoint...")
    sam2_dir = os.path.join(args.checkpoints_dir, "sam2")
    os.makedirs(sam2_dir, exist_ok=True)
    os.system(
        f"wget -q --show-progress -O {sam2_dir}/sam2_hiera_large.pt "
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    )
    print("  Done!")

    # --- Stable Diffusion VAE ---
    print("\n[4/4] Downloading Stable Diffusion VAE...")
    from diffusers import AutoencoderKL
    import torch
    vae_dir = os.path.join(args.checkpoints_dir, "vae")
    vae = AutoencoderKL.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16
    )
    vae.save_pretrained(vae_dir)
    print("  Done!")

    print("\n" + "=" * 60)
    print("All downloads complete!")
    print(f"  Dataset: {args.output_dir}")
    print(f"  Checkpoints: {args.checkpoints_dir}")


if __name__ == "__main__":
    main()
