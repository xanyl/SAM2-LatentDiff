#!/usr/bin/env python3
"""
Preprocess LOL-v2: Extract SAM2 features and VAE latents.

Runs once to cache all features to disk, avoiding loading heavy
models during training.

Usage:
    python scripts/preprocess.py \
        --data_dir data/ \
        --output_dir data/preprocessed \
        --sam2_checkpoint checkpoints/sam2/sam2_hiera_large.pt \
        --vae_checkpoint checkpoints/vae
"""

import argparse, os, glob
import torch, numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def extract_sam2_features(data_dir, output_dir, sam2_checkpoint):
    """Extract SAM2 encoder features for all low-light images."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    feat_dir = os.path.join(output_dir, "sam2_features")
    os.makedirs(feat_dir, exist_ok=True)

    print("Loading SAM2 model...")
    sam2 = build_sam2("sam2_hiera_l.yaml", sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2)

    image_paths = []
    for subset in ["LOLv2_Real", "LOLv2_Synthetic"]:
        for split in ["Train", "Test"]:
            input_dir = os.path.join(data_dir, subset, split, "Input")
            image_paths += sorted(glob.glob(os.path.join(input_dir, "*.png")))
            image_paths += sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

    print(f"Extracting SAM2 features for {len(image_paths)} images...")
    for img_path in tqdm(image_paths):
        name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(feat_dir, f"{name}.pt")
        if os.path.exists(out_path):
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        predictor.set_image(img)
        features = predictor.get_image_embedding()
        torch.save(features.cpu(), out_path)

    del sam2, predictor
    torch.cuda.empty_cache()
    print(f"  Saved {len(image_paths)} feature files to {feat_dir}")


def extract_vae_latents(data_dir, output_dir, vae_checkpoint):
    """Encode all images (low + GT) to VAE latent space."""
    from diffusers import AutoencoderKL

    latent_dir = os.path.join(output_dir, "vae_latents")
    os.makedirs(latent_dir, exist_ok=True)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(vae_checkpoint, torch_dtype=torch.float16).to("cuda")
    vae.eval()

    transform = transforms.Compose([
        transforms.Resize((400, 600)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    all_images = []
    for subset in ["LOLv2_Real", "LOLv2_Synthetic"]:
        for split in ["Train", "Test"]:
            for folder in ["Input", "GT"]:
                img_dir = os.path.join(data_dir, subset, split, folder)
                if os.path.exists(img_dir):
                    all_images += [(p, folder) for p in sorted(
                        glob.glob(os.path.join(img_dir, "*.png")) +
                        glob.glob(os.path.join(img_dir, "*.jpg"))
                    )]

    print(f"Encoding {len(all_images)} images to latent space...")
    for img_path, folder in tqdm(all_images):
        name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(latent_dir, f"{folder}_{name}.pt")
        if os.path.exists(out_path):
            continue
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).half().to("cuda")
        with torch.no_grad():
            latent = vae.encode(x).latent_dist.sample() * 0.18215
        torch.save(latent.cpu(), out_path)

    del vae
    torch.cuda.empty_cache()
    print(f"  Saved {len(all_images)} latent files to {latent_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to LOLv2_Real/, LOLv2_Synthetic/")
    parser.add_argument("--output_dir", required=True, help="Where to save preprocessed files")
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--skip_sam2", action="store_true")
    parser.add_argument("--skip_vae", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("SAM2-LatentDiff — Preprocessing")
    print("=" * 60)

    if not args.skip_sam2:
        print("\n[1/2] SAM2 Feature Extraction")
        extract_sam2_features(args.data_dir, args.output_dir, args.sam2_checkpoint)

    if not args.skip_vae:
        print("\n[2/2] VAE Latent Encoding")
        extract_vae_latents(args.data_dir, args.output_dir, args.vae_checkpoint)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    # Summary
    feat_dir = os.path.join(args.output_dir, "sam2_features")
    lat_dir = os.path.join(args.output_dir, "vae_latents")
    n_feat = len(os.listdir(feat_dir)) if os.path.exists(feat_dir) else 0
    n_lat = len(os.listdir(lat_dir)) if os.path.exists(lat_dir) else 0
    print(f"  SAM2 features: {n_feat} files")
    print(f"  VAE latents:   {n_lat} files")


if __name__ == "__main__":
    main()
