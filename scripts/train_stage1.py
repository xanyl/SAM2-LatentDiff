#!/usr/bin/env python3
"""
Stage 1: Latent-Space Training for SAM2-LatentDiff.

Trains LoRA adapters + SAM2 Adapter + modified conv_in on precomputed
VAE latents and SAM2 features using L1 + L2 loss in latent space.

Usage:
    python scripts/train_stage1.py --config configs/default.yaml
"""

import argparse
import copy
import os
import sys
import time

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.pipeline import SAM2LatentDiffPipeline
from src.data.dataset import build_datasets
from src.losses.losses import latent_loss


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2-LatentDiff Stage 1 Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override paths from CLI
    data_dir = args.data_dir or config["paths"]["data_dir"]
    dataset_dir = args.dataset_dir or config["paths"]["dataset_dir"]
    output_dir = args.output_dir or os.path.join(config["paths"]["checkpoint_dir"], "stage1")
    os.makedirs(output_dir, exist_ok=True)

    cfg = config["stage1"]
    print("=" * 60)
    print("SAM2-LatentDiff — Stage 1: Latent Training")
    print("=" * 60)

    # --- Build model ---
    print("\nBuilding model...")
    pipeline = SAM2LatentDiffPipeline(config)
    counts = pipeline.param_count()
    print(f"  U-Net: {counts['unet_total']/1e6:.1f}M total, {counts['unet_trainable']/1e6:.1f}M trainable")
    print(f"  Adapter: {counts['adapter_total']/1e3:.0f}K")
    print(f"  Total trainable: {counts['total_trainable']/1e6:.2f}M ({100*counts['total_trainable']/counts['total']:.1f}%)")

    # --- Build datasets ---
    print("\nLoading datasets...")
    train_dataset, test_dataset = build_datasets(config, data_dir, dataset_dir, stage=1)
    dataloader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True
    )

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        pipeline.get_trainable_params_stage1(),
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scaler = GradScaler() if cfg["mixed_precision"] else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # --- EMA ---
    ema_adapter = copy.deepcopy(pipeline.sam2_adapter.state_dict())
    ema_conv_in = copy.deepcopy(pipeline.unet.base_model.model.conv_in.state_dict())
    ema_decay = cfg["ema_decay"]

    # --- Resume ---
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, weights_only=True)
        pipeline.load_stage1(args.resume)
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed from epoch {start_epoch}")

    # --- Training loop ---
    pipeline.unet.train()
    pipeline.sam2_adapter.train()
    best_loss = float("inf")
    log = []

    print(f"\nTraining for {cfg['epochs']} epochs, batch={cfg['batch_size']}, lr={cfg['learning_rate']}")
    start_time = time.time()

    for epoch in range(start_epoch, cfg["epochs"]):
        total_loss = 0
        n = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")

        for low, gt, feat in pbar:
            low, gt, feat = low.cuda(), gt.cuda(), feat.cuda()

            enhanced = pipeline.forward_latent(low, feat)
            loss = latent_loss(enhanced, gt, cfg)

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(pipeline.get_trainable_params_stage1(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pipeline.get_trainable_params_stage1(), cfg["grad_clip"])
                optimizer.step()

            # EMA update
            with torch.no_grad():
                for k, v in pipeline.sam2_adapter.state_dict().items():
                    ema_adapter[k] = ema_decay * ema_adapter[k] + (1 - ema_decay) * v
                for k, v in pipeline.unet.base_model.model.conv_in.state_dict().items():
                    ema_conv_in[k] = ema_decay * ema_conv_in[k] + (1 - ema_decay) * v

            total_loss += loss.item()
            n += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / n
        elapsed = (time.time() - start_time) / 60
        log.append({"epoch": epoch + 1, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]})

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save with EMA weights
            lora_state = {k: v for k, v in pipeline.unet.state_dict().items() if "lora" in k}
            torch.save({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "unet_lora": lora_state,
                "sam2_adapter": copy.deepcopy(ema_adapter),
                "conv_in": copy.deepcopy(ema_conv_in),
            }, os.path.join(output_dir, "best.pt"))
            print(f"  ★ Epoch {epoch+1}: loss={avg_loss:.4f} [SAVED] [{elapsed:.1f}min]")
        else:
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f} [{elapsed:.1f}min]")

    # Save training log
    import pandas as pd
    pd.DataFrame(log).to_csv(os.path.join(output_dir, "training_log.csv"), index=False)
    print(f"\nStage 1 complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {os.path.join(output_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
