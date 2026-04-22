#!/usr/bin/env python3
"""
Stage 2: Pixel-Space Refinement Training.

Trains the PixelRefiner CNN with frozen latent pipeline, using
Charbonnier + L2 + LPIPS loss with early stopping.

Usage:
    python scripts/train_stage2.py --config configs/default.yaml \
        --stage1_checkpoint checkpoints/stage1/best.pt
"""

import argparse, copy, os, sys, time
import torch, yaml, lpips
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.pipeline import SAM2LatentDiffPipeline
from src.data.dataset import build_datasets
from src.losses.losses import pixel_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config["paths"]["data_dir"]
    dataset_dir = args.dataset_dir or config["paths"]["dataset_dir"]
    output_dir = args.output_dir or os.path.join(config["paths"]["checkpoint_dir"], "stage2")
    os.makedirs(output_dir, exist_ok=True)
    cfg = config["stage2"]

    print("=" * 60)
    print("SAM2-LatentDiff — Stage 2: Pixel Refinement")
    print("=" * 60)

    # Build model + load Stage 1
    pipeline = SAM2LatentDiffPipeline(config)
    pipeline.load_stage1(args.stage1_checkpoint)
    pipeline.load_vae(os.path.join(config["paths"]["checkpoint_dir"], "vae"))
    pipeline.freeze_for_stage2()
    print(f"  Refiner params: {sum(p.numel() for p in pipeline.refiner.parameters()):,}")

    # Datasets
    print("\nLoading datasets...")
    train_dataset, test_dataset = build_datasets(config, data_dir, dataset_dir, stage=2)
    dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(pipeline.get_trainable_params_stage2(),
                                  lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    lpips_fn = lpips.LPIPS(net="alex").cuda()

    ema_state = copy.deepcopy(pipeline.refiner.state_dict())
    best_loss, patience_counter = float("inf"), 0
    save_path = os.path.join(output_dir, "best.pt")
    pipeline.refiner.train()
    start_time = time.time()
    log = []

    print(f"\nTraining for up to {cfg['epochs']} epochs (patience={cfg['patience']})")

    for epoch in range(cfg["epochs"]):
        total_loss, n = 0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")

        for low, gt_lat, feat, gt_rgb, low_rgb in pbar:
            low, feat = low.cuda(), feat.cuda()
            gt_rgb, low_rgb = gt_rgb.cuda(), low_rgb.cuda()

            with torch.no_grad():
                enhanced_latent = pipeline.forward_latent(low, feat)
                decoded = pipeline.decode_latent(enhanced_latent)

            refined = pipeline.refiner(decoded, low_rgb)
            loss = pixel_loss(refined, gt_rgb, lpips_fn, cfg)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pipeline.get_trainable_params_stage2(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                for k in ema_state:
                    ema_state[k] = cfg["ema_decay"] * ema_state[k] + (1 - cfg["ema_decay"]) * pipeline.refiner.state_dict()[k]

            total_loss += loss.item()
            n += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg = total_loss / n
        log.append({"epoch": epoch + 1, "loss": avg})
        elapsed = (time.time() - start_time) / 60

        if avg < best_loss:
            best_loss = avg
            patience_counter = 0
            torch.save({"refiner": copy.deepcopy(ema_state), "epoch": epoch + 1, "loss": avg}, save_path)
            print(f"  ★ Epoch {epoch+1}: loss={avg:.4f} [SAVED] [{elapsed:.1f}min]")
        else:
            patience_counter += 1
            print(f"    Epoch {epoch+1}: loss={avg:.4f} (patience {patience_counter}/{cfg['patience']}) [{elapsed:.1f}min]")

        if patience_counter >= cfg["patience"]:
            print(f"\n  Early stopping at epoch {epoch+1}.")
            break

    import pandas as pd
    pd.DataFrame(log).to_csv(os.path.join(output_dir, "training_log.csv"), index=False)
    print(f"\nStage 2 complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
