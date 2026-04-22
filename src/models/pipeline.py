"""
SAM2-LatentDiff Pipeline: End-to-end model assembly and inference.

Combines pretrained SD U-Net (with LoRA), SAM2 Adapter, VAE, and PixelRefiner
into a single callable pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from diffusers import UNet2DConditionModel, AutoencoderKL
from peft import LoraConfig, get_peft_model

from .sam2_adapter import SAM2Adapter
from .pixel_refiner import PixelRefiner


class SAM2LatentDiffPipeline(nn.Module):
    """Complete SAM2-LatentDiff pipeline.
    
    Args:
        config: Dictionary with model configuration.
        device: Target device.
    """

    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.config = config
        model_cfg = config["model"]

        # --- Load pretrained SD U-Net ---
        self.unet = UNet2DConditionModel.from_pretrained(
            model_cfg["sd_model_id"], subfolder="unet", torch_dtype=torch.float32
        ).to(device)
        self.unet.requires_grad_(False)

        # --- Add LoRA ---
        lora_config = LoraConfig(
            r=model_cfg["lora_rank"],
            lora_alpha=model_cfg["lora_alpha"],
            target_modules=model_cfg["lora_target_modules"],
            lora_dropout=model_cfg["lora_dropout"],
        )
        self.unet = get_peft_model(self.unet, lora_config)

        # --- SAM2 Adapter ---
        self.sam2_adapter = SAM2Adapter(
            sam2_dim=model_cfg["sam2_dim"],
            sd_cross_dim=model_cfg["sd_cross_dim"],
            seq_len=model_cfg["adapter_seq_len"],
        ).to(device)

        # --- Modify conv_in for 8-channel input ---
        old_conv = self.unet.base_model.model.conv_in
        new_conv = nn.Conv2d(
            8, old_conv.out_channels,
            old_conv.kernel_size, old_conv.stride, old_conv.padding
        ).to(device)
        with torch.no_grad():
            new_conv.weight[:, :4] = old_conv.weight
            new_conv.weight[:, 4:] = 0.0
            new_conv.bias = old_conv.bias
        self.unet.base_model.model.conv_in = new_conv

        # --- PixelRefiner ---
        self.refiner = PixelRefiner(channels=model_cfg["refiner_channels"]).to(device)

        # --- VAE (frozen, loaded separately) ---
        self.vae = None  # Loaded via load_vae()
        self.vae_scale = config["data"]["vae_scale_factor"]

    def load_vae(self, vae_path: str):
        """Load pretrained VAE decoder."""
        self.vae = AutoencoderKL.from_pretrained(
            vae_path, torch_dtype=torch.float16
        ).to(self.device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to RGB image [0, 1]."""
        with torch.no_grad():
            img = self.vae.decode(latent.half() / self.vae_scale).sample
        return img.float().clamp(-1, 1) * 0.5 + 0.5

    def forward_latent(
        self, low_latent: torch.Tensor, sam2_feat: torch.Tensor
    ) -> torch.Tensor:
        """Stage 1: Latent-space enhancement.
        
        Args:
            low_latent: (B, 4, H, W) VAE-encoded low-light image.
            sam2_feat: (B, 256, h, w) SAM2 encoder features.
        
        Returns:
            (B, 4, H, W) enhanced latent.
        """
        x_in = torch.cat([low_latent, low_latent], dim=1)  # (B, 8, H, W)
        sam2_ctx = self.sam2_adapter(sam2_feat)               # (B, 77, 768)
        t_zero = torch.zeros(
            low_latent.shape[0], dtype=torch.long, device=self.device
        )
        with autocast():
            pred = self.unet(x_in, t_zero, encoder_hidden_states=sam2_ctx).sample
        return low_latent + pred  # Residual connection

    def forward(
        self,
        low_latent: torch.Tensor,
        sam2_feat: torch.Tensor,
        low_rgb: torch.Tensor,
    ) -> dict:
        """Full pipeline: latent enhancement → decode → pixel refinement.
        
        Args:
            low_latent: (B, 4, H, W) VAE-encoded low-light image.
            sam2_feat: (B, 256, h, w) SAM2 encoder features.
            low_rgb: (B, 3, H, W) original low-light image [0, 1].
        
        Returns:
            Dictionary with 'enhanced_latent', 'decoded', and 'refined'.
        """
        enhanced_latent = self.forward_latent(low_latent, sam2_feat)
        decoded = self.decode_latent(enhanced_latent)
        refined = self.refiner(decoded, low_rgb)
        return {
            "enhanced_latent": enhanced_latent,
            "decoded": decoded,
            "refined": refined,
        }

    def load_stage1(self, checkpoint_path: str):
        """Load Stage 1 checkpoint (LoRA + SAM2 Adapter + conv_in)."""
        ckpt = torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        self.sam2_adapter.load_state_dict(ckpt["sam2_adapter"])
        self.unet.base_model.model.conv_in.load_state_dict(ckpt["conv_in"])
        state = self.unet.state_dict()
        for k, v in ckpt["unet_lora"].items():
            if k in state:
                state[k] = v
        self.unet.load_state_dict(state)

    def load_stage2(self, checkpoint_path: str):
        """Load Stage 2 checkpoint (PixelRefiner)."""
        ckpt = torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        self.refiner.load_state_dict(ckpt["refiner"])

    def get_trainable_params_stage1(self):
        """Return trainable parameters for Stage 1 optimizer."""
        return (
            list(filter(lambda p: p.requires_grad, self.unet.parameters()))
            + list(self.sam2_adapter.parameters())
            + list(self.unet.base_model.model.conv_in.parameters())
        )

    def get_trainable_params_stage2(self):
        """Return trainable parameters for Stage 2 optimizer."""
        return list(self.refiner.parameters())

    def freeze_for_stage2(self):
        """Freeze everything except PixelRefiner for Stage 2."""
        self.unet.eval()
        self.sam2_adapter.eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.sam2_adapter.parameters():
            p.requires_grad_(False)

    def param_count(self) -> dict:
        """Return parameter counts for each component."""
        def count(m):
            total = sum(p.numel() for p in m.parameters())
            train = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, train

        unet_t, unet_tr = count(self.unet)
        adp_t, adp_tr = count(self.sam2_adapter)
        ref_t, ref_tr = count(self.refiner)
        return {
            "unet_total": unet_t, "unet_trainable": unet_tr,
            "adapter_total": adp_t, "adapter_trainable": adp_tr,
            "refiner_total": ref_t, "refiner_trainable": ref_tr,
            "total": unet_t + adp_t + ref_t,
            "total_trainable": unet_tr + adp_tr + ref_tr,
        }
