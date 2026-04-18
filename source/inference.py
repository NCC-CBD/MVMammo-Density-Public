"""
Lightweight inference script for MVMammo-Density models.

This script performs inference given:
- Model name (timm/torchvision key), path to pretrained checkpoint,
- Evaluation NCC JSON file, image size, batch size, and related options.

Example usage:
  python inference.py --model-name resnet50 --checkpoint ./checkpoints/resnet50/.../best_model.pt --ncc-json ./some.json --img-size 224 --batch-size 16

Output: predictions.npz (containing y_true and probs) will be saved in the same directory as the checkpoint by default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import MVMammo, MammoDataset

def load_ncc_samples(json_path: str) -> List[dict]:
    """Load NCC evaluation samples from JSON metadata."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [
        s for s in data
        if "BREAST_DENSITY" in s and os.path.exists(s.get("IMAGE_PATH", ""))
    ]
    return samples

def build_loader(
    samples: Sequence[dict],
    img_size: int,
    batch_size: int,
    num_workers: int = 2
) -> DataLoader:
    """Create a DataLoader for evaluation."""
    ds = MammoDataset(list(samples), image_size=img_size, is_train=False)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on the DataLoader and collect true labels and predicted probabilities."""
    model.eval()
    true_labels: List[float] = []
    probabilities: List[np.ndarray] = []
    for img, label in tqdm(loader, desc="Inference"):
        img = img.to(device, non_blocking=True)
        logits, _ = model(img)
        probs = torch.sigmoid(logits)
        probabilities.append(probs.cpu().numpy())
        true_labels.append(label.cpu().numpy().reshape(-1))
    y = np.concatenate(true_labels, axis=0).astype(int)
    p = np.concatenate(probabilities, axis=0)
    return y, p

def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> None:
    """Load checkpoint weights into the model."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight MVMammo-Density Inference")
    p.add_argument("--model-name", required=True, help="timm/torchvision model name (e.g., resnet50, efficientnet_b5_224, etc.)")
    p.add_argument("--checkpoint", required=True, help="Path to pretrained weight (.pt file)")
    p.add_argument("--ncc-json", required=True, help="NCC metadata JSON file (such as filtered_final.json)")
    p.add_argument("--img-size", type=int, required=True, help="Input image size (e.g., 224, 456, etc.)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None, help="cuda:0 or cpu. Default: cuda:0 if available")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output", default=None, help="Path to save predictions.npz. If not specified, saves to checkpoint directory")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Load evaluation samples
    samples = load_ncc_samples(args.ncc_json)
    if not samples:
        print("No evaluation samples found.")
        sys.exit(1)
    print(f"# of evaluation samples: {len(samples)}")

    loader = build_loader(
        samples,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = MVMammo(
        model_name=args.model_name,
        num_classes=4,
        pretrained=False,
    ).to(device)
    load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    y_true, probs = collect_predictions(model, loader, device)
    print(f"y_true shape: {y_true.shape}, probs shape: {probs.shape}")

    out_path = args.output
    if out_path is None:
        # By default, save predictions.npz alongside the checkpoint
        ckpt_dir = os.path.dirname(args.checkpoint)
        out_path = os.path.join(ckpt_dir, "predictions.npz")
    np.savez(
        out_path,
        y_true=y_true,
        probs=probs.astype(np.float32),
    )
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
