# ============================================================
# Image Generator using trained VAE
# Dataset: MNIST / FashionMNIST / EMNIST
#
# Usage:
#   python src/generate.py --dataset fashion --num_samples 16
# ============================================================

import argparse
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Fix Python path so `src` is importable when running the script
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.vae import ConvVAE


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load trained model
    model = ConvVAE(latent_dim=32).to(device)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "grayscale" / f"vae_{args.dataset}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Sample latent space
    z = torch.randn(args.num_samples, 32).to(device)
    with torch.no_grad():
        samples = model.decoder(z)

    # De-normalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2

    # Output directory
    out_dir = PROJECT_ROOT / "outputs" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save images
    for i in range(args.num_samples):
        plt.imshow(samples[i, 0].cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(
            out_dir / f"sample_{i}.png",
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close()

    print(f"Generated {args.num_samples} images in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Image Generator")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fashion",
        choices=["mnist", "fashion", "emnist"],
        help="Dataset to generate images from"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of images to generate"
    )

    args = parser.parse_args()
    main(args)
