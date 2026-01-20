import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """
    recon_x : reconstructed image
    x       : original image
    mu      : latent mean
    logvar  : latent log variance
    beta    : KL weight (beta-VAE)
    """

    # Reconstruction loss (MSE works better than BCE for normalized images)
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    # KL Divergence
    kl_div = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = recon_loss + beta * kl_div

    return total_loss, recon_loss, kl_div
