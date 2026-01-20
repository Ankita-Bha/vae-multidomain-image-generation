import torch
import torch.nn as nn
from src.models.encoder import ConvEncoder
from src.models.decoder import ConvDecoder


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
