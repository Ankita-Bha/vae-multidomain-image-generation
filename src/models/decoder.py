import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # 7 -> 7
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 7 -> 14
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 14 -> 28
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        x = self.deconv(x)
        return x
