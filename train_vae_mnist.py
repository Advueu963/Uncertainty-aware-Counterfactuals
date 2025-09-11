import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils as vutils
from torch.utils.data import DataLoader
import os

# ------------------------------
# Model
# ------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 14 -> 7
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), # 14 -> 28
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()  # outputs in [0,1]
        )

    def encode(self, x):
        h = self.enc(x)
        #print(h.shape)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 64, 7, 7)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        #print(z.shape)
        x_rec = self.decode(z)
        return x_rec, mu, logvar

class DeepVAE(VAE):
    def __init__(self, latent_dim=16):
        super().__init__(latent_dim)
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),   # 28 -> 14
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 14 -> 7
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), # 7 -> 4
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128*4*4, latent_dim)
        self.fc_logvar = nn.Linear(128*4*4, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 128 * 7 * 7)
        self.dec = nn.Sequential(
            # starting spatial size: 7 x 7
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 14 -> 28
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),             # keep 28 -> 28
            nn.Sigmoid()
        )

    def decode(self, z):
        h = self.fc_dec(z)                      # [B, 128*7*7]
        h = h.view(-1, 128, 7, 7)               # [B,128,7,7]
        return self.dec(h)
    
# ------------------------------
# Loss
# ------------------------------
def vae_loss(x, x_rec, mu, logvar, beta=1.0):
    recon = F.binary_cross_entropy(x_rec, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kld, recon, kld


# ------------------------------
# Training loop
# ------------------------------
def train_vae(vae_model, latent_dim=16, epochs=5, batch_size=128, lr=1e-3, beta=1.0, outdir="vae_artifacts", num_workers=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    transform = transforms.ToTensor()
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Model + optimizer
    vae = copy.deepcopy(vae_model).to(device)
    opt = optim.Adam(vae.parameters(), lr=lr)

    os.makedirs(outdir, exist_ok=True)

    for ep in range(1, epochs + 1):
        vae.train()
        total_loss, total_recon, total_kld = 0, 0, 0
        for x, _ in dl:
            x = x.to(device)
            x_rec, mu, logvar = vae(x)
            loss, recon, kld = vae_loss(x, x_rec, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total_recon += recon.item() * x.size(0)
            total_kld += kld.item() * x.size(0)

        n = len(ds)
        print(f"[Epoch {ep}] loss={total_loss/n:.4f}, recon={total_recon/n:.4f}, kld={total_kld/n:.4f}")

        # Save reconstructions
        vae.eval()
        with torch.no_grad():
            sample = torch.randn(64, latent_dim).to(device)
            samples = vae.decode(sample).cpu()
            vutils.save_image(samples, f"{outdir}/sample_ep{ep}.png", nrow=8)

            x, _ = next(iter(dl))
            x = x[:8].to(device)
            x_rec, _, _ = vae(x)
            comp = torch.cat([x.cpu(), x_rec.cpu()])
            vutils.save_image(comp, f"{outdir}/recon_ep{ep}.png", nrow=8)

    torch.save(vae.state_dict(), f"{outdir}/vae.pth")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    latent_dim = 32
    vae_model = VAE(latent_dim=latent_dim)
    train_vae(vae_model, epochs=20,latent_dim=latent_dim, num_workers=6, outdir="vae_artifacts_32")

    deep_vae_model = DeepVAE(latent_dim=latent_dim)
    train_vae(deep_vae_model, epochs=20, latent_dim=latent_dim, num_workers=6, outdir="deep_vae_artifacts_32")
