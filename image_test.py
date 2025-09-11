import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_vae_mnist import VAE  # your simple VAE code
DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
# -----------------------------
# Parameters
# -----------------------------
latent_dim = 32
batch_size = 256
population_size = 100

# -----------------------------
# Load MNIST
# -----------------------------
transform = transforms.ToTensor()
ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

# -----------------------------
# Load trained VAE
# -----------------------------
vae = VAE(latent_dim)
vae.load_state_dict(torch.load(f"vae_artifacts_{latent_dim}/vae.pth", map_location=DEVICE))
vae.eval()

# -----------------------------
# Collect latent vectors (mu)
# -----------------------------
z_mu_list = []
with torch.no_grad():
    for x, _ in dl:
        mu, _ = vae.encode(x)
        z_mu_list.append(mu)
z_mu_all = torch.cat(z_mu_list, dim=0).detach().cpu().numpy()  # shape [num_samples, latent_dim]

# -----------------------------
# Plot histogram per latent dimension
# -----------------------------
num_cols = 4
num_rows = (latent_dim + num_cols - 1) // num_cols

plt.figure(figsize=(12, 3*num_rows))
for i in range(latent_dim):
    plt.subplot(num_rows, num_cols, i+1)
    plt.hist(z_mu_all[:, i], bins=50, density=True)
    plt.title(f"Latent dim {i}")
    plt.axvline(-3, color='r', linestyle='--')
    plt.axvline(3, color='r', linestyle='--')
plt.tight_layout()
plt.savefig("test_latent_encoding.pdf")
plt.show()

# -----------------------------
# Biased initialization from reference image
# -----------------------------
# Choose a reference image
ref_img, _ = ds[0]  # can choose any image
ref_img = ref_img.unsqueeze(0)  # add batch dimension

with torch.no_grad():
    z_ref, _ = vae.encode(ref_img)  # [1, latent_dim]
    
# Generate population by perturbing z_ref
sigma = 0.2  # adjust based on histogram
population = z_ref + sigma * torch.randn(population_size, latent_dim)

# Optionally clip to ±3
population = torch.clamp(population, -3, 3)

print("Generated initial population shape:", population.shape)
