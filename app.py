import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import sys

# ----------------------------
# Project path setup
# ----------------------------
current = Path(__file__).resolve()
while not (current / "src").exists():
    current = current.parent

sys.path.append(str(current))
from src.models.vae import ConvVAE

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="VAE Latent Explorer",
    layout="wide"
)

st.title("🧠 VAE Latent Space Explorer")
st.caption("SHARP VAE + CNN Semantic Validation + Grad-CAM")

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Controls")

DATASET = st.sidebar.selectbox("Select Dataset", ["mnist", "fashion"])
MODE = st.sidebar.radio(
    "Generation Mode",
    ["Free Latent (Exploration)", "Class-Anchored (Recognisable)"]
)

latent_dim = 32

# ----------------------------
# Load VAE
# ----------------------------
@st.cache_resource
def load_vae(dataset):
    model = ConvVAE(latent_dim=latent_dim).to(device)
    ckpt = current / "checkpoints" / "grayscale" / f"vae_{dataset}_sharp_64.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model

vae = load_vae(DATASET)

# ----------------------------
# Dataset + latent bank
# ----------------------------
@st.cache_resource
def load_dataset(dataset):
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if dataset == "mnist":
        return datasets.MNIST("data", train=True, download=True, transform=transform)
    else:
        return datasets.FashionMNIST("data", train=True, download=True, transform=transform)

@st.cache_resource
def build_latent_bank(_vae, dataset_name, samples_per_class=200):
    dataset = load_dataset(dataset_name)
    latent_bank = {i: [] for i in range(10)}
    counts = {i: 0 for i in range(10)}

    with torch.no_grad():
        for img, label in dataset:
            if counts[label] >= samples_per_class:
                continue
            img = img.unsqueeze(0).to(device)
            _, mu, _ = _vae(img)
            latent_bank[label].append(mu.squeeze(0).cpu())
            counts[label] += 1
            if all(counts[c] >= samples_per_class for c in counts):
                break
    return latent_bank

latent_bank = build_latent_bank(vae, DATASET)

# ----------------------------
# Latent input
# ----------------------------
if MODE == "Free Latent (Exploration)":
    active_dims = st.sidebar.slider("Latent dimensions", 2, 10, 6)
    z = torch.zeros(1, latent_dim)
    for i in range(active_dims):
        z[0, i] = st.sidebar.slider(f"z[{i}]", -3.0, 3.0, 0.0, 0.1)
    z = z.to(device)
else:
    target_class = st.sidebar.selectbox("Target Class", list(range(10)))
    noise_scale = st.sidebar.slider("Variation (noise strength)", 0.05, 0.6, 0.25)
    mu_anchor = latent_bank[target_class][np.random.randint(len(latent_bank[target_class]))]
    z = (mu_anchor + torch.randn_like(mu_anchor) * noise_scale).unsqueeze(0).to(device)

# ----------------------------
# Decode
# ----------------------------
def decode(z):
    with torch.no_grad():
        h = vae.decoder.fc(z)
        h = h.view(1, 128, 7, 7)
        img = vae.decoder.deconv(h)
    img = (img + 1) / 2
    return F.interpolate(img, size=(64, 64))

img_64 = decode(z)

# ----------------------------
# Load CNN
# ----------------------------
@st.cache_resource
def load_cnn(dataset):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    ckpt = current / "checkpoints" / "grayscale" / f"resnet18_{dataset}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model.to(device)

cnn = load_cnn(DATASET)

img_64.requires_grad_(True)
logits = cnn(img_64)
probs = torch.softmax(logits, dim=1)
pred_class = probs.argmax(dim=1).item()
confidence = probs.max(dim=1).values.item()

# ----------------------------
# Grad-CAM
# ----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        score = self.model(x)[:, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=(64, 64), mode="bilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Latent Space",
    "Latent Traversal",
    "Class-Anchored & CNN",
    "Evaluation"
])

# ----------------------------
# TAB 1 — Latent Space (static grids)
# ----------------------------
with tab1:
    st.markdown("### Original vs Reconstructed Images")
    img_path = current / "outputs" / f"{DATASET}_orig_vs_recon.png"
    if img_path.exists():
        st.image(str(img_path), width=900)
    else:
        st.warning("Reconstruction grid not found. Run evaluation notebook.")

# ----------------------------
# TAB 2 — Latent Traversal
# ----------------------------
with tab2:
    st.markdown("### Latent Traversal")
    st.image(img_64.detach().cpu().numpy().squeeze(), width=256)

# ----------------------------
# TAB 3 — Class-Anchored + CNN
# ----------------------------
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("VAE Generated Image")
        st.image(img_64.detach().cpu().numpy().squeeze(), width=256)

    with col2:
        st.subheader("CNN Prediction")
        st.metric("Predicted Class", pred_class)
        st.progress(float(confidence))
        st.caption(f"Confidence: {confidence:.2%}")

    if st.checkbox("Explain CNN prediction (Grad-CAM)"):
        cam = GradCAM(cnn, cnn.layer4).generate(img_64, pred_class)
        img_np = img_64.detach().cpu().numpy().squeeze()
        overlay = np.clip(
            0.6 * np.stack([img_np]*3, axis=-1) +
            0.4 * np.stack([cam, np.zeros_like(cam), 1-cam], axis=-1),
            0, 1
        )
        st.image(overlay, width=256)

# ----------------------------
# TAB 4 — Evaluation (side by side)
# ----------------------------
with tab4:
    st.markdown("### Model Evaluation")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("VAE Training Loss (Sharp VAE — Real Data)")

        # ── REAL numbers from your training logs ──
        real_loss = {
            "mnist":   [2759755, 1215020, 1113698, 1059621, 1021342,
                         992118,  968967,  951817,  937565,  924583,
                         915095,  905336],
            "fashion": [3020905, 1922835, 1784209, 1708311, 1661062,
                        1623802, 1597358, 1574646, 1555683, 1541268,
                        1527936, 1517140],
        }
        real_recon = {
            "mnist":   [2496538, 941084, 840369, 787309, 750243,
                         722446, 700161, 683818, 670294, 657818,
                         648927, 639623],
            "fashion": [2778740, 1673836, 1535459, 1460079, 1413760,
                        1377022, 1351030, 1328881, 1310511, 1296527,
                        1283353, 1273032],
        }
        real_kl = {
            "mnist":   [5264341, 5478731, 5466592, 5446244, 5421996,
                        5393444, 5376114, 5359982, 5345433, 5335300,
                        5323354, 5314251],
            "fashion": [4843311, 4979980, 4975007, 4964628, 4946046,
                        4935610, 4926549, 4915302, 4903441, 4894822,
                        4891671, 4882150],
        }

        ds_key = "fashion" if DATASET == "fashion" else "mnist"
        epochs_x = list(range(1, len(real_loss[ds_key]) + 1))

        # Normalise to 0-1 scale so both lines are readable on same chart
        recon_vals = np.array(real_recon[ds_key])
        kl_vals    = np.array(real_kl[ds_key])
        recon_norm = recon_vals / recon_vals[0]
        kl_norm    = kl_vals    / kl_vals[-1]

        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.plot(epochs_x, recon_norm, label="Reconstruction Loss (normalised)", color="#3b82f6")
        ax1.plot(epochs_x, kl_norm,    label="KL Divergence (normalised)",       color="#f59e0b")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (normalised to 1.0)")
        ax1.set_title(f"{DATASET.upper()} — Sharp VAE (12 epochs)")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        # Show actual numbers below
        st.caption(
            f"Epoch 1 → Recon: {recon_vals[0]:,.0f} | "
            f"Epoch 12 → Recon: {recon_vals[-1]:,.0f} | "
            f"Reduction: {(1 - recon_vals[-1]/recon_vals[0])*100:.0f}%"
        )

    with col2:
        st.subheader("CNN Accuracy on Real Images (Real Data)")

        # ── REAL numbers from your classifier training ──
        cnn_accuracy = {
            "mnist": {
                "per_epoch": [95.85, 98.25, 98.70, 98.83, 99.00,
                               99.18, 99.22, 99.26],
                "final": 99.26
            },
            "fashion": {
                "per_epoch": [84.47, 88.78, 90.04, 91.16, 92.02,
                               92.56, 93.05, 93.73],
                "final": 93.73
            },
        }

        acc_data = cnn_accuracy[ds_key]
        epochs_cnn = list(range(1, len(acc_data["per_epoch"]) + 1))

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(epochs_cnn, acc_data["per_epoch"],
                 marker="o", color="#10b981", linewidth=2)
        ax2.set_ylim(80, 100)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"{DATASET.upper()} — ResNet-18 Classifier (8 epochs)")
        ax2.grid(True)
        st.pyplot(fig2)

        st.metric(
            label=f"Final CNN Accuracy on {DATASET.upper()}",
            value=f"{acc_data['final']}%"
        )
        st.caption("Trained on real images only. Used to validate generated image quality.")
# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "This dashboard demonstrates latent exploration, class-conditioned generation, "
    "and CNN-based semantic validation. Blurriness is expected due to probabilistic decoding."
)
