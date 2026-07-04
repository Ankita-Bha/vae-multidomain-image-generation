<div align="center">

# 🧠 VAE Multidomain Image Generation

**A convolutional Variational Autoencoder for multi-dataset image generation, latent space exploration, and CNN-based semantic validation — with an interactive Streamlit dashboard.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logoColor=white)

</div>

---

## 📖 Overview

This project studies **image generation and representation learning** with a CNN-based Variational Autoencoder trained on grayscale benchmark datasets (MNIST, FashionMNIST, EMNIST). The emphasis is on **latent space interpretability and controlled generation** rather than raw visual realism: latent traversal, class-anchored sampling, and semantic validation of generated images using an independently trained ResNet-18 classifier. An interactive **Streamlit dashboard** ties everything together, including Grad-CAM explanations of the classifier's decisions on generated samples.

## ✨ Features

- Convolutional VAE (`ConvVAE`) with a 32-dimensional latent space, plus a sharper low-beta ("SHARP") training variant
- Multi-dataset training pipeline covering MNIST, FashionMNIST, and EMNIST
- Exploratory **VAE-GAN** hybrid training on FashionMNIST using an auxiliary discriminator
- Independent **ResNet-18 CNN classifiers** (1-channel input) trained per dataset for semantic validation of generated images
- Interactive **latent traversal** via Streamlit sliders — vary one latent dimension while holding the rest fixed
- **Class-anchored generation** — sample latent vectors from class-specific regions with controlled noise
- **Grad-CAM heatmaps** explaining classifier predictions on generated images
- Evaluation notebooks: original-vs-reconstruction grids, random latent samples, classifier prediction distributions over 1,000 generated images, and latent progression across training epochs

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3 |
| Deep Learning | PyTorch, torchvision |
| Models | Convolutional VAE, VAE-GAN (discriminator), ResNet-18 classifier |
| Dashboard | Streamlit |
| Explainability | Grad-CAM |
| Visualization | Matplotlib |
| Environment | Jupyter Notebooks |

## 📂 Project Structure

```text
vae-multidomain-image-generation/
├── app.py                                    # Streamlit dashboard: latent explorer + CNN validation + Grad-CAM
├── app1.py                                   # Extended/alternative dashboard variant
├── notebooks/
│   ├── 01_data/01_data_sanity.ipynb          # Dataset & checkpoint sanity checks
│   ├── 02_models/01_grayscale_vae_arch.ipynb # VAE architecture definition
│   ├── 03_training/
│   │   ├── 01_train_grayscale_vae.ipynb      # Train base + SHARP VAE on MNIST/Fashion/EMNIST
│   │   ├── 02_train_fashion_vae_gan.ipynb    # Exploratory VAE-GAN on FashionMNIST
│   │   └── 03_train_classifier.ipynb         # Train ResNet-18 classifiers on real data
│   ├── 04_evaluation/
│   │   ├── 01_generate_samples.ipynb         # Sampling & reconstruction grids
│   │   ├── 02_eval_generated.ipynb           # CNN prediction distribution on generated images
│   │   └── ieee_figures.ipynb                # Paper-style figures (ieee_outputs/)
│   └── 05_latent_progression/06_latent_progression.ipynb  # Decoding a fixed z across epochs
├── src/
│   ├── models/                               # encoder.py, decoder.py, vae.py, discriminator.py
│   ├── training/                             # losses.py, scheduler.py
│   ├── evaluation/                           # reconstruction / interpolation / sampling scripts
│   ├── config/                               # dataset configs
│   └── utils/                                # seeding, device, visualization helpers
├── outputs/                                  # mnist / fashion original-vs-reconstruction grids
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch and torchvision (CPU is sufficient; CUDA used automatically if available)
- Jupyter Notebook / JupyterLab
- Streamlit (for the interactive dashboard)

### Installation

```bash
git clone https://github.com/Ankita-Bha/vae-multidomain-image-generation.git
cd vae-multidomain-image-generation
pip install torch torchvision streamlit matplotlib numpy jupyter
```

> Note: trained checkpoints (`checkpoints/grayscale/*.pt`) and raw datasets are excluded from version control — run the training notebooks first to regenerate them.

### Usage

```bash
# 1. Train the VAEs (MNIST, FashionMNIST, EMNIST)
jupyter notebook notebooks/03_training/01_train_grayscale_vae.ipynb

# 2. Train the validation classifiers
jupyter notebook notebooks/03_training/03_train_classifier.ipynb

# 3. Generate and evaluate samples
jupyter notebook notebooks/04_evaluation/01_generate_samples.ipynb

# 4. Launch the interactive dashboard
streamlit run app.py
```

## 📊 Results

- Training notebooks show stable convergence of the ELBO objective (reconstruction + KL) across all three datasets, for both the base VAE and the low-beta SHARP variant.
- `outputs/` contains original-vs-reconstruction grids for MNIST and FashionMNIST — reconstructions are recognizable with the mild blur characteristic of pixel-wise VAE objectives.
- The evaluation notebook classifies 1,000 randomly generated FashionMNIST samples with the independent ResNet-18 and reports the full prediction distribution, honestly surfacing mode concentration in unconditioned sampling (most samples map to a single class).
- Latent traversal in the dashboard demonstrates a smooth, continuous latent space where individual dimensions encode distinct visual factors.

## 🔮 Future Improvements

- Conditional VAE (CVAE) to directly address mode concentration in free sampling
- Higher-resolution color datasets (e.g., CelebA — a config stub already exists)
- Quantitative generation metrics such as FID/IS alongside classifier-based validation
- Comparison against diffusion-based generators
- Consolidate notebook training logic into the `src/training` package with CLI entry points

## 👤 Author

**Ankita Bhamidimarri** — [@Ankita-Bha](https://github.com/Ankita-Bha)

---

<div align="center">
<sub>⭐ If you found this project useful, consider giving it a star!</sub>
</div>
