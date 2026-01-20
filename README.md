# VAE Multidomain Image Generation

## Overview
This project explores **image generation and representation learning** using a **Variational Autoencoder (VAE)**.  
The primary focus is not only on generating images, but on understanding **how latent spaces are learned** and how they can be **controlled and interpreted**.

The project emphasizes clarity, explainability, and research-aligned experimentation over purely aesthetic results.

---

## Key Objectives
- Build a CNN-based Variational Autoencoder for image generation
- Analyze the structure and behavior of the learned latent space
- Demonstrate controlled image generation via latent traversal
- Study trade-offs between reconstruction quality and latent regularization

---

## Datasets
The project uses standard benchmark datasets commonly employed in generative modeling research:

- **MNIST** – handwritten digits
- **FashionMNIST** – grayscale clothing images

All datasets are 28×28 grayscale, making them suitable for controlled experimentation and latent space analysis.

---

## Model Architecture
The core model is a **Convolutional Variational Autoencoder** consisting of:

- **Encoder**
  - Convolutional layers for feature extraction
  - Outputs mean (μ) and log-variance (logσ²)

- **Latent Space**
  - 32-dimensional continuous representation
  - Enables smooth interpolation and sampling

- **Decoder**
  - Reconstructs images from latent vectors
  - Generates new images via sampling

---

## Training Objective
The model is trained using a composite loss function:


- **Reconstruction Loss** ensures fidelity to input images
- **KL Divergence** enforces a structured, continuous latent space
- The β parameter controls the trade-off between:
  - visual clarity
  - latent space regularization

---

## Latent Space Analysis
A key component of this project is **latent traversal**, where:

- One latent dimension is varied at a time
- All other dimensions are held constant
- Resulting image changes are observed

This analysis demonstrates that:
- The latent space is continuous
- Individual latent dimensions encode meaningful visual factors
- Changes are smooth and interpretable, not random

---

## Image Generation Results
- The model is capable of generating new images by sampling from the latent space
- Generated images are slightly blurred, which is a known characteristic of VAEs
- This behavior arises from:
  - low-resolution datasets
  - pixel-wise reconstruction objectives
  - probabilistic decoding

Images may be upscaled for visualization purposes only; generation occurs at the original resolution.

---

## Project Status
Current progress includes:
- CNN-based VAE implemented and trained
- Latent space traversal and analysis completed
- Image generation validated
- Results documented with clear limitations and observations

---

## Research Significance
This project demonstrates:
- Effective unsupervised representation learning
- Interpretable latent space structure
- Controlled image generation
- Awareness of fundamental trade-offs in generative modeling

These aspects align closely with research themes in **representation learning** and **unsupervised deep learning**.

---

## Future Work
Potential extensions include:
- CNN-based VAE with classification head
- Conditional Variational Autoencoders
- Higher-resolution datasets (e.g., CelebA)
- Comparative studies with GANs or diffusion models

---

## Reproducibility Notes
- Datasets and trained model weights are excluded from version control
- The repository contains only source code and documentation
- Experiments can be reproduced by retraining locally

---

## Author
**Ankita**  
Computer Science | Machine Learning & Representation Learning

