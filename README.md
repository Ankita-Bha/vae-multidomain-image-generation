# VAE Multidomain Image Generation

## Overview
This project explores **image generation and representation learning** using a **Variational Autoencoder (VAE)**, with a strong emphasis on **latent space interpretability and controlled generation** rather than purely visual realism.

In addition to generating images, the project focuses on understanding **how latent representations are learned**, how they can be **manipulated through traversal**, and how generated samples can be **semantically validated** using an independent classifier.

An interactive **Streamlit dashboard** is provided to visualize and analyze these behaviors in real time.

---

## Key Objectives
- Build a CNN-based Variational Autoencoder for grayscale image generation  
- Analyze the structure and smoothness of the learned latent space  
- Enable controlled image generation via latent traversal  
- Validate generated images using a CNN-based classifier  
- Provide explainability through visualization and Grad-CAM  
- Present quantitative evaluation through training and validation plots  

---

## Datasets
The project uses standard benchmark datasets commonly employed in generative modeling research:

- **MNIST** – handwritten digits  
- **FashionMNIST** – grayscale clothing images  

All datasets are:
- 28×28 grayscale  
- low-resolution by design  
- suitable for controlled experimentation and latent space analysis  

---

## Model Architecture

### Variational Autoencoder (VAE)
The core generative model is a **Convolutional Variational Autoencoder** consisting of:

- **Encoder**
  - Convolutional layers for feature extraction  
  - Outputs latent mean (μ) and log-variance (logσ²)  

- **Latent Space**
  - 32-dimensional continuous representation  
  - Enables smooth interpolation and controlled sampling  

- **Decoder**
  - Reconstructs images from latent vectors  
  - Generates new samples via latent sampling  

---

### CNN Classifier (Semantic Validation)
A **ResNet-18–based CNN** is trained separately on the same datasets and is used to:

- classify VAE-generated images  
- measure semantic consistency  
- validate whether generated samples retain class-specific structure  

The classifier is **not involved in VAE training**, ensuring independent validation.

---

## Training Objective
The VAE is trained using the standard Evidence Lower Bound (ELBO):

- **Reconstruction Loss** ensures fidelity to input images  
- **KL Divergence** enforces a structured, continuous latent space  

The balance between these terms introduces a known trade-off between:
- visual sharpness  
- latent space regularization and interpretability  

---

## Latent Space Analysis
A key component of this project is **latent space exploration**, implemented interactively via Streamlit:

- One or more latent dimensions can be varied using sliders  
- Remaining dimensions are held constant  
- Resulting image changes are observed in real time  

This analysis demonstrates that:
- the latent space is continuous  
- changes are smooth rather than abrupt  
- different dimensions encode meaningful visual factors  

---

## Class-Anchored Generation
In addition to free latent exploration, the project supports **class-anchored sampling**:

- Latent vectors are sampled from class-specific regions  
- Controlled noise is added to introduce variation  
- Generated images remain human-recognisable  

This mode highlights how class information is implicitly embedded within the latent space.

---

## Explainability with Grad-CAM
To enhance interpretability, the project applies **Grad-CAM** to the CNN classifier:

- Heatmaps highlight image regions influencing predictions  
- Provides visual justification for CNN decisions on generated images  

Grad-CAM explains **why the CNN predicts a class**, not how the VAE generates the image.

---

## Evaluation and Results

### Qualitative Results
- Generated images are recognisable but slightly blurred  
- This behavior is expected in VAEs due to:
  - probabilistic decoding  
  - pixel-wise reconstruction objectives  
  - low-resolution datasets  

### Quantitative Results
The Streamlit dashboard includes:
- **VAE training loss curves**
  - reconstruction loss  
  - KL divergence  
- **CNN accuracy on VAE-generated images**

These results demonstrate:
- stable VAE training  
- structured latent space learning  
- meaningful semantic retention in generated samples  

---

## Interactive Dashboard
A Streamlit application is included to demonstrate:

- latent traversal  
- class-anchored image generation  
- CNN predictions with confidence scores  
- Grad-CAM visual explanations  
- evaluation plots  

The dashboard allows both technical and non-technical users to understand the model’s behavior.

---

## Project Status
Completed components include:
- CNN-based VAE implementation and training  
- Latent space traversal and analysis  
- Class-anchored image generation  
- CNN-based semantic validation  
- Grad-CAM explainability  
- Interactive Streamlit dashboard  
- Evaluation plots and documentation  

The project is considered **feature-complete**.

---

## Research Significance
This project demonstrates:
- effective unsupervised representation learning  
- interpretable latent space structure  
- controlled generative modeling  
- principled evaluation of generative outputs  

The focus on explainability and limitations aligns with research themes in **representation learning** and **unsupervised deep learning**.

---

## Future Work
Potential extensions include:
- Conditional Variational Autoencoders (CVAE)  
- Joint VAE–classifier training  
- Higher-resolution datasets (e.g., CelebA)  
- Comparative analysis with GANs or diffusion models  

---

## Reproducibility Notes
- Raw datasets and trained model weights are excluded from version control  
- The repository contains source code, notebooks, static outputs, and documentation  
- Experiments can be reproduced by retraining locally  

---

## Author
**Ankita**  
Computer Science | Machine Learning | Representation Learning
