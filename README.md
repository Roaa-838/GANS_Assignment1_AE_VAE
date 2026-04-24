# Representation Learning with Autoencoders
**DSAI 490 — Assignment 1**

##  Project Overview
This repository contains the implementation, training, and evaluation of a standard **Autoencoder (AE)** and a **Variational Autoencoder (VAE)**. The models are trained on the **Medical MNIST** dataset to reconstruct grayscale medical images ($64 \times 64$) across six different modalities. 

This project explores the fundamental trade-off between exact reconstruction fidelity (AE) and generative capability/latent space regularization (VAE).

### Key Features
* **Modular Architecture:** Clean separation of data pipelines, model definitions, and visualization utilities.
* **Optimized Data Pipeline:** Fully utilizes the `tf.data` API with caching and prefetching (`AUTOTUNE`) for high-performance, out-of-memory image streaming.
* **Probabilistic Modeling:** VAE implementation includes the reparameterization trick and ELBO loss (Reconstruction + KL Divergence).
* **Latent Space Analysis:** Includes t-SNE and PCA visualizations of the 64-dimensional latent space.
* **Generative Sampling:** Demonstrates the VAE's ability to generate novel medical images by sampling from a standard normal prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

---

##  Repository Structure

```text
DSAI490-Assignment1/
│
├── GANS_A1_AE_VAE.ipynb       # notebook colab
├── dataset.py                 # tf.data pipeline, normalization, and train/val splitting
├── models.py                  # TensorFlow/Keras class definitions for AE and VAE
├── utils.py                   # Matplotlib/t-SNE visualization functions
├── experiment_notebook.ipynb  # Main Jupyter/Colab notebook for execution
├── README.md                  # Project documentation
└── .gitignore                 # Ignored files (weights, datasets, cache)
```

---

## Dataset
The models are trained on the **Medical MNIST** dataset, comprising 55,128 grayscale images across six classes:
`AbdomenCT`, `BreastMRI`, `CXR`, `ChestCT`, `Hand`, `HeadCT`.

**Note:** The dataset is not included in this repository. To run the code, you must place the dataset in your Google Drive or local storage and update the `DATA_ROOT` variable in the experiment notebook.

---

## How to Run

### Running in Google Colab (Recommended)
1. Upload the `medicalMNIST` dataset to your Google Drive.
2. Upload `dataset.py`, `models.py`, `utils.py`, and `experiment_notebook.ipynb` to your Colab environment or Drive.
3. Open `experiment_notebook.ipynb` in Google Colab.
4. Run the first cell to mount your Google Drive.
5. Execute the remaining cells to train the models and generate visualizations.

### Running Locally
Ensure you have the following dependencies installed:
```bash
pip install tensorflow numpy matplotlib scikit-learn pillow tqdm
```
Update the `DATA_ROOT` variable in the main script to point to your local dataset folder, then execute the notebook.

---

## Key Results

| Metric / Feature | Autoencoder (AE) | Variational Autoencoder (VAE) |
| :--- | :--- | :--- |
| **Latent Space Type** | Deterministic | Probabilistic ($\mu, \sigma$) |
| **Loss Function** | MSE | MSE + KL Divergence |
| **Validation MSE** | **0.00383** | 0.00874 |
| **Final KL Divergence**| N/A | 15.91 |
| **Generative Capability**| No | **Yes** |
| **Latent Continuity** | Discontinuous (Clusters with gaps) | Smooth and continuous |

### Insights
* **Reconstruction:** The standard AE achieves a sharper, lower-error reconstruction (MSE 0.0038) because it dedicates its entire parameter capacity to exact pixel matching.
* **Latent Space:** The VAE trades a slight amount of reconstruction sharpness for a highly structured, regularized latent space. t-SNE projections show the VAE smoothly connects the image classes, preventing the "empty gaps" seen in the AE's latent space.
* **Generation:** Because the VAE's latent space is regularized toward a unit Gaussian, we can successfully generate completely new, plausible medical scans by sampling random vectors from the prior and passing them through the decoder.

