# VAE and Deep Spatial Interpolation for Rock Mass Domaining

This repository contains code for a research project focused on rock mass domaining using Variational Autoencoders (VAE) and a neural-compatible spatial interpolation method. The goal is to generate spatially contiguous domains with consistent rock hardness from Measurements While Drilling (MWD) data.

## 📁 Project Structure

### Core Components

- **`variational_model.py`**  
  Defines various VAE architectures used throughout the project.

- **`correlation_vae.py`**  
  Implements a modified VAE (Correlational VAE) that enforces correlation between the latent representation of the target data and their spatial coordinates.

- **`DeepIDW.py`**  
  Implements a deep learning-based Inverse Distance Weighting (IDW) model. This spatial interpolator is differentiable and compatible with neural networks, allowing integration with Correlational VAE to enforce spatial constraints.

- **`idw_cluster.py`**  
  Combines the Correlational VAE and DeepIDW to learn spatially-aware latent representations, referred to as the *pseudo-BI*, which are used for clustering and domain generation.

- **`kernels.py`**  
  Contains the learnable kernels used to compute the spatial weights for the DeepIDW model.

### Data Simulation and Training

- **`simulate.py`**  
  Generates synthetic drilling data for experimentation and evaluation, in place of confidential real-world data.

- **`train_save.py`**  
  Trains the combined VAE-IDW model and saves the trained model in `.pth` format.

- **`test.py`**  
  Loads a trained model to perform inference and generate spatial domain predictions on the synthetic dataset.

### Utilities and Evaluation

- **`metrics.py`**  
  Defines the evaluation metrics used in the project, such as domain contiguity and hardness consistency.

- **`utils/`**  
  Contains helper functions for data loading, preprocessing, and visualization.

- **`losses/`**  
  Defines custom VAE losses including reconstruction loss and Kullback-Leibler divergence.

### Environment

- **`vae_idw_env`**  
  requirements.txt Contains the required Python environment to run the project.

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone git@github.com:hajlaouiyakin/idw_and_vae_for_rock_mass_domaining.git
   cd idw_and_vae_for_rock_mass_domaining
