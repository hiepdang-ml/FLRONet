# FLRONet: Deep Operator Learning for Flow Field Reconstruction from Sparse Sensors

Official implementation of **FLRONet**, a spatio-temporal deep operator network for reconstructing high-fidelity fluid flow fields from sparse sensor measurements.

This repository accompanies the paper:

> **Hiep Vo Dang, Phong C. H. Nguyen**  
> *Deep Operator Learning for High-Fidelity Fluid Flow Field Reconstruction From Sparse Sensor Measurements*  
> Journal of Computing and Information Science in Engineering, 2026  
> DOI: https://doi.org/10.1115/1.4070332

---

## 🔥 Visual Example
![Visual Example](assets/merged.gif)

---

## Overview

Reconstructing full fluid flow fields from sparse sensor measurements is a severely ill-posed inverse problem. FLRONet addresses this challenge by learning the **inverse measurement operator in function space**, enabling discretization-independent reconstruction across both spatial and temporal domains.

FLRONet integrates:
- Voronoi tessellation for sparse sensor embedding  
- Fourier Neural Operators (FNO) for spatial invariance  
- DeepONet-style branch–trunk architecture for continuous temporal reconstruction  

<p align="center">
  <img src="assets/Fig1.png" alt="Architecture" width="100%">
</p>

---

## Key Ideas & Contributions

- Operator-learning formulation of sparse-sensor flow reconstruction  
- Discretization independence in **both space and time**  
- Zero-shot spatial super-resolution (no retraining)  
- Continuous temporal interpolation at arbitrary time resolution  
- Robustness to missing and noisy sensors  
- Faster inference than 3D FNO baselines

---

## Method at a Glance

FLRONet decomposes the inverse reconstruction operator into:
- A **spatial branch network** (FNO-based) that maps sparse sensor snapshots to a latent field representation
- A **temporal trunk network** that maps continuous query times to reconstruction weights
- A dot-product fusion yielding the reconstructed flow field as a continuous function of space and time

<p align="center">
  <img src="assets/Fig2.png" alt="Architecture" width="100%">
</p>

---

## Zero-Shot Super-Resolution

### Spatial Super-Resolution

Example: ![](assets/gifs/spatial_super_resolution.gif)

FLRONet is trained on a coarse grid and evaluated on significantly finer spatial resolutions without retraining.

---

### Temporal Super-Resolution

<!-- GIF: reconstruction at intermediate continuous time steps -->
<!-- Example: ![](assets/gifs/temporal_super_resolution.gif) -->

FLRONet reconstructs flow fields at arbitrary continuous time points within the observation window, even when no sensor data exist at those times.

---

## Robustness to Real-World Sensor Failures

### Missing Sensors

<!-- GIF: progressive sensor dropout with stable reconstruction -->
<!-- Example: ![](assets/gifs/sensor_dropout.gif) -->

### Noisy Sensors

<!-- OPTIONAL GIF or static figure -->
<!-- Example: ![](assets/gifs/noise_robustness.gif) -->

---

## Repository Organization (Placeholder)

This repository is organized into logical components for **data handling**, **model definition**, **training**, and **inference**.

<!-- PLACEHOLDER: Brief bullet descriptions instead of a tree -->
- `data/` — datasets, preprocessing, and metadata  
- `models/` — FLRONet, FNO blocks, DeepONet components  
- `embeddings/` — Voronoi and temporal embedding layers  
- `training/` — dataset loaders, loss functions, training loops  
- `inference/` — reconstruction and super-resolution scripts  
- `configs/` — experiment and model configuration files  
- `scripts/` — entry points for training and evaluation  
- `results/` — saved figures, metrics, and visualizations  

---

## Installation

### Environment Setup

```bash
conda env create -f environment.yml
conda activate flronet
