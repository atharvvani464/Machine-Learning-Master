# Copilot Instructions for ML Coursework

## Overview
This workspace contains Jupyter notebooks for a machine learning course, focusing on computer vision (CNNs, visualization) and natural language processing (sentiment analysis) using PyTorch. Assignments are modular, with each HW in separate notebooks.

## Key Technologies
- **PyTorch**: Core deep learning framework for models and training
- **torchvision**: Image preprocessing and datasets
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Additional ML utilities
- **Jupyter**: Interactive experimentation environment

## File Structure
- `HW*/`: Homework notebooks (e.g., `HW6/homework_6.ipynb` for CNN interpretability)
- `data/MNIST/`: Raw MNIST dataset in binary format
- `HW6/cnn_viz_utils.py`: Custom utilities for CNN visualization (deconvolution, feature maps)
- `HW6/tiny-imagenet-200/`: Tiny ImageNet dataset
- `HW6/IMDB_500.csv`: Sentiment analysis dataset
- `HW6/hw6env/`: Virtual environment with all dependencies installed

## Development Workflow
1. Activate venv: `cd HW6 && source hw6env/bin/activate`
2. Launch Jupyter: `jupyter notebook`
3. Run cells sequentially; notebooks include data loading, model training, and evaluation
4. For visualization, use functions from `cnn_viz_utils.py` (e.g., `preprocess_torchvision`)

## Coding Patterns
- **Imports**: `import torch, torch.nn as nn, torch.nn.functional as F` followed by torchvision and matplotlib
- **Custom Layers**: Subclass `nn.Module` for deconvolution (see `DConv2d` in `cnn_viz_utils.py`)
- **Data Loading**: Use torchvision datasets or custom binary readers for MNIST
- **Visualization**: Matplotlib for feature maps, saliency maps; normalize images with ImageNet stats
- **Model Saving**: `.pth` files for checkpoints (e.g., `tinycnn_demo.pth`)

## Examples
- Deconvolution for feature visualization: `DConv2d` class reconstructs activations
- Image preprocessing: `transforms.Compose([Resize, ToTensor, Normalize])`
- Sentiment data: CSV with 'review' and 'sentiment' columns

Reference: `HW6/cnn_viz_utils.py` for visualization patterns, `HW6/homework_6.ipynb` for CNN workflows.