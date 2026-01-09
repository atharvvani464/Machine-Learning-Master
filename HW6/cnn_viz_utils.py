# cnn_viz_utils.py
# Helper functions for HW6 – CNN Visualization and Interpretability

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms, datasets
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from PIL import Image
import numpy as np, matplotlib.pyplot as plt, random, os
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# 1. Image preprocessing
def preprocess_torchvision(img_path):
    """Load and preprocess an image using ImageNet normalization."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return preprocess(img).unsqueeze(0)

# 2. Build VGG partial model and reconstruct via DeconvNet
class DConv2d(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.deconv = nn.ConvTranspose2d(
            conv.out_channels, conv.in_channels,
            conv.kernel_size, stride=conv.stride, padding=conv.padding)
        self.deconv.weight.data = conv.weight.data
        self.deconv.bias.data.zero_()
    def forward(self, x): return self.conv(x)
    def backward_pass(self, x): return self.deconv(x)

class DReLU(nn.Module):
    def __init__(self): super().__init__(); self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(x)
    def backward_pass(self, x): return self.relu(x)

class DPooling(nn.Module):
    def __init__(self, k, stride=None, pad=0):
        super().__init__()
        self.pool = nn.MaxPool2d(k, stride, pad, return_indices=True)
        self.unpool = nn.MaxUnpool2d(k, stride, pad)
    def forward(self, x):
        out, self.indices = self.pool(x); self.size = x.size()
        return out
    def backward_pass(self, x): return self.unpool(x, self.indices, output_size=self.size)

def build_vgg_deconv(layer_name='features.14'):
    """Build VGG16 model truncated at the specified layer for visualization."""
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    modules = []
    full_layers = list(vgg.features._modules.items())
    for name, layer in full_layers:
        if isinstance(layer, nn.Conv2d):
            modules.append(DConv2d(layer)); modules.append(DReLU())
        elif isinstance(layer, nn.MaxPool2d):
            modules.append(DPooling(layer.kernel_size, layer.stride, layer.padding))
        else:
            modules.append(layer)
        if name == layer_name.split('.')[-1]:
            break
    return nn.ModuleList(modules)

def visualize_torch(model_layers, img_tensor, feature_idx, mode='all'):
    """Forward -> isolate channel -> backward (deconv) for visualization."""
    activations, x = [], img_tensor
    for layer in model_layers:
        x = layer(x); activations.append(x)
    target_act = activations[-1].clone()
    fmap = target_act[:, feature_idx:feature_idx+1, :, :]
    if mode == 'max':
        mask = fmap == fmap.max(); fmap = fmap * mask
    out = torch.zeros_like(target_act)
    out[:, feature_idx:feature_idx+1, :, :] = fmap
    x = out
    for layer in reversed(model_layers):
        if hasattr(layer, 'backward_pass'):
            x = layer.backward_pass(x)
    return x

# 3. TinyImageNet data loader utility
def get_tinyimagenet_loaders(data_dir='./tiny-imagenet-200', fraction=0.1):
    """
    Return DataLoaders for TinyImageNet subset for quick experiments.
    Keeps only a small random fraction of the training and validation data.
    """
    transform = transforms.Compose([
        transforms.Resize(64, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Training set
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
    
    # Validation set
    val_annos = {}
    with open(os.path.join(data_dir, 'val', 'val_annotations.txt')) as f:
        for line in f:
            parts = line.strip().split('\t')
            val_annos[parts[0]] = parts[1]
    class_to_idx = train_data.class_to_idx

    from torch.utils.data import Dataset
    class TinyVal(Dataset):
        def __init__(self, root, val_annos, transform):
            self.root, self.val_annos, self.transform = root, val_annos, transform
            self.items = list(val_annos.items())
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            img_name, wnid = self.items[idx]
            img = Image.open(os.path.join(self.root, img_name)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            label = class_to_idx[wnid]
            return img, label

    val_data = TinyVal(os.path.join(data_dir, 'val/images'), val_annos, transform)

    # Random subsampling to save time
    train_indices = random.sample(range(len(train_data)), int(len(train_data)*fraction))
    val_indices = random.sample(range(len(val_data)), int(len(val_data)*fraction))
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=128)
    return train_loader, val_loader

class TinyCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))

def train_tinycnn(data_dir, train_fraction=0.1, epochs=5, lr=1e-3, save_path=None, device=None):
    """
    Train TinyCNN on a subset of TinyImageNet.
    
    Args:
        data_dir (str): Path to TinyImageNet dataset.
        train_fraction (float): Fraction of training data to use.
        epochs (int): Number of epochs.
        lr (float): Learning rate for Adam optimizer.
        save_path (str): Optional path to save trained model.
        device (str): 'cuda' or 'cpu'. Defaults to CUDA if available.
    
    Returns:
        model (TinyCNN): Trained TinyCNN model.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader, val_loader = get_tinyimagenet_loaders(fraction=train_fraction, data_dir=data_dir)

    # Initialize model, loss, optimizer
    model = TinyCNN(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

    # Save model if path provided
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Saved trained TinyCNN to {save_path}")

    return model
