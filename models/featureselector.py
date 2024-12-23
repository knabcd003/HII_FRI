import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision.transforms import functional as F
from astropy.io import fits
import numpy as np
import os
import random
import math


# Download with progress bar suppressed
with torch.no_grad():
    model = resnet50(pretrained=True)
print("Download complete!")

# Class to load images from dataset
class GalacticImageDataset(Dataset):
    def __init__(self, data_dir, scale_factor=0.25, target_size=224, transform=None):
        """
        Dataset for pre-training on scaled galactic images
        Args:
            data_dir: Directory containing sets of 4-band FITS images
            scale_factor: Factor to reduce original image size (0.25 = quarter size)
            target_size: Final size for ResNet input (default 224 for ResNet50)
            transform: Optional additional transforms
        """
        self.data_dir = data_dir
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.transform = transform
        self.image_sets = self._find_image_sets()
        
    def _find_image_sets(self):
        """Find all complete sets of 4-band images"""
        image_sets = []
        for region_dir in os.listdir(self.data_dir):
            full_path = os.path.join(self.data_dir, region_dir)
            if not os.path.isdir(full_path):
                continue
                
            fits_files = sorted([f for f in os.listdir(full_path) if f.endswith('.fits')])
            if len(fits_files) == 4:
                image_sets.append([os.path.join(full_path, f) for f in fits_files])
        return image_sets
    
    def _process_fits_image(self, fits_path):
        """Load, scale, and normalize FITS data"""
        with fits.open(fits_path) as hdul:
            # Get original data
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            # Handle invalid values
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # First scale down the image
            h, w = data.shape
            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
            
            # Convert to tensor for resize operation
            tensor_data = torch.from_numpy(data).float().unsqueeze(0)
            scaled_data = F.resize(tensor_data, [new_h, new_w], 
                                    interpolation=F.InterpolationMode.BILINEAR).squeeze(0)
            
            # Robust normalization per image
            scaled_data = scaled_data.numpy()
            non_zero_mask = scaled_data != 0
            if non_zero_mask.any():
                median = np.median(scaled_data[non_zero_mask])
                mad = np.median(np.abs(scaled_data[non_zero_mask] - median))
                if mad > 0:
                    normalized = (scaled_data - median) / (mad * 1.4826)
                else:
                    normalized = scaled_data - median
            else:
                normalized = scaled_data
            
            # Clip outliers
            normalized = np.clip(normalized, -5, 5)
            
            return normalized
    
    def _extract_center_or_random(self, image, is_training=True):
        """Extract center patch or random patch to match target size"""
        h, w = image.shape
        
        if h < self.target_size or w < self.target_size:
            # Pad if image is too small
            pad_h = max(0, self.target_size - h)
            pad_w = max(0, self.target_size - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
            h, w = image.shape
        
        if is_training:
            # Random crop during training
            top = random.randint(0, h - self.target_size)
            left = random.randint(0, w - self.target_size)
        else:
            # Center crop during validation
            top = (h - self.target_size) // 2
            left = (w - self.target_size) // 2
            
        patch = image[top:top+self.target_size, left:left+self.target_size]
        return patch
    
    def __len__(self):
        return len(self.image_sets)
    
    def __getitem__(self, idx):
        image_paths = self.image_sets[idx]
        
        # Process all 4 bands
        bands = []
        for path in image_paths:
            # Load and scale the image
            scaled_data = self._process_fits_image(path)
            
            # Extract appropriate patch
            patch = self._extract_center_or_random(scaled_data)
            bands.append(patch)
        
        # Stack only the first three bands and convert to tensor
        stacked_image = np.stack(bands[:3], axis=0)
        image_tensor = torch.from_numpy(stacked_image).float()
        
        if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
            print(f"Warning: Invalid values in image tensor for index {idx}")
            image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize the stacked tensor as a whole
        mean = image_tensor.mean()
        std = image_tensor.std()
        if std > 0:
            image_tensor = (image_tensor - mean) / std
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, image_tensor  # Return same tensor for self-supervised learning

# class which defines ResNet50 which will be fine tuned
class ModifiedResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        # Start with standard ResNet50
        self.resnet = model
        
        # Store the original conv1 weights - no need to modify since it's already 3 channels
        self.resnet.conv1 = self.resnet.conv1  # Keep original conv1 layer (3 channels)
        self.resnet.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        
        # Add decoder for self-supervised learning
        self.decoder = nn.ModuleDict({
            'upconv1': nn.Sequential(
                nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
                nn.GroupNorm(32, 1024),  
                nn.LeakyReLU(0.2, inplace=True)
            ),
            'upconv2': nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            'upconv3': nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            'upconv4': nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            'final': nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Changed output to 3 channels
                nn.InstanceNorm2d(3),  # Updated to 3 channels
                nn.Hardtanh()
            )
        })
        
    def forward(self, x):
        # Normalize each channel separately
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-5
        x = (x - mean) / std
        x = x.view(b, c, h, w)
        x = torch.clamp(x, -5, 5)
        
        # ResNet forward pass with debugging
        x = self.resnet.conv1(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # Safety check after conv1
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x) 
        x = self.resnet.layer4(x)
        
        # Decoder forward pass with debugging
        for name, layer in self.decoder.items():
            x = layer(x)
            x = torch.clamp(x, -10, 10)
        
        return x

def get_layer_groups(model):
    """Get groups of layers for gradual unfreezing"""
    return [
        model.resnet.conv1,
        model.resnet.bn1,
        model.resnet.layer1,
        model.resnet.layer2,
        model.resnet.layer3,
        model.resnet.layer4,
        model.decoder
    ]

# Format data for fine tuning
def create_dataloaders(data_dir, sample_size=None, batch_size=16, scale_factor=0.25):
    dataset = GalacticImageDataset(
        data_dir=data_dir,
        scale_factor=scale_factor,
        target_size=224
    )
    
    if sample_size and sample_size < len(dataset):
        indices = torch.randperm(len(dataset))[:sample_size]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Modified DataLoader settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0, 
        pin_memory=False,  
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=False  
    )
    
    return train_loader, val_loader

#Fine tuning process through gradual unfreezing
def train_with_gradual_unfreeze(model, train_loader, val_loader, num_epochs=30, device='cuda'):
    """Training with gradual unfreezing of layers"""
    criterion = nn.MSELoss()
    grad_clip = 0.5  # Reduced gradient clipping
    layer_groups = get_layer_groups(model)
    num_groups = len(layer_groups)
    epochs_per_group = num_epochs // num_groups
    
    # Initially freeze all layers except decoder
    for group in layer_groups[:-1]:
        for param in group.parameters():
            param.requires_grad = False
    
    best_loss = float('inf')
    best_state = None
    
    def check_grad_norm(model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    for epoch in range(num_epochs):
        # Unfreeze next group if it's time
        current_group = epoch // epochs_per_group
        if current_group < len(layer_groups) and epoch % epochs_per_group == 0:
            print(f"\nUnfreezing group {current_group}")
            for param in layer_groups[current_group].parameters():
                param.requires_grad = True
        
        # Smaller learning rate and separate decoder parameters
        optimizer = optim.AdamW([
            {'params': [p for g in layer_groups[:-1] for p in g.parameters() if p.requires_grad], 
             'lr': 1e-5, 'weight_decay': 0.01},
            {'params': layer_groups[-1].parameters(),  # decoder parameters
             'lr': 1e-4, 'weight_decay': 0.001}
        ])
        
        # Training phase
        model.train()
        train_loss = 0
        valid_batch_count = 0
        
        for batch_idx, (data, target) in enumerate((train_loader)):
            # Skip batches with invalid values
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"Skipping batch {batch_idx} due to invalid input values")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass with debug info
            output = model(data)
            if output is None:  # NaN detected in forward pass
                print("NaN detected in forward pass ,skipping batch")
                continue
                
            # Scale target to match output range (-1 to 1)
            target_min = target.min()
            target_max = target.max()
            if target_max - target_min == 0:
                print(f"Skipping batch {batch_idx} due to constant target values")
                continue
                
            target_scaled = 2 * (target - target_min) / (target_max - target_min) - 1
            
            try:
                loss = criterion(output, target_scaled)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Skipping batch {batch_idx} due to invalid loss value")
                    continue
                    
                loss.backward()
                
                # Check gradient norms
                grad_norm = check_grad_norm(model)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                grad_norm = check_grad_norm(model)
                
                optimizer.step()
                train_loss += loss.item()
                valid_batch_count += 1
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        if valid_batch_count > 0:
            train_loss /= valid_batch_count
            print(f'Epoch: {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.6f}')
            
            if train_loss < best_loss:
                best_loss = train_loss
                best_state = model.state_dict()
        else:
            print(f"Epoch {epoch+1} had no valid batches")
    
    return best_state

if __name__ == "__main__":
    # Set device
    device = torch.device('cpu')
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        data_dir = "/Volumes/NODE NAME/feature_selector/organized", #CHANGE TO CORRECT PATH
        batch_size=4,
        scale_factor=1.00 #Full resolution
    )
    
    # Create model
    model = ModifiedResNet50().to(device)
    
    # Train with gradual unfreezing
    best_state = train_with_gradual_unfreeze(model, train_loader, val_loader)
    
    # Save the pre-trained backbone
    torch.save(best_state, 'pretrained_backbone.pth')