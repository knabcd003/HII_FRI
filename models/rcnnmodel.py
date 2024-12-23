import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms
from torchvision.transforms import functional as F
import torch.nn as nn
from torch.nn import GroupNorm
import numpy as np
from astropy.io import fits
import pandas as pd
import os
import time


class HIIRegionDataset(Dataset):
    def __init__(self, data_dir, scale_factor=0.5, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.scale_factor = scale_factor
        self.regions = self._load_dataset_info()
        
        print(f"Found {len(self.regions)} valid regions")
        
    def _load_dataset_info(self):
        regions = []
        for region_dir in os.listdir(self.data_dir):
            full_path = os.path.join(self.data_dir, region_dir)
            if not os.path.isdir(full_path):
                continue
            
            # Get all FITS files
            fits_files = sorted([f for f in os.listdir(full_path) if f.endswith('.fits')])
            csv_files = [f for f in os.listdir(full_path) if f.endswith('.csv')]
            
            # Access first 3 bands
            if len(fits_files) >= 3:
                fits_files = fits_files[:3]  
                
                if len(csv_files) >= 1:
                    csv_files = csv_files[:1]
                    # Validate annotations before adding to dataset
                    csv_path = os.path.join(full_path, csv_files[0])
                    try:
                        scale_factor = self.scale_factor
                        annotations = pd.read_csv(csv_path)
                        # Get original image dimensions from first FITS file
                        with fits.open(os.path.join(full_path, fits_files[0])) as hdul:
                            h, w = hdul[0].data.shape
                        
                        # Scale dimensions
                        scaled_h, scaled_w = int(h * scale_factor), int(w * scale_factor)
                        
                        # Scale annotations
                        scaled_annotations = annotations.copy()
                        scaled_annotations[['xmin', 'ymin', 'xmax', 'ymax']] *= scale_factor
                        
                        # Verify box validity
                        valid = True
                        for _, row in scaled_annotations.iterrows():
                            if (row['xmin'] > row['xmax'] or 
                                row['ymin'] > row['ymax'] or 
                                row['xmax'] > scaled_w or 
                                row['ymax'] > scaled_h):
                                valid = False
                                print(f"Invalid box found in {region_dir}")
                                break
                        
                        if valid:
                            regions.append({
                                'fits_paths': [os.path.join(full_path, f) for f in fits_files],
                                'annotation_path': csv_path
                            })
                    except Exception as e:
                        print(f"Error processing {region_dir}: {str(e)}")
        
        return regions
    
    def __getitem__(self, idx):
        region = self.regions[idx]
        
        # Load and stack FITS data
        image_data = []
        for fits_path in region['fits_paths']:
            with fits.open(fits_path) as hdul:
                data = hdul[0].data
                
                normalized = normalize_fits_data(data)

                h, w = normalized.shape
                new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
    
                tensor_data = torch.from_numpy(normalized).float().unsqueeze(0)
                scaled_data = F.resize(tensor_data, [new_h, new_w], 
                                    interpolation=F.InterpolationMode.BILINEAR).squeeze(0) 
                image_data.append(scaled_data)
        
        stacked_image = torch.stack(image_data, dim=0) 
        
        # Load annotations
        annotations = pd.read_csv(region['annotation_path'])
        scaled_annotations = annotations.copy()
        scaled_annotations[['xmin', 'ymin', 'xmax', 'ymax']] *= self.scale_factor
        
        boxes = torch.tensor(scaled_annotations[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float32)
        labels = torch.ones((len(annotations),), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        if self.transform:
            stacked_image = self.transform(stacked_image)
            
        return stacked_image, target

   

    def __len__(self):
            return len(self.regions)

"""
Helper method to normalize data during dataset construction
"""
def normalize_fits_data(data):
        data = np.array(data, dtype=np.float32, copy=True)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Robust normalization per image
        non_zero_mask = data != 0
        if non_zero_mask.any():
            median = np.median(data[non_zero_mask])
            mad = np.median(np.abs(data[non_zero_mask] - median))
            
            epsilon = 1e-8
            if mad > epsilon:
                normalized = (data - median) / (mad * 1.4826)
            else:
                normalized = data - median
                print("Warning: Very small MAD detected, only centering data")
        else:
            normalized = data
            print("Warning: All-zero image detected")
        
        # Clip outliers (between -5 and 5 sigma)
        normalized = np.clip(normalized, -5, 5)
        
        return normalized

class CustomRPNHead(RPNHead):
    """Custom RPN head with additional features for diffuse regions"""
    def __init__(self, in_channels, num_anchors):
        super().__init__(in_channels, num_anchors)
        
        # Replace BatchNorm with GroupNorm for better brightness handling
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            GroupNorm(8, in_channels), 
            nn.ReLU(inplace=True)
        )
        
        # Additional pathway for diffuse regions
        self.diffuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2),
            GroupNorm(8, in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Separate classification layers for compact and diffuse regions
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.diffuse_cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        
        # Shared bbox regression
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, x):
        # Regular pathway
        regular_features = self.conv(x)
        regular_logits = self.cls_logits(regular_features)
        
        # Diffuse pathway
        diffuse_features = self.diffuse_conv(x)
        diffuse_logits = self.diffuse_cls_logits(diffuse_features)
        
        # Combine logits with learned weights
        combined_logits = regular_logits + diffuse_logits
        
        # Bbox prediction (shared between pathways)
        bbox_reg = self.bbox_pred(regular_features)
        
        return combined_logits, bbox_reg


class CustomRegionProposalNetwork(RegionProposalNetwork):
    """Custom RPN with modified NMS and handling for diffuse regions"""
    def __init__(self, anchor_generator, head, *args, **kwargs):
        super().__init__(anchor_generator, head, *args, **kwargs)
        
        # Modified NMS thresholds for diffuse regions
        self.nms_thresh = 0.7  
        self.nms_thresh_diffuse = 0.8  
        
    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        num_images = proposals.shape[0]
        device = proposals.device
        
        # Modified object selection for diffuse regions
        objectness_prob = torch.sigmoid(objectness)
        
        # Separate thresholds for regular and diffuse regions
        regular_thresh = 0.5
        diffuse_thresh = 0.3  
        
        final_boxes = []
        final_scores = []
        
        for img_idx in range(num_images):
            boxes = proposals[img_idx]
            scores = objectness_prob[img_idx]
            image_shape = image_shapes[img_idx]
            
            # Clip boxes to image
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            
            # Remove small boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-3)
            boxes, scores = boxes[keep], scores[keep]
            
            # Modified NMS strategy
            # First pass: regular NMS
            keep_regular = nms(boxes, scores, self.nms_thresh)
            
            # Second pass: diffuse NMS with higher threshold
            remaining_boxes = boxes[~torch.isin(torch.arange(len(boxes), device=device), keep_regular)]
            remaining_scores = scores[~torch.isin(torch.arange(len(scores), device=device), keep_regular)]
            keep_diffuse = nms(remaining_boxes, remaining_scores, self.nms_thresh_diffuse)
            
            # Combine keeps
            boxes_regular = boxes[keep_regular]
            boxes_diffuse = remaining_boxes[keep_diffuse]
            scores_regular = scores[keep_regular]
            scores_diffuse = remaining_scores[keep_diffuse]
            
            final_boxes.append(torch.cat([boxes_regular, boxes_diffuse]))
            final_scores.append(torch.cat([scores_regular, scores_diffuse]))
        
        return final_boxes, final_scores

def get_model(num_classes=2):
    """
    Create and configure the Faster R-CNN model with a custom pre-trained backbone
    """
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)
    
    backbone_state = torch.load('pretrained_backbone.pth')
    
    backbone_dict = {}
    
    # Mapping dictionary for layer names
    layer_mapping = {
        'resnet.conv1': 'conv1',
        'resnet.bn1': 'bn1',
        'resnet.layer1': 'layer1',
        'resnet.layer2': 'layer2',
        'resnet.layer3': 'layer3',
        'resnet.layer4': 'layer4',
    }
    
    # Get target state dict for shape comparison
    target_state_dict = backbone.body.state_dict()
    

    for old_key, value in backbone_state.items():
        if any(key in old_key for key in layer_mapping.keys()):
            base_layer = next(k for k in layer_mapping.keys() if k in old_key)
            new_key = old_key.replace(base_layer, layer_mapping[base_layer])
            if 'num_batches_tracked' in new_key and new_key not in backbone_state:
                continue
            if new_key in target_state_dict:
                target_shape = target_state_dict[new_key].shape
                if value.shape == target_shape:
                    backbone_dict[new_key] = value
                else:
                    print(f"Shape mismatch for {new_key}: Expected {target_shape}, got {value.shape}")
            else:
                print(f"Skipping {new_key} - not found in target model")
    
    # Load the weights into the backbone
    try:
        for key in target_state_dict:
            if 'num_batches_tracked' in key and key not in backbone_dict:
                backbone_dict[key] = torch.tensor(0)
        
        backbone.body.load_state_dict(backbone_dict, strict=False)
        print("Successfully loaded backbone weights")
    except Exception as e:
        print(f"Error loading backbone weights: {e}")
        
    # Custom anchor generator with scales appropriate for HII regions
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32, 64, 128),) * 5,  
        aspect_ratios=((0.5, 1.0, 2.0, 3.0),) * 5  
    )
    
    # Custom RPN head
    rpn_head = CustomRPNHead(
        in_channels=backbone.out_channels,
        num_anchors=len(anchor_generator.sizes[0]) * len(anchor_generator.aspect_ratios[0])
    )
    
    # Custom RPN
    rpn = CustomRegionProposalNetwork(
        anchor_generator=anchor_generator,
        head=rpn_head,
        fg_iou_thresh=0.7,  
        bg_iou_thresh=0.3,  
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n={'training': 2000, 'testing': 1000},
        post_nms_top_n={'training': 1000, 'testing': 500},
        nms_thresh=0.7
    )

    # Build the custom model
    model = FasterRCNN(
        backbone=backbone,
        rpn=rpn,
        num_classes=num_classes,
        box_detections_per_img=200,  
        box_nms_thresh=0.7,  
        box_score_thresh=0.05,  
        box_fg_iou_thresh=0.7,
        box_bg_iou_thresh=0.3
    )

    return model

def create_data_loaders(data_dir, batch_size=2):
    """
    Create training and validation data loaders
    """
    dataset = HIIRegionDataset(data_dir, scale_factor = 0.5)
    
    # Split into train/validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    return train_loader, val_loader

def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    
    for images, targets in data_loader:
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(losses):
            print('WARNING: non-finite loss, ending training ', loss_dict)
            continue
        
        optimizer.zero_grad()
        losses.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

def train(data_dir, scale_factor=0.25, batch_size=4, num_epochs=10):
    train_loader, val_loader = create_data_loaders(
        data_dir,
        batch_size
    )
    # Initialize model
    model = get_model(num_classes=2) 
    
    # Initialize optimizer with learning rate scheduling
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=2, 
        verbose=True
    )
    
    # Training loop with timing
    print(f"Starting training at {scale_factor*100}% resolution")
    start_time = time.time()

    best_loss = float('inf')
    for epoch in range(num_epochs):
        print("Training epoch " + epoch+1 + "...")
        epoch_loss = train_epoch(model, train_loader, optimizer)
        scheduler.step(epoch_loss)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {elapsed/60:.1f} min")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Training completed in {elapsed/60:.1f} minutes")
    return model

if __name__ == "__main__":
    print("Training") 
    #REPLACE THIS FILE PATH WITH PROPER ONE FOR TRAINING IMAGES
    train("/Volumes/NODE NAME/minitraining", 
      scale_factor=1.00,  # Full resolution
      batch_size=4,       # Batch size of 4
      num_epochs=10)










