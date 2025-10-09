"""
Created on Thu Mar 6 23:41:04 2025

@author: drsaq 

Utility functions for SAMA-UNet, includes loss functions, metrics, and helper functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
import os
import shutil


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation
    """
    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        pred: (B, C, H, W) - logits
        target: (B, H, W) - class indices
        """
        # Convert to one-hot
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten
        pred = pred.contiguous().view(pred.size(0), self.num_classes, -1)
        target_one_hot = target_one_hot.contiguous().view(target_one_hot.size(0), self.num_classes, -1)
        
        # Compute dice
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


def calculate_dice(pred, target, num_classes):
    """
    Calculate Dice Similarity Coefficient for each class
    
    Args:
        pred: (B, H, W) - predicted class indices
        target: (B, H, W) - ground truth class indices
        num_classes: number of classes
    
    Returns:
        dice_per_class: list of dice scores for each class
        mean_dice: mean dice score
    """
    dice_per_class = []
    
    for c in range(1, num_classes):  # Skip background (class 0)
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union == 0:
            dice_per_class.append(1.0)  # Perfect score if class not present
        else:
            dice = (2. * intersection + 1e-5) / (union + 1e-5)
            dice_per_class.append(dice.item())
    
    mean_dice = np.mean(dice_per_class) if dice_per_class else 0.0
    
    return dice_per_class, mean_dice


def calculate_nsd(pred, target, num_classes, tolerance=2.0):
    """
    Calculate Normalized Surface Distance (NSD)
    
    Args:
        pred: (B, H, W) - predicted class indices (numpy array)
        target: (B, H, W) - ground truth class indices (numpy array)
        num_classes: number of classes
        tolerance: tolerance distance in pixels
    
    Returns:
        nsd_per_class: list of NSD scores for each class
        mean_nsd: mean NSD score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    nsd_per_class = []
    
    for c in range(1, num_classes):  # Skip background
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)
        
        # Check if class exists in either prediction or target
        if pred_c.sum() == 0 and target_c.sum() == 0:
            nsd_per_class.append(1.0)
            continue
        elif pred_c.sum() == 0 or target_c.sum() == 0:
            nsd_per_class.append(0.0)
            continue
        
        # Extract boundaries
        pred_boundary = extract_boundary(pred_c)
        target_boundary = extract_boundary(target_c)
        
        # Calculate distances
        dist_pred_to_target = distance_transform_edt(~target_boundary.astype(bool))
        dist_target_to_pred = distance_transform_edt(~pred_boundary.astype(bool))
        
        # Get boundary points
        pred_points = np.where(pred_boundary)
        target_points = np.where(target_boundary)
        
        # Calculate NSD
        pred_distances = dist_pred_to_target[pred_points]
        target_distances = dist_target_to_pred[target_points]
        
        pred_within_tolerance = (pred_distances <= tolerance).sum()
        target_within_tolerance = (target_distances <= tolerance).sum()
        
        nsd = (pred_within_tolerance + target_within_tolerance) / (len(pred_distances) + len(target_distances))
        nsd_per_class.append(nsd)
    
    mean_nsd = np.mean(nsd_per_class) if nsd_per_class else 0.0
    
    return nsd_per_class, mean_nsd


def extract_boundary(mask):
    """
    Extract boundary from binary mask using morphological operations
    
    Args:
        mask: (H, W) binary mask
    
    Returns:
        boundary: (H, W) binary boundary mask
    """
    from scipy.ndimage import binary_erosion
    
    eroded = binary_erosion(mask)
    boundary = mask - eroded
    
    return boundary.astype(np.uint8)


def calculate_metrics(pred, target, num_classes):
    """
    Calculate all evaluation metrics (DSC and NSD)
    
    Args:
        pred: (B, H, W) - predicted class indices
        target: (B, H, W) - ground truth class indices
        num_classes: number of classes
    
    Returns:
        metrics: dictionary with dice and nsd scores
    """
    # Calculate Dice
    dice_per_class, mean_dice = calculate_dice(pred, target, num_classes)
    
    # Calculate NSD
    nsd_per_class, mean_nsd = calculate_nsd(pred, target, num_classes)
    
    metrics = {
        'dice_per_class': dice_per_class,
        'mean_dice': mean_dice,
        'nsd_per_class': nsd_per_class,
        'mean_nsd': mean_nsd
    }
    
    return metrics


def save_checkpoint(state, checkpoint_path, is_best=False, checkpoint_dir='./checkpoints'):
    """
    Save model checkpoint
    
    Args:
        state: dictionary containing model state, optimizer state, etc.
        checkpoint_path: path to save checkpoint
        is_best: whether this is the best model so far
        checkpoint_dir: directory to save checkpoints
    """
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        shutil.copyfile(checkpoint_path, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: path to checkpoint file
        model: model to load weights into
        optimizer: optimizer to load state into (optional)
        scheduler: scheduler to load state into (optional)
    
    Returns:
        start_epoch: epoch to resume from
        best_metric: best metric value
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_dice', 0.0)
    
    return start_epoch, best_metric


def compute_flops_and_params(model, input_size=(1, 1, 224, 224)):
    """
    Compute FLOPs and number of parameters
    
    Args:
        model: PyTorch model
        input_size: input tensor size (B, C, H, W)
    
    Returns:
        flops: FLOPs in billions
        params: number of parameters in millions
    """
    try:
        from thop import profile
        
        input_tensor = torch.randn(input_size)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        flops_g = flops / 1e9
        params_m = params / 1e6
        
        return flops_g, params_m
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        
        # Fallback: count parameters only
        params = sum(p.numel() for p in model.parameters())
        params_m = params / 1e6
        
        return None, params_m


class EarlyStopping:
    """
    Early stopping to stop training when validation metric doesn't improve
    """
    def __init__(self, patience=20, min_delta=0.0, mode='max'):
        """
        Args:
            patience: number of epochs to wait before stopping
            min_delta: minimum change to qualify as improvement
            mode: 'max' for maximization (e.g., dice), 'min' for minimization (e.g., loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0


def visualize_predictions(images, targets, predictions, num_classes, save_path=None):
    """
    Visualize segmentation predictions
    
    Args:
        images: (B, C, H, W) input images
        targets: (B, H, W) ground truth masks
        predictions: (B, H, W) predicted masks
        num_classes: number of classes
        save_path: path to save visualization (optional)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    batch_size = images.shape[0]
    
    # Create colormap
    colors = plt.cm.get_cmap('tab20', num_classes)
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Convert image to numpy and denormalize
        img = images[i].cpu().numpy()
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = img.transpose(1, 2, 0)
        
        # Normalize to [0, 1] for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        target = targets[i].cpu().numpy()
        pred = predictions[i].cpu().numpy()
        
        # Plot image
        axes[i, 0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(target, cmap=colors, vmin=0, vmax=num_classes-1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot prediction
        axes[i, 2].imshow(pred, cmap=colors, vmin=0, vmax=num_classes-1)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def get_dataset_statistics(dataset_name):
    """
    Get statistics for different datasets
    
    Returns:
        stats: dictionary with dataset statistics
    """
    stats = {
        'BTCV': {
            'num_classes': 14,
            'class_names': ['Background', 'Aorta', 'Gallbladder', 'Left Kidney', 
                          'Right Kidney', 'Liver', 'Pancreas', 'Spleen', 'Stomach',
                          'IVC', 'Portal Vein', 'Left Adrenal', 'Right Adrenal'],
            'input_size': (224, 224),
            'modality': 'CT',
            'num_train': 18,
            'num_test': 12
        },
        'ACDC': {
            'num_classes': 4,
            'class_names': ['Background', 'Right Ventricle', 'Myocardium', 'Left Ventricle'],
            'input_size': (224, 256),
            'modality': 'MRI',
            'num_train': 80,
            'num_test': 20
        },
        'EndoVis17': {
            'num_classes': 8,
            'class_names': ['Background', 'Bipolar Forceps', 'Prograsp Forceps', 
                          'Large Needle Driver', 'Monopolar Curved Scissors',
                          'Ultrasound Probe', 'Suction', 'Clip Applier'],
            'input_size': (384, 640),
            'modality': 'Endoscopy',
            'num_train': 1800,
            'num_test': 1200
        },
        'ATLAS23': {
            'num_classes': 3,
            'class_names': ['Background', 'Liver', 'Tumor'],
            'input_size': (250, 320),
            'modality': 'CE-MRI',
            'num_train': 48,
            'num_test': 12
        }
    }
    
    return stats.get(dataset_name, None)


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test Dice Loss
    num_classes = 4
    pred = torch.randn(2, num_classes, 224, 224)
    target = torch.randint(0, num_classes, (2, 224, 224))
    
    dice_loss = DiceLoss(num_classes)
    loss = dice_loss(pred, target)
    print(f"Dice Loss: {loss.item():.4f}")
    
    # Test metrics
    pred_class = torch.argmax(pred, dim=1)
    metrics = calculate_metrics(pred_class, target, num_classes)
    print(f"Mean Dice: {metrics['mean_dice']:.4f}")
    print(f"Mean NSD: {metrics['mean_nsd']:.4f}")
    
    # Test dataset statistics
    stats = get_dataset_statistics('BTCV')
    print(f"\nBTCV Statistics:")
    print(f"  Num classes: {stats['num_classes']}")
    print(f"  Input size: {stats['input_size']}")
    print(f"  Modality: {stats['modality']}")
