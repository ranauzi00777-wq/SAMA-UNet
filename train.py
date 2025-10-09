"""

Created on Thu March 12 11:41:25 2025

@author: drsaq

Training script for SAMA-UNet

"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import custom modules
from model import SAMA_UNet
from dataloader import get_dataloader
from utils import (
    DiceLoss,
    calculate_dice,
    calculate_nsd,
    calculate_metrics,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    AverageMeter,
    set_seed,
    get_dataset_statistics,
    visualize_predictions,
    compute_flops_and_params
)


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Dice Loss"""
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        
    def forward(self, predictions, targets):
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.ce_weight * ce + self.dice_weight * dice


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, args):
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
    
    for batch_idx, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            pred_labels = torch.argmax(outputs, dim=1)
            dice_per_class, mean_dice = calculate_dice(pred_labels, labels, args.num_classes)
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        dice_meter.update(mean_dice, images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'dice': f'{dice_meter.avg:.4f}'
        })
        
        # Log to tensorboard
        if batch_idx % args.log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss_meter.avg, global_step)
            writer.add_scalar('Train/Dice', dice_meter.avg, global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
    
    return loss_meter.avg, dice_meter.avg


def validate(model, dataloader, criterion, device, epoch, writer, args, save_vis=False):
    """Validate the model"""
    model.eval()
    
    loss_meter = AverageMeter()
    dice_all_classes = [[] for _ in range(args.num_classes)]
    nsd_all_classes = [[] for _ in range(args.num_classes)]
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
    
    # For visualization
    vis_images = []
    vis_targets = []
    vis_predictions = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Get predictions
            pred_labels = torch.argmax(outputs, dim=1)
            
            # Calculate metrics
            metrics = calculate_metrics(pred_labels, labels, args.num_classes)
            
            # Store per-class scores
            for cls_idx, (dice_score, nsd_score) in enumerate(
                zip(metrics['dice_per_class'], metrics['nsd_per_class'])
            ):
                dice_all_classes[cls_idx + 1].append(dice_score)  # +1 to skip background
                nsd_all_classes[cls_idx + 1].append(nsd_score)
            
            loss_meter.update(loss.item(), images.size(0))
            
            # Collect samples for visualization
            if save_vis and batch_idx == 0:
                vis_images = images[:min(4, images.size(0))]
                vis_targets = labels[:min(4, labels.size(0))]
                vis_predictions = pred_labels[:min(4, pred_labels.size(0))]
            
            # Update progress bar
            mean_dice = metrics['mean_dice']
            mean_nsd = metrics['mean_nsd']
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'dice': f'{mean_dice:.4f}',
                'nsd': f'{mean_nsd:.4f}'
            })
    
    # Calculate average metrics per class
    dice_per_class = []
    nsd_per_class = []
    
    for cls_idx in range(args.num_classes):
        if dice_all_classes[cls_idx]:
            dice_per_class.append(np.mean(dice_all_classes[cls_idx]))
        else:
            dice_per_class.append(0.0)
        
        if nsd_all_classes[cls_idx]:
            nsd_per_class.append(np.mean(nsd_all_classes[cls_idx]))
        else:
            nsd_per_class.append(0.0)
    
    # Calculate mean metrics (excluding background)
    mean_dice = np.mean([d for d in dice_per_class[1:] if d > 0])
    mean_nsd = np.mean([n for n in nsd_per_class[1:] if n > 0])
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/Dice', mean_dice, epoch)
    writer.add_scalar('Val/NSD', mean_nsd, epoch)
    
    # Log per-class metrics
    if args.dataset in ['BTCV', 'ACDC', 'EndoVis17', 'ATLAS23']:
        stats = get_dataset_statistics(args.dataset)
        class_names = stats['class_names']
        
        for cls_idx in range(1, args.num_classes):  # Skip background
            if cls_idx < len(class_names):
                class_name = class_names[cls_idx]
                writer.add_scalar(f'Val/Dice_{class_name}', dice_per_class[cls_idx], epoch)
                writer.add_scalar(f'Val/NSD_{class_name}', nsd_per_class[cls_idx], epoch)
    
    # Save visualization
    if save_vis and len(vis_images) > 0:
        vis_dir = Path(args.output_dir) / args.dataset / args.run_name / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f'epoch_{epoch+1:03d}.png'
        visualize_predictions(vis_images, vis_targets, vis_predictions, 
                            args.num_classes, save_path=vis_path)
    
    return loss_meter.avg, mean_dice, mean_nsd, dice_per_class, nsd_per_class


def print_metrics_summary(dice_per_class, nsd_per_class, dataset_name):
    """Print detailed metrics summary"""
    stats = get_dataset_statistics(dataset_name)
    if stats is None:
        return
    
    class_names = stats['class_names']
    
    print("\n" + "="*80)
    print("Per-Class Metrics Summary:")
    print("="*80)
    print(f"{'Class':<30} {'DSC (%)':<15} {'NSD (%)':<15}")
    print("-"*80)
    
    for cls_idx in range(1, len(dice_per_class)):  # Skip background
        if cls_idx < len(class_names):
            class_name = class_names[cls_idx]
            dice = dice_per_class[cls_idx] * 100
            nsd = nsd_per_class[cls_idx] * 100
            print(f"{class_name:<30} {dice:<15.2f} {nsd:<15.2f}")
    
    mean_dice = np.mean([d for d in dice_per_class[1:] if d > 0]) * 100
    mean_nsd = np.mean([n for n in nsd_per_class[1:] if n > 0]) * 100
    print("-"*80)
    print(f"{'Mean':<30} {mean_dice:<15.2f} {mean_nsd:<15.2f}")
    print("="*80 + "\n")


def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create run name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.run_name = timestamp
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.dataset / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"SAMA-UNet Training")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Get dataset-specific parameters
    stats = get_dataset_statistics(args.dataset)
    if stats is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    args.num_classes = stats['num_classes']
    args.in_channels = 1 if args.dataset != 'EndoVis17' else 3
    
    print(f"\nDataset: {args.dataset}")
    print(f"  Modality: {stats['modality']}")
    print(f"  Number of classes: {args.num_classes}")
    print(f"  Input channels: {args.in_channels}")
    print(f"  Input size: {stats['input_size']}")
    print(f"  Train samples: {stats['num_train']}")
    print(f"  Test samples: {stats['num_test']}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        train_loader = get_dataloader(
            args.dataset, 
            args.data_dir, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split='train'
        )
        
        val_loader = get_dataloader(
            args.dataset,
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split='val'
        )
        
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print("Please ensure dataset path is correct and data is properly formatted.")
        return
    
    # Create model
    print("\nCreating SAMA-UNet model...")
    model = SAMA_UNet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        embed_dims=args.embed_dims,
        depths=args.depths,
        num_heads=args.num_heads
    )
    model = model.to(device)
    
    # Count parameters and compute FLOPs
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    if args.compute_flops:
        input_size = (1, args.in_channels, stats['input_size'][0], stats['input_size'][1])
        flops, params = compute_flops_and_params(model, input_size)
        if flops is not None:
            print(f"  FLOPs: {flops:.2f}G")
    
    # Setup loss function
    criterion = CombinedLoss(
        num_classes=args.num_classes,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight
    )
    print(f"\nLoss function: Combined (CE: {args.ce_weight}, Dice: {args.dice_weight})")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    
    # Setup learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    else:
        scheduler = None
    
    print(f"Scheduler: {args.scheduler if scheduler else 'None'}")
    
    # Setup early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            mode='max'
        )
        print(f"Early stopping: Enabled (patience={args.patience})")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_dice = 0.0
    best_nsd = 0.0
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        try:
            start_epoch, best_dice = load_checkpoint(
                args.resume, model, optimizer, scheduler
            )
            start_epoch += 1
            print(f"  Resumed from epoch {start_epoch}")
            print(f"  Best Dice so far: {best_dice:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Initial LR: {args.lr}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, epoch, writer, args
        )
        
        # Validate
        if (epoch + 1) % args.val_interval == 0:
            save_vis = (epoch + 1) % args.vis_interval == 0
            val_loss, val_dice, val_nsd, val_dice_per_class, val_nsd_per_class = validate(
                model, val_loader, criterion, 
                device, epoch, writer, args, save_vis=save_vis
            )
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Check for improvement
            is_best = val_dice > best_dice
            if is_best:
                best_dice = val_dice
                best_nsd = val_nsd
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_dice': best_dice,
                'best_nsd': best_nsd,
                'train_loss': train_loss,
                'train_dice': train_dice,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_nsd': val_nsd,
                'val_dice_per_class': val_dice_per_class,
                'val_nsd_per_class': val_nsd_per_class,
                'args': vars(args)
            }
            
            # Save regular checkpoint (keep last N)
            if (epoch + 1) % args.save_interval == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
                save_checkpoint(checkpoint, checkpoint_path, is_best=False, 
                              checkpoint_dir=checkpoint_dir)
            
            # Save best checkpoint
            if is_best:
                best_checkpoint_path = checkpoint_dir / 'best_model.pth'
                save_checkpoint(checkpoint, best_checkpoint_path, is_best=True,
                              checkpoint_dir=checkpoint_dir)
                print(f"\n✓ New best model saved! Dice: {best_dice:.4f}, NSD: {best_nsd:.4f}")
                
                # Print detailed metrics
                print_metrics_summary(val_dice_per_class, val_nsd_per_class, args.dataset)
            
            # Save latest checkpoint
            latest_checkpoint_path = checkpoint_dir / 'latest_checkpoint.pth'
            save_checkpoint(checkpoint, latest_checkpoint_path, is_best=False,
                          checkpoint_dir=checkpoint_dir)
            
            # Early stopping check
            if early_stopping is not None:
                early_stopping(val_dice)
                if early_stopping.early_stop:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs} Summary")
        print(f"{'='*80}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        if (epoch + 1) % args.val_interval == 0:
            print(f"  Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val NSD: {val_nsd:.4f}")
            print(f"  Best Dice: {best_dice:.4f} | Best NSD: {best_nsd:.4f}")
            if scheduler is not None:
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*80}\n")
    
    print("\n" + "="*80)
    print("Training Completed!")
    print("="*80)
    print(f"Best Dice Score: {best_dice:.4f}")
    print(f"Best NSD Score: {best_nsd:.4f}")
    print(f"Model saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")
    print("="*80 + "\n")
    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SAMA-UNet for Medical Image Segmentation')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='BTCV',
                        choices=['BTCV', 'ACDC', 'EndoVis17', 'ATLAS23'],
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay coefficient')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Loss parameters
    parser.add_argument('--ce_weight', type=float, default=0.5,
                        help='Weight for Cross-Entropy loss')
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Weight for Dice loss')
    
    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--step_size', type=int, default=100,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    
    # Early stopping parameters
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Minimum delta for early stopping')
    
    # Model parameters
    parser.add_argument('--embed_dims', type=int, nargs='+', 
                        default=[96, 192, 384, 768],
                        help='Embedding dimensions per stage')
    parser.add_argument('--depths', type=int, nargs='+',
                        default=[2, 2, 2, 2],
                        help='Number of SAMA blocks per stage')
    parser.add_argument('--num_heads', type=int, nargs='+',
                        default=[3, 6, 12, 24],
                        help='Number of attention heads per stage')
    
    # I/O parameters
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    
    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval in iterations')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validation interval in epochs')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Checkpoint saving interval in epochs')
    parser.add_argument('--vis_interval', type=int, default=10,
                        help='Visualization interval in epochs')
    
    # Compute parameters
    parser.add_argument('--compute_flops', action='store_true',
                        help='Compute FLOPs (requires thop package)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.embed_dims) != len(args.depths) or len(args.embed_dims) != len(args.num_heads):
        raise ValueError("embed_dims, depths, and num_heads must have the same length")
    
    if args.ce_weight + args.dice_weight != 1.0:
        print(f"Warning: ce_weight ({args.ce_weight}) + dice_weight ({args.dice_weight}) != 1.0")
    
    main(args)
