"""
Created on Thu Feb 26 19:21:45 2025

@author: drsaq

Data loading utilities for SAMA-UNet
Supports BTCV, ACDC, EndoVis17, and ATLAS23 datasets
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import json


class BTCVDataset(Dataset):
    """
    BTCV (Synapse Multi-Organ Segmentation) Dataset
    CT scans with 13 organ classes
    Input size: 224x224 (as per paper)
    """
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load image and label paths
        self.image_paths = []
        self.label_paths = []
        
        split_file = os.path.join(data_dir, f'{split}_slices.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    img_path, lbl_path = line.strip().split(',')
                    self.image_paths.append(os.path.join(data_dir, img_path))
                    self.label_paths.append(os.path.join(data_dir, lbl_path))
        else:
            # Auto-discover slices
            img_dir = os.path.join(data_dir, 'imagesTr' if split == 'train' else 'imagesTs')
            lbl_dir = os.path.join(data_dir, 'labelsTr' if split == 'train' else 'labelsTs')
            
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith('.nii.gz'):
                    self.image_paths.append(os.path.join(img_dir, fname))
                    self.label_paths.append(os.path.join(lbl_dir, fname))
        
        print(f"Loaded {len(self.image_paths)} samples for {split} split")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load NIfTI file
        img = nib.load(self.image_paths[idx]).get_fdata()
        lbl = nib.load(self.label_paths[idx]).get_fdata()
        
        # Handle 3D volumes - take middle slice if 3D
        if len(img.shape) == 3:
            img = img[:, :, img.shape[2] // 2]
            lbl = lbl[:, :, lbl.shape[2] // 2]
        
        # Normalize image to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Convert to uint8 for augmentation
        img = (img * 255).astype(np.uint8)
        lbl = lbl.astype(np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img, mask=lbl)
            img = transformed['image']
            lbl = transformed['mask']
        
        # Add channel dimension if needed
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        
        return img.float(), lbl.long()


class ACDCDataset(Dataset):
    """
    ACDC (Automated Cardiac Diagnosis Challenge) Dataset
    MRI scans with 3 cardiac structures
    Input size: 256x224 (as per paper)
    """
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.image_paths = []
        self.label_paths = []
        
        # Load from split file or auto-discover
        split_file = os.path.join(data_dir, f'{split}_list.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    img_path = line.strip()
                    lbl_path = img_path.replace('_image', '_label')
                    self.image_paths.append(os.path.join(data_dir, img_path))
                    self.label_paths.append(os.path.join(data_dir, lbl_path))
        
        print(f"Loaded {len(self.image_paths)} samples for ACDC {split} split")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = nib.load(self.image_paths[idx]).get_fdata()
        lbl = nib.load(self.label_paths[idx]).get_fdata()
        
        # Handle 3D
        if len(img.shape) == 3:
            img = img[:, :, img.shape[2] // 2]
            lbl = lbl[:, :, lbl.shape[2] // 2]
        
        # Normalize
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        lbl = lbl.astype(np.uint8)
        
        if self.transform:
            transformed = self.transform(image=img, mask=lbl)
            img = transformed['image']
            lbl = transformed['mask']
        
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        
        return img.float(), lbl.long()


class EndoVis17Dataset(Dataset):
    """
    EndoVis17 (MICCAI 2017 Endoscopic Vision Challenge) Dataset
    Surgical instrument segmentation (7 classes)
    Input size: 384x640 (as per paper)
    """
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.image_paths = []
        self.label_paths = []
        
        # Typically organized as videos with frames
        split_dir = os.path.join(data_dir, split)
        
        for video_folder in sorted(os.listdir(split_dir)):
            video_path = os.path.join(split_dir, video_folder)
            if not os.path.isdir(video_path):
                continue
                
            img_folder = os.path.join(video_path, 'left_frames')
            lbl_folder = os.path.join(video_path, 'labels')
            
            if os.path.exists(img_folder) and os.path.exists(lbl_folder):
                for fname in sorted(os.listdir(img_folder)):
                    if fname.endswith(('.png', '.jpg')):
                        self.image_paths.append(os.path.join(img_folder, fname))
                        # Label typically has same name
                        lbl_name = fname.replace('.jpg', '.png')
                        self.label_paths.append(os.path.join(lbl_folder, lbl_name))
        
        print(f"Loaded {len(self.image_paths)} samples for EndoVis17 {split} split")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        lbl = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            transformed = self.transform(image=img, mask=lbl)
            img = transformed['image']
            lbl = transformed['mask']
        
        return img.float(), lbl.long()


class ATLAS23Dataset(Dataset):
    """
    ATLAS23 (Liver Tumor Segmentation) Dataset
    CE-MRI T1-weighted scans
    Input size: 320x250 (as per paper)
    """
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.image_paths = []
        self.label_paths = []
        
        # Load split
        split_file = os.path.join(data_dir, f'{split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    case_id = line.strip()
                    img_path = os.path.join(data_dir, 'images', f'{case_id}.nii.gz')
                    lbl_path = os.path.join(data_dir, 'labels', f'{case_id}.nii.gz')
                    
                    if os.path.exists(img_path) and os.path.exists(lbl_path):
                        self.image_paths.append(img_path)
                        self.label_paths.append(lbl_path)
        
        print(f"Loaded {len(self.image_paths)} samples for ATLAS23 {split} split")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = nib.load(self.image_paths[idx]).get_fdata()
        lbl = nib.load(self.label_paths[idx]).get_fdata()
        
        # Handle 3D
        if len(img.shape) == 3:
            img = img[:, :, img.shape[2] // 2]
            lbl = lbl[:, :, lbl.shape[2] // 2]
        
        # Normalize
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        lbl = lbl.astype(np.uint8)
        
        if self.transform:
            transformed = self.transform(image=img, mask=lbl)
            img = transformed['image']
            lbl = transformed['mask']
        
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        
        return img.float(), lbl.long()


def get_training_augmentation(dataset_name='BTCV'):
    """
    Get training augmentation pipeline based on dataset
    Following nnUNet preprocessing strategy
    """
    if dataset_name == 'BTCV':
        size = (224, 224)
    elif dataset_name == 'ACDC':
        size = (224, 256)
    elif dataset_name == 'EndoVis17':
        size = (384, 640)
    elif dataset_name == 'ATLAS23':
        size = (250, 320)
    else:
        size = (224, 224)
    
    return A.Compose([
        A.Resize(size[0], size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.GridDistortion(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ])


def get_validation_augmentation(dataset_name='BTCV'):
    """
    Get validation/test augmentation pipeline
    Only resize and normalize
    """
    if dataset_name == 'BTCV':
        size = (224, 224)
    elif dataset_name == 'ACDC':
        size = (224, 256)
    elif dataset_name == 'EndoVis17':
        size = (384, 640)
    elif dataset_name == 'ATLAS23':
        size = (250, 320)
    else:
        size = (224, 224)
    
    return A.Compose([
        A.Resize(size[0], size[1]),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ])


def get_dataloader(dataset_name, data_dir, batch_size=8, num_workers=4, split='train'):
    """
    Factory function to create dataloaders for different datasets
    
    Args:
        dataset_name: 'BTCV', 'ACDC', 'EndoVis17', or 'ATLAS23'
        data_dir: path to dataset directory
        batch_size: batch size
        num_workers: number of workers for dataloader
        split: 'train', 'val', or 'test'
    
    Returns:
        DataLoader object
    """
    
    # Get appropriate transform
    if split == 'train':
        transform = get_training_augmentation(dataset_name)
    else:
        transform = get_validation_augmentation(dataset_name)
    
    # Create dataset
    if dataset_name == 'BTCV':
        dataset = BTCVDataset(data_dir, split=split, transform=transform)
    elif dataset_name == 'ACDC':
        dataset = ACDCDataset(data_dir, split=split, transform=transform)
    elif dataset_name == 'EndoVis17':
        dataset = EndoVis17Dataset(data_dir, split=split, transform=transform)
    elif dataset_name == 'ATLAS23':
        dataset = ATLAS23Dataset(data_dir, split=split, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloader
    shuffle = (split == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


def create_dataset_splits(data_dir, dataset_name='BTCV', train_ratio=0.8):
    """
    Create train/val split files if they don't exist
    
    Args:
        data_dir: path to dataset directory
        dataset_name: name of dataset
        train_ratio: ratio of training samples
    """
    
    if dataset_name == 'BTCV':
        # BTCV has 30 cases: 18 train, 12 test (as per paper)
        # We'll use the official split
        train_cases = list(range(1, 19))  # cases 1-18
        test_cases = list(range(19, 31))   # cases 19-30
        
        train_file = os.path.join(data_dir, 'train_slices.txt')
        test_file = os.path.join(data_dir, 'test_slices.txt')
        
        # Write split files (placeholder - actual implementation would scan volumes)
        with open(train_file, 'w') as f:
            for case in train_cases:
                f.write(f"case_{case:04d},case_{case:04d}_label\n")
        
        with open(test_file, 'w') as f:
            for case in test_cases:
                f.write(f"case_{case:04d},case_{case:04d}_label\n")
        
        print(f"Created BTCV splits: {len(train_cases)} train, {len(test_cases)} test")
    
    elif dataset_name == 'ACDC':
        # ACDC: 80 train, 20 test (as per paper)
        all_cases = list(range(1, 101))
        np.random.shuffle(all_cases)
        
        train_cases = all_cases[:80]
        test_cases = all_cases[80:]
        
        # Write splits
        train_file = os.path.join(data_dir, 'train_list.txt')
        test_file = os.path.join(data_dir, 'test_list.txt')
        
        with open(train_file, 'w') as f:
            for case in train_cases:
                f.write(f"patient{case:03d}_image.nii.gz\n")
        
        with open(test_file, 'w') as f:
            for case in test_cases:
                f.write(f"patient{case:03d}_image.nii.gz\n")
        
        print(f"Created ACDC splits: {len(train_cases)} train, {len(test_cases)} test")


if __name__ == "__main__":
    # Test dataloader
    print("Testing BTCV dataloader...")
    
    # Create dummy transforms
    transform = get_training_augmentation('BTCV')
    
    # Test with dummy data directory
    # In practice, replace with actual data path
    data_dir = './data/BTCV'
    
    try:
        loader = get_dataloader('BTCV', data_dir, batch_size=2, num_workers=0, split='train')
        print(f"Created dataloader with {len(loader)} batches")
        
        # Test one batch
        for images, labels in loader:
            print(f"Image batch shape: {images.shape}")
            print(f"Label batch shape: {labels.shape}")
            break
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        print("Make sure to set correct data_dir path")