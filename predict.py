"""
Created on Thu March  20 22:49:39 2025

@author: drsaq

Prediction script for SAMA-UNet

Generates segmentation masks for test images
"""

import os
import argparse
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
import cv2
from PIL import Image

from model import SAMA_UNet
from dataloader import get_validation_augmentation
from utils import calculate_metrics, visualize_predictions, get_dataset_statistics, load_checkpoint


class Predictor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get dataset info
        self.dataset_stats = get_dataset_statistics(args.dataset)
        if self.dataset_stats is None:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        self.num_classes = self.dataset_stats['num_classes']
        self.in_channels = 3 if args.dataset == 'EndoVis17' else 1
        
        # Initialize model
        self.model = SAMA_UNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            embed_dims=args.embed_dims,
            depths=args.depths,
            num_heads=args.num_heads
        ).to(self.device)
        
        # Load checkpoint
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
        print(f"Loading checkpoint from {args.checkpoint}")
        load_checkpoint(args.checkpoint, self.model)
        self.model.eval()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Get transforms
        self.transform = get_validation_augmentation(args.dataset)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Dataset: {args.dataset}")
        print(f"Number of classes: {self.num_classes}")
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path: path to image file
        
        Returns:
            image_tensor: preprocessed image tensor (1, C, H, W)
            original_size: original image size for resizing back
        """
        # Load image based on format
        if image_path.endswith('.nii.gz') or image_path.endswith('.nii'):
            # NIfTI format (medical images)
            img = nib.load(image_path).get_fdata()
            
            # Handle 3D volumes - take middle slice
            if len(img.shape) == 3:
                img = img[:, :, img.shape[2] // 2]
            
            original_size = img.shape
            
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
            
        elif image_path.endswith(('.png', '.jpg', '.jpeg')):
            # Standard image formats
            if self.in_channels == 3:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            original_size = img.shape[:2]
        
        else:
            raise ValueError(f"Unsupported file format: {image_path}")
        
        # Apply transforms
        transformed = self.transform(image=img)
        img_tensor = transformed['image']
        
        # Add batch dimension
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor, original_size
    
    @torch.no_grad()
    def predict_single(self, image_path):
        """
        Predict segmentation for a single image
        
        Args:
            image_path: path to input image
        
        Returns:
            prediction: predicted segmentation mask (H, W)
        """
        # Preprocess
        image_tensor, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        output = self.model(image_tensor)
        
        # Get class predictions
        prediction = torch.argmax(output, dim=1).squeeze(0)
        
        # Resize to original size if needed
        prediction = prediction.cpu().numpy()
        if prediction.shape != original_size:
            prediction = cv2.resize(
                prediction.astype(np.uint8), 
                (original_size[1], original_size[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
        return prediction
    
    def predict_folder(self):
        """
        Predict segmentation for all images in a folder
        """
        input_dir = self.args.input_dir
        output_dir = self.args.output_dir
        
        # Get list of images
        image_files = []
        for ext in ['*.nii.gz', '*.nii', '*.png', '*.jpg', '*.jpeg']:
            import glob
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
        if len(image_files) == 0:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        results = []
        
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                # Predict
                prediction = self.predict_single(image_path)
                
                # Save prediction
                base_name = os.path.basename(image_path)
                save_name = os.path.splitext(base_name)[0] + '_pred.png'
                save_path = os.path.join(output_dir, save_name)
                
                # Save as PNG (scale to 0-255 for visualization)
                pred_vis = (prediction * (255 // (self.num_classes - 1))).astype(np.uint8)
                cv2.imwrite(save_path, pred_vis)
                
                # Also save as numpy for later analysis
                np_save_path = os.path.join(output_dir, os.path.splitext(base_name)[0] + '_pred.npy')
                np.save(np_save_path, prediction)
                
                results.append({
                    'image': base_name,
                    'prediction': save_name,
                    'prediction_npy': np_save_path
                })
                
                # If ground truth is provided, calculate metrics
                if self.args.gt_dir:
                    gt_path = os.path.join(self.args.gt_dir, base_name)
                    if os.path.exists(gt_path):
                        gt = self.load_ground_truth(gt_path)
                        
                        # Calculate metrics
                        pred_tensor = torch.from_numpy(prediction).unsqueeze(0)
                        gt_tensor = torch.from_numpy(gt).unsqueeze(0)
                        metrics = calculate_metrics(pred_tensor, gt_tensor, self.num_classes)
                        
                        results[-1]['dice'] = metrics['mean_dice']
                        results[-1]['nsd'] = metrics['mean_nsd']
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
        
        # Save results summary
        summary_path = os.path.join(output_dir, 'results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nProcessing complete!")
        print(f"Results saved to {output_dir}")
        print(f"Summary saved to {summary_path}")
        
        # Print average metrics if ground truth was provided
        if self.args.gt_dir and any('dice' in r for r in results):
            avg_dice = np.mean([r['dice'] for r in results if 'dice' in r])
            avg_nsd = np.mean([r['nsd'] for r in results if 'nsd' in r])
            print(f"\nAverage Metrics:")
            print(f"  Dice: {avg_dice:.4f}")
            print(f"  NSD: {avg_nsd:.4f}")
    
    def load_ground_truth(self, gt_path):
        """
        Load ground truth mask
        
        Args:
            gt_path: path to ground truth file
        
        Returns:
            gt: ground truth mask (H, W)
        """
        if gt_path.endswith('.nii.gz') or gt_path.endswith('.nii'):
            gt = nib.load(gt_path).get_fdata()
            if len(gt.shape) == 3:
                gt = gt[:, :, gt.shape[2] // 2]
        else:
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        return gt.astype(np.uint8)
    
    def visualize_results(self, num_samples=5):
        """
        Create visualizations of predictions
        
        Args:
            num_samples: number of samples to visualize
        """
        input_dir = self.args.input_dir
        output_dir = self.args.output_dir
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Get image files
        import glob
        image_files = []
        for ext in ['*.nii.gz', '*.nii', '*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
        # Select random samples
        if len(image_files) > num_samples:
            import random
            image_files = random.sample(image_files, num_samples)
        
        for idx, image_path in enumerate(image_files):
            # Load image and prediction
            image_tensor, _ = self.preprocess_image(image_path)
            prediction = self.predict_single(image_path)
            
            # Load ground truth if available
            if self.args.gt_dir:
                base_name = os.path.basename(image_path)
                gt_path = os.path.join(self.args.gt_dir, base_name)
                if os.path.exists(gt_path):
                    gt = self.load_ground_truth(gt_path)
                else:
                    gt = np.zeros_like(prediction)
            else:
                gt = np.zeros_like(prediction)
            
            # Visualize
            pred_tensor = torch.from_numpy(prediction).unsqueeze(0)
            gt_tensor = torch.from_numpy(gt).unsqueeze(0)
            
            save_path = os.path.join(vis_dir, f'sample_{idx+1}.png')
            visualize_predictions(
                image_tensor, 
                gt_tensor, 
                pred_tensor, 
                self.num_classes, 
                save_path
            )
        
        print(f"Visualizations saved to {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description='SAMA-UNet Inference')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save predictions')
    
    # Optional arguments
    parser.add_argument('--dataset', type=str, default='BTCV',
                       choices=['BTCV', 'ACDC', 'EndoVis17', 'ATLAS23'],
                       help='Dataset name')
    parser.add_argument('--gt_dir', type=str, default='',
                       help='Directory containing ground truth masks (for evaluation)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations of predictions')
    parser.add_argument('--num_vis_samples', type=int, default=5,
                       help='Number of samples to visualize')
    
    # Model parameters (should match training)
    parser.add_argument('--embed_dims', type=int, nargs='+', default=[96, 192, 384, 768],
                       help='Embedding dimensions for each stage')
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 2, 2],
                       help='Number of blocks in each stage')
    parser.add_argument('--num_heads', type=int, nargs='+', default=[3, 6, 12, 24],
                       help='Number of attention heads in each stage')
    
    args = parser.parse_args()
    
    # Create predictor and run inference
    predictor = Predictor(args)
    predictor.predict_folder()
    
    # Create visualizations if requested
    if args.visualize:
        predictor.visualize_results(args.num_vis_samples)


if __name__ == '__main__':
    main()
