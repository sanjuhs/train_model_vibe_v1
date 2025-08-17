#!/usr/bin/env python3
"""
Training Pipeline for Audio-to-Blendshapes TCN
Implements specialized loss functions and training procedures
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import joblib
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path to import models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.tcn_model import create_model

class AudioBlendshapeLoss(nn.Module):
    """
    Specialized loss function for audio-to-blendshapes training
    """
    def __init__(self, 
                 base_weight=1.0,           # Base loss weight
                 temporal_weight=0.0,       # Start with 0, enable after base loss drops
                 silence_weight=0.0,        # Start with 0, enable after base loss drops
                 pose_clamp_weight=0.0):    # Start with 0, enable after base loss drops
        super().__init__()
        
        self.base_weight = base_weight
        self.temporal_weight = temporal_weight
        self.silence_weight = silence_weight
        self.pose_clamp_weight = pose_clamp_weight
        
        # Mouth-related blendshape indices (approximate - adjust based on MediaPipe mapping)
        # These are typically jaw and mouth-related indices
        self.mouth_indices = [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]  # Approximate
        
        # Pose indices (last 7 values: x,y,z,qw,qx,qy,qz)
        self.pose_indices = list(range(52, 59))
        
        print(f"Loss function initialized:")
        print(f"  Base weight: {base_weight}")
        print(f"  Temporal weight: {temporal_weight}")
        print(f"  Silence weight: {silence_weight}")
        print(f"  Pose clamp weight: {pose_clamp_weight}")
    
    def enable_temporal_loss(self, weight=0.1):
        """Enable temporal smoothness loss"""
        self.temporal_weight = weight
        print(f"Temporal loss enabled with weight: {weight}")
    
    def enable_silence_loss(self, weight=2.0):
        """Enable silence weighting loss"""
        self.silence_weight = weight
        print(f"Silence loss enabled with weight: {weight}")
    
    def enable_pose_loss(self, weight=0.05):
        """Enable pose clamping loss"""
        self.pose_clamp_weight = weight
        print(f"Pose loss enabled with weight: {weight}")
    
    def forward(self, predictions, targets, voice_activity=None):
        """
        Compute specialized loss
        
        Args:
            predictions: Model predictions (batch, time, 59)
            targets: Ground truth targets (batch, time, 59)
            voice_activity: Voice activity detection (batch, time) - optional
        
        Returns:
            dict: Loss components and total loss
        """
        batch_size, seq_len, feature_dim = predictions.shape
        
        # 1. Base loss (Huber with delta=1, equivalent to smooth L1)
        base_loss = nn.functional.smooth_l1_loss(predictions, targets, reduction='mean')
        
        # 2. Temporal smoothness penalties
        pred_diff1 = predictions[:, 1:] - predictions[:, :-1]  # First derivative
        target_diff1 = targets[:, 1:] - targets[:, :-1]
        temporal_loss1 = torch.mean(torch.abs(pred_diff1 - target_diff1))
        
        pred_diff2 = pred_diff1[:, 1:] - pred_diff1[:, :-1]  # Second derivative
        target_diff2 = target_diff1[:, 1:] - target_diff1[:, :-1]
        temporal_loss2 = torch.mean(torch.abs(pred_diff2 - target_diff2))
        
        temporal_loss = temporal_loss1 + temporal_loss2
        
        # 3. Silence weighting for mouth features
        silence_loss = 0.0
        if voice_activity is not None and self.silence_weight > 0:
            # During silence (VAD=0), up-weight mouth features
            vad = voice_activity.unsqueeze(-1)  # (batch, time, 1)
            silence_mask = (vad == 0)  # Silence frames
            
            if silence_mask.any():
                # Focus on mouth-related blendshapes during silence
                mouth_pred = predictions[:, :, self.mouth_indices]
                mouth_target = targets[:, :, self.mouth_indices]
                mouth_error = torch.abs(mouth_pred - mouth_target)
                
                # Weight mouth errors during silence - average only over silence frames
                silence_mask_expanded = silence_mask.expand_as(mouth_error)
                silence_error = mouth_error[silence_mask_expanded]
                silence_loss = silence_error.mean() if silence_error.numel() > 0 else 0.0
        
        # 4. Pose clamping loss (keep pose near training distribution)
        pose_loss = 0.0
        if self.pose_clamp_weight > 0:
            pose_pred = predictions[:, :, self.pose_indices]
            pose_target = targets[:, :, self.pose_indices]
            
            # Translation components (x,y,z) - indices 52,53,54
            trans_pred = pose_pred[:, :, :3]
            trans_target = pose_target[:, :, :3]
            
            # Check if targets are within expected range (silently check, no spam)
            target_in_range = torch.all(torch.abs(trans_target) <= 0.2)
            # Removed excessive warning print - pose targets can exceed range normally
            
            # Clamp translation to reasonable range and penalize large deviations
            trans_clamped = torch.clamp(trans_pred, -0.2, 0.2)
            clamp_penalty = torch.mean((trans_pred - trans_clamped) ** 2)
            
            # L2 loss on pose
            pose_loss = nn.functional.mse_loss(pose_pred, pose_target, reduction='mean')
            pose_loss = pose_loss + clamp_penalty
        
        # Combine losses
        total_loss = (
            self.base_weight * base_loss +
            self.temporal_weight * temporal_loss +
            self.silence_weight * silence_loss +
            self.pose_clamp_weight * pose_loss
        )
        
        return {
            'total_loss': total_loss,
            'base_loss': base_loss.item(),
            'temporal_loss': temporal_loss.item(),
            'silence_loss': silence_loss if isinstance(silence_loss, float) else silence_loss.item(),
            'pose_loss': pose_loss if isinstance(pose_loss, float) else pose_loss.item()
        }

class TCNTrainer:
    """
    Trainer for the TCN model
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        self.criterion = AudioBlendshapeLoss()
        
        # Key mouth and pose indices for validation metrics
        self.jaw_open_idx = 25  # Approximate index for jaw open
        self.lip_close_idx = 12  # Approximate index for lip closure
        self.smile_idx = 20     # Approximate index for smile
        
        print(f"Trainer initialized on device: {device}")
    
    def setup_optimizer(self, learning_rate=2e-3, weight_decay=1e-4):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # We'll set up OneCycleLR after we know total steps
        self.base_lr = learning_rate
        self.max_lr = 3e-3
        
        print(f"Optimizer configured: AdamW(lr={learning_rate}, wd={weight_decay})")
    
    def setup_scheduler(self, total_steps):
        """Setup OneCycleLR scheduler"""
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            total_steps=total_steps,
            pct_start=0.3,  # 30% warmup
            anneal_strategy='cos'
        )
        print(f"Scheduler configured: OneCycleLR(max_lr={self.max_lr}, total_steps={total_steps})")
    
    def load_dataset(self, data_dir="extracted_features", normalize=True):
        """Load training dataset with optional normalization"""
        data_path = Path(data_dir)
        
        # Load arrays
        audio_sequences = np.load(data_path / "audio_sequences.npy")
        target_sequences = np.load(data_path / "target_sequences.npy")
        vad_sequences = np.load(data_path / "vad_sequences.npy")
        
        # Load metadata
        with open(data_path / "dataset_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"Dataset loaded:")
        print(f"  Audio sequences: {audio_sequences.shape}")
        print(f"  Target sequences: {target_sequences.shape}")
        print(f"  VAD sequences: {vad_sequences.shape}")
        
        # Normalize audio features (mel spectrograms)
        if normalize:
            print(f"Normalizing audio features...")
            
            # Audio normalization: z-score per feature across all samples
            audio_mean = audio_sequences.mean(axis=(0, 1), keepdims=True)
            audio_std = audio_sequences.std(axis=(0, 1), keepdims=True) + 1e-6
            audio_sequences = (audio_sequences - audio_mean) / audio_std
            
            print(f"  Audio - Mean: {audio_mean.mean():.4f}, Std: {audio_std.mean():.4f}")
            print(f"  Audio range after norm: [{audio_sequences.min():.3f}, {audio_sequences.max():.3f}]")
            
            # Target normalization: already bounded but check ranges
            print(f"  Target range: [{target_sequences.min():.3f}, {target_sequences.max():.3f}]")
            
            # Check if targets need normalization (blendshapes should be [0,1], pose might need scaling)
            blendshapes = target_sequences[:, :, :52]
            pose = target_sequences[:, :, 52:]
            
            print(f"  Blendshapes range: [{blendshapes.min():.3f}, {blendshapes.max():.3f}]")
            print(f"  Pose range: [{pose.min():.3f}, {pose.max():.3f}]")
            
            # Warn if blendshapes are outside [0,1]
            if blendshapes.min() < -0.1 or blendshapes.max() > 1.1:
                print(f"  WARNING: Blendshapes outside expected [0,1] range!")
            
            # Warn if pose is outside reasonable range
            if np.abs(pose).max() > 0.5:
                print(f"  WARNING: Pose values seem large. Max abs: {np.abs(pose).max():.3f}")
        
        return {
            'audio': torch.FloatTensor(audio_sequences),
            'targets': torch.FloatTensor(target_sequences),
            'vad': torch.FloatTensor(vad_sequences),
            'metadata': metadata
        }
    
    def create_data_loader(self, dataset, batch_size=16, shuffle=True):
        """Create PyTorch data loader"""
        dataset_tensor = torch.utils.data.TensorDataset(
            dataset['audio'],
            dataset['targets'],
            dataset['vad']
        )
        
        return torch.utils.data.DataLoader(
            dataset_tensor,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if self.device == 'cuda' else False
        )
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'base_loss': 0.0,
            'temporal_loss': 0.0,
            'silence_loss': 0.0,
            'pose_loss': 0.0,
            'num_batches': 0
        }
        
        pbar = tqdm(data_loader, desc="Training")
        
        for batch_idx, (audio, targets, vad) in enumerate(pbar):
            audio = audio.to(self.device)
            targets = targets.to(self.device)
            vad = vad.to(self.device)
            
            # Forward pass
            predictions = self.model(audio)
            
            # Compute loss
            loss_dict = self.criterion(predictions, targets, vad)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            for key in epoch_metrics:
                if key == 'num_batches':
                    epoch_metrics[key] += 1
                else:
                    epoch_metrics[key] += loss_dict[key.replace('total_loss', 'total_loss').replace('total_loss', 'total_loss')]
            
            # Log detailed loss components every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"\nBatch {batch_idx+1}: "
                      f"base={loss_dict['base_loss']:.4f}, "
                      f"temp={loss_dict['temporal_loss']:.4f}, "
                      f"sil={loss_dict['silence_loss']:.4f}, "
                      f"pose={loss_dict['pose_loss']:.4f}, "
                      f"grad_norm={grad_norm:.3f}")
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}",
                'grad': f"{grad_norm:.3f}"
            })
        
        # Average metrics
        for key in epoch_metrics:
            if key != 'num_batches':
                epoch_metrics[key] /= epoch_metrics['num_batches']
        
        return epoch_metrics
    
    def validate(self, data_loader):
        """Validate the model"""
        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'base_loss': 0.0,
            'temporal_loss': 0.0,
            'silence_loss': 0.0,
            'pose_loss': 0.0,
            'num_batches': 0
        }
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for audio, targets, vad in tqdm(data_loader, desc="Validation"):
                audio = audio.to(self.device)
                targets = targets.to(self.device)
                vad = vad.to(self.device)
                
                predictions = self.model(audio)
                loss_dict = self.criterion(predictions, targets, vad)
                
                # Update metrics
                for key in val_metrics:
                    if key == 'num_batches':
                        val_metrics[key] += 1
                    else:
                        val_metrics[key] += loss_dict[key.replace('total_loss', 'total_loss').replace('total_loss', 'total_loss')]
                
                # Collect predictions for detailed metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Average metrics
        for key in val_metrics:
            if key != 'num_batches':
                val_metrics[key] /= val_metrics['num_batches']
        
        # Compute detailed validation metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        detailed_metrics = self.compute_detailed_metrics(all_predictions, all_targets)
        val_metrics.update(detailed_metrics)
        
        return val_metrics
    
    def compute_detailed_metrics(self, predictions, targets):
        """Compute detailed validation metrics"""
        # Flatten sequences for metric computation
        pred_flat = predictions.view(-1, predictions.size(-1)).numpy()
        target_flat = targets.view(-1, targets.size(-1)).numpy()
        
        # MAE per channel
        mae_per_channel = np.mean(np.abs(pred_flat - target_flat), axis=0)
        
        # Key feature MAEs
        jaw_open_mae = mae_per_channel[self.jaw_open_idx]
        lip_close_mae = mae_per_channel[self.lip_close_idx]
        smile_mae = mae_per_channel[self.smile_idx]
        
        # Pose MAE (last 7 features)
        pose_mae = np.mean(mae_per_channel[52:59])
        
        # Pearson correlations for key features
        jaw_corr = pearsonr(pred_flat[:, self.jaw_open_idx], target_flat[:, self.jaw_open_idx])[0]
        lip_corr = pearsonr(pred_flat[:, self.lip_close_idx], target_flat[:, self.lip_close_idx])[0]
        smile_corr = pearsonr(pred_flat[:, self.smile_idx], target_flat[:, self.smile_idx])[0]
        
        # Overall metrics
        overall_mae = np.mean(mae_per_channel)
        mouth_mae = np.mean(mae_per_channel[10:30])  # Approximate mouth region
        
        return {
            'overall_mae': overall_mae,
            'mouth_mae': mouth_mae,
            'jaw_open_mae': jaw_open_mae,
            'lip_close_mae': lip_close_mae,
            'smile_mae': smile_mae,
            'pose_mae': pose_mae,
            'jaw_corr': jaw_corr if not np.isnan(jaw_corr) else 0.0,
            'lip_corr': lip_corr if not np.isnan(lip_corr) else 0.0,
            'smile_corr': smile_corr if not np.isnan(smile_corr) else 0.0
        }
    
    def train(self, num_epochs=10, batch_size=16, validation_split=0.2):
        """Full training pipeline"""
        print(f"\\nStarting training for {num_epochs} epochs...")
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Split into train/validation
        total_samples = len(dataset['audio'])
        val_size = int(total_samples * validation_split)
        train_size = total_samples - val_size
        
        indices = torch.randperm(total_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_dataset = {
            'audio': dataset['audio'][train_indices],
            'targets': dataset['targets'][train_indices],
            'vad': dataset['vad'][train_indices]
        }
        
        val_dataset = {
            'audio': dataset['audio'][val_indices],
            'targets': dataset['targets'][val_indices],
            'vad': dataset['vad'][val_indices]
        }
        
        # Create data loaders
        train_loader = self.create_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = self.create_data_loader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and scheduler
        total_steps = len(train_loader) * num_epochs
        self.setup_optimizer()
        self.setup_scheduler(total_steps)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_jaw_corr': [],
            'val_lip_corr': []
        }
        
        best_val_loss = float('inf')
        
        print(f"\\nTraining configuration:")
        print(f"  Total samples: {total_samples}")
        print(f"  Train samples: {train_size}")
        print(f"  Validation samples: {val_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Total steps: {total_steps}")
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['total_loss'])
            history['val_loss'].append(val_metrics['total_loss'])
            history['val_mae'].append(val_metrics['overall_mae'])
            history['val_jaw_corr'].append(val_metrics['jaw_corr'])
            history['val_lip_corr'].append(val_metrics['lip_corr'])
            
            # Print metrics
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"  Val MAE: {val_metrics['overall_mae']:.4f}")
            print(f"  Jaw Open r: {val_metrics['jaw_corr']:.3f}")
            print(f"  Lip Close r: {val_metrics['lip_corr']:.3f}")
            print(f"  Mouth MAE: {val_metrics['mouth_mae']:.4f}")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self.save_model("models/best_tcn_model.pth", val_metrics)
                print(f"  ✅ New best model saved!")
        
        print(f"\\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        return history
    
    def save_model(self, path, metrics=None):
        """Save model and training info"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_model_info(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        if metrics:
            save_dict['validation_metrics'] = metrics
        
        torch.save(save_dict, path)
        print(f"Model saved to: {path}")

def main():
    """Main training function"""
    print("=== TCN Audio-to-Blendshapes Training ===")
    
    # Create model
    model = create_model()
    
    # Create trainer
    trainer = TCNTrainer(model)
    
    # Train model
    history = trainer.train(
        num_epochs=15,        # Quick training for initial validation
        batch_size=16,        # Adjust based on GPU memory
        validation_split=0.2
    )
    
    print("\\nTraining completed successfully!")

if __name__ == "__main__":
    main()