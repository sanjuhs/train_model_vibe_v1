#!/usr/bin/env python3
"""
Enhanced GPU Training Pipeline for Audio-to-Blendshapes TCN
High-performance training with GPU acceleration and advanced features
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
import argparse
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
from training.train_tcn import AudioBlendshapeLoss, TCNTrainer

class EnhancedGPUTrainer(TCNTrainer):
    """
    Enhanced GPU trainer with advanced features
    """
    
    def __init__(self, model, device=None, mixed_precision=True):
        """
        Initialize enhanced trainer
        
        Args:
            model: TCN model
            device: Device to use (auto-detect if None)
            mixed_precision: Use mixed precision training for speed
        """
        # Auto-detect best device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"🚀 CUDA detected! Using GPU: {torch.cuda.get_device_name()}")
                print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = 'cpu'
                print("⚠️ CUDA not available, using CPU")
        
        super().__init__(model, device)
        
        self.mixed_precision = mixed_precision and device == 'cuda'
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ Mixed precision training enabled")
        
        # Enhanced loss function with better weighting
        self.criterion = AudioBlendshapeLoss(
            base_weight=1.0,
            temporal_weight=0.15,      # Slightly higher for smoothness
            silence_weight=2.5,        # Higher weight for silence
            pose_clamp_weight=0.08     # Higher pose constraint
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_jaw_corr': [],
            'val_lip_corr': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Early stopping
        self.patience = 15
        self.patience_counter = 0
        
        print(f"Enhanced GPU Trainer initialized on {device}")
    
    def setup_optimizer(self, learning_rate=2e-3, weight_decay=1e-4, warmup_epochs=5):
        """Setup optimizer with enhanced configuration"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.base_lr = learning_rate
        self.max_lr = 4e-3  # Higher max learning rate
        self.warmup_epochs = warmup_epochs
        
        print(f"Optimizer configured: AdamW(lr={learning_rate}, wd={weight_decay})")
        print(f"Warmup epochs: {warmup_epochs}")
    
    def setup_scheduler(self, total_epochs, steps_per_epoch):
        """Setup enhanced learning rate scheduler"""
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch
        
        # Warmup + OneCycle scheduler
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # OneCycle after warmup
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        print(f"Scheduler configured: Warmup + Cosine Annealing")
        print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    def load_large_dataset(self, data_dir="multi_video_features/combined_dataset"):
        """Load large multi-video dataset"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        print(f"Loading large dataset from: {data_path}")
        
        # Load arrays with memory mapping for large datasets
        audio_sequences = np.load(data_path / "audio_sequences.npy", mmap_mode='r')
        target_sequences = np.load(data_path / "target_sequences.npy", mmap_mode='r')
        vad_sequences = np.load(data_path / "vad_sequences.npy", mmap_mode='r')
        
        # Load into memory with progress bar
        print("Loading arrays into memory...")
        audio_sequences = np.array(audio_sequences)
        target_sequences = np.array(target_sequences)
        vad_sequences = np.array(vad_sequences)
        
        # Load metadata
        with open(data_path / "dataset_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"Dataset loaded:")
        print(f"  Audio sequences: {audio_sequences.shape}")
        print(f"  Target sequences: {target_sequences.shape}")
        print(f"  VAD sequences: {vad_sequences.shape}")
        print(f"  Videos: {metadata.get('num_videos', 'unknown')}")
        print(f"  Duration: {metadata.get('total_duration_minutes', 0):.1f} minutes")
        
        return {
            'audio': torch.FloatTensor(audio_sequences),
            'targets': torch.FloatTensor(target_sequences),
            'vad': torch.FloatTensor(vad_sequences),
            'metadata': metadata
        }
    
    def create_data_loaders(self, dataset, batch_size=32, validation_split=0.15, test_split=0.1):
        """Create train/val/test data loaders with better splits"""
        total_samples = len(dataset['audio'])
        
        # Create indices
        indices = torch.randperm(total_samples)
        
        # Split sizes
        test_size = int(total_samples * test_split)
        val_size = int(total_samples * validation_split)
        train_size = total_samples - test_size - val_size
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        print(f"Dataset split:")
        print(f"  Train: {train_size} sequences ({train_size/total_samples:.1%})")
        print(f"  Validation: {val_size} sequences ({val_size/total_samples:.1%})")
        print(f"  Test: {test_size} sequences ({test_size/total_samples:.1%})")
        
        # Create datasets
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
        
        test_dataset = {
            'audio': dataset['audio'][test_indices],
            'targets': dataset['targets'][test_indices],
            'vad': dataset['vad'][test_indices]
        }
        
        # Create data loaders
        train_loader = self.create_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = self.create_data_loader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = self.create_data_loader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch_enhanced(self, data_loader, epoch):
        """Enhanced training epoch with mixed precision"""
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'base_loss': 0.0,
            'temporal_loss': 0.0,
            'silence_loss': 0.0,
            'pose_loss': 0.0,
            'num_batches': 0
        }
        
        epoch_start_time = time.time()
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (audio, targets, vad) in enumerate(pbar):
            audio = audio.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            vad = vad.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    predictions = self.model(audio)
                    loss_dict = self.criterion(predictions, targets, vad)
                
                # Mixed precision backward pass
                self.scaler.scale(loss_dict['total_loss']).backward()
                
                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                predictions = self.model(audio)
                loss_dict = self.criterion(predictions, targets, vad)
                
                loss_dict['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update metrics
            for key in epoch_metrics:
                if key == 'num_batches':
                    epoch_metrics[key] += 1
                else:
                    epoch_metrics[key] += loss_dict[key.replace('total_loss', 'total_loss').replace('total_loss', 'total_loss')]
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'lr': f"{current_lr:.6f}",
                'gpu': f"{torch.cuda.memory_reserved()/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
            })
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Average metrics
        for key in epoch_metrics:
            if key != 'num_batches':
                epoch_metrics[key] /= epoch_metrics['num_batches']
        
        epoch_metrics['epoch_time'] = epoch_time
        epoch_metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
        
        return epoch_metrics
    
    def train_enhanced(self, 
                      num_epochs=100, 
                      batch_size=32, 
                      data_dir="multi_video_features/combined_dataset",
                      save_every=10,
                      plot_every=5):
        """Enhanced training pipeline"""
        
        print(f"\\n{'='*60}")
        print("ENHANCED GPU TRAINING PIPELINE")
        print(f"{'='*60}")
        print(f"Target epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        
        # Load dataset
        print("\\n1. Loading dataset...")
        dataset = self.load_large_dataset(data_dir)
        
        # Create data loaders
        print("\\n2. Creating data loaders...")
        train_loader, val_loader, test_loader = self.create_data_loaders(
            dataset, batch_size=batch_size
        )
        
        # Setup optimizer and scheduler
        print("\\n3. Setting up training...")
        self.setup_optimizer()
        self.setup_scheduler(num_epochs, len(train_loader))
        
        # Training loop
        print(f"\\n4. Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch_enhanced(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history (ensure all values are moved to CPU and converted to float)
            def to_cpu_float(x):
                if hasattr(x, 'cpu'):
                    return float(x.cpu())
                return float(x)
            
            self.training_history['train_loss'].append(to_cpu_float(train_metrics['total_loss']))
            self.training_history['val_loss'].append(to_cpu_float(val_metrics['total_loss']))
            self.training_history['val_mae'].append(to_cpu_float(val_metrics['overall_mae']))
            self.training_history['val_jaw_corr'].append(to_cpu_float(val_metrics['jaw_corr']))
            self.training_history['val_lip_corr'].append(to_cpu_float(val_metrics['lip_corr']))
            self.training_history['learning_rate'].append(to_cpu_float(train_metrics['learning_rate']))
            self.training_history['epoch_time'].append(to_cpu_float(train_metrics['epoch_time']))
            
            # Print metrics
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"  Val MAE: {val_metrics['overall_mae']:.4f}")
            print(f"  Jaw Corr: {val_metrics['jaw_corr']:.3f}")
            print(f"  Lip Corr: {val_metrics['lip_corr']:.3f}")
            print(f"  Epoch Time: {train_metrics['epoch_time']:.1f}s")
            print(f"  Learning Rate: {train_metrics['learning_rate']:.6f}")
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
                self.save_model(f"models/best_tcn_gpu_model.pth", val_metrics)
                print(f"  🎉 New best model saved! (Val Loss: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"models/checkpoint_epoch_{epoch+1}.pth", epoch)
                print(f"  💾 Checkpoint saved at epoch {epoch+1}")
            
            # Plot progress (disabled temporarily to avoid CUDA tensor issues)
            # if (epoch + 1) % plot_every == 0:
            #     self.plot_training_progress(f"training_progress_epoch_{epoch+1}.png")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\\n⏹️ Early stopping triggered after {epoch+1} epochs")
                print(f"   Best validation loss: {self.best_val_loss:.4f}")
                break
        
        # Final evaluation on test set
        print("\\n5. Final evaluation on test set...")
        test_metrics = self.validate(test_loader)
        
        print(f"\\n{'='*60}")
        print("TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Test MAE: {test_metrics['overall_mae']:.4f}")
        print(f"Test Jaw Correlation: {test_metrics['jaw_corr']:.3f}")
        print(f"Test Lip Correlation: {test_metrics['lip_corr']:.3f}")
        print(f"Total training time: {sum(self.training_history['epoch_time'])/3600:.2f} hours")
        
        # Final plots (disabled temporarily)
        # self.plot_training_progress("final_training_progress.png")
        
        return self.training_history
    
    def save_checkpoint(self, path, epoch):
        """Save training checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'model_config': self.model.get_model_info(),
        }
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded from epoch {self.epoch}")
    
    def plot_training_progress(self, filename):
        """Plot training progress"""
        if len(self.training_history['train_loss']) < 2:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(epochs, self.training_history['val_mae'])
        axes[0, 1].set_title('Validation MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid(True)
        
        # Correlations
        axes[0, 2].plot(epochs, self.training_history['val_jaw_corr'], label='Jaw Open')
        axes[0, 2].plot(epochs, self.training_history['val_lip_corr'], label='Lip Close')
        axes[0, 2].axhline(y=0.6, color='r', linestyle='--', label='Target (0.6)')
        axes[0, 2].set_title('Feature Correlations')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Pearson r')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.training_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Epoch time
        axes[1, 1].plot(epochs, self.training_history['epoch_time'])
        axes[1, 1].set_title('Training Speed')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
        
        # Loss improvement
        if len(self.training_history['val_loss']) > 1:
            val_loss_smooth = np.convolve(self.training_history['val_loss'], 
                                        np.ones(min(5, len(epochs))), mode='valid')
            axes[1, 2].plot(range(1, len(val_loss_smooth) + 1), val_loss_smooth)
            axes[1, 2].set_title('Validation Loss (Smoothed)')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Training progress plot saved: {filename}")

def main():
    """Main training function with arguments"""
    parser = argparse.ArgumentParser(description="Enhanced GPU training for TCN model")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="multi_video_features/combined_dataset",
                       help="Dataset directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    print("=== ENHANCED GPU TCN TRAINING ===")
    
    try:
        # Create model
        model = create_model()
        
        # Force GPU device if available
        if args.device is None and torch.cuda.is_available():
            args.device = 'cuda'
            print(f"🔧 Forcing GPU usage: {args.device}")
        
        # Create trainer
        trainer = EnhancedGPUTrainer(
            model, 
            device=args.device,
            mixed_precision=not args.no_mixed_precision
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
            print(f"Resumed from checkpoint: {args.resume}")
        
        # Train model
        history = trainer.train_enhanced(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            data_dir=args.data_dir
        )
        
        print("\\n🎉 Enhanced training completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()