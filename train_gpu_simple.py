#!/usr/bin/env python3
"""
Simple GPU Training Script - Force GPU usage with minimal complexity
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
import sys

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent))
from models.tcn_model import create_model
from training.train_tcn import AudioBlendshapeLoss, TCNTrainer

class SimpleGPUTrainer:
    """Simple GPU trainer with forced CUDA usage"""
    
    def __init__(self, model, force_gpu=True):
        # Force GPU if available
        if force_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🚀 FORCED GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print("⚠️ Using CPU")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Loss function
        self.criterion = AudioBlendshapeLoss(
            base_weight=1.0,
            temporal_weight=0.15,
            silence_weight=2.5,
            pose_clamp_weight=0.08
        )
        
        # Mixed precision if GPU
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ Mixed precision enabled")
        
        # Training state
        self.best_val_loss = float('inf')
        self.epoch = 0
        
    def load_dataset(self, data_dir):
        """Load dataset from directory"""
        data_dir = Path(data_dir)
        
        print(f"Loading dataset from: {data_dir}")
        
        # Load arrays
        audio_sequences = np.load(data_dir / "audio_sequences.npy")
        target_sequences = np.load(data_dir / "target_sequences.npy") 
        vad_sequences = np.load(data_dir / "vad_sequences.npy")
        
        print(f"Loaded: Audio {audio_sequences.shape}, Targets {target_sequences.shape}")
        
        # Convert to tensors
        audio_tensor = torch.FloatTensor(audio_sequences)
        target_tensor = torch.FloatTensor(target_sequences)
        vad_tensor = torch.FloatTensor(vad_sequences)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(audio_tensor, target_tensor, vad_tensor)
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(0.75 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        print(f"Split: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, train_dataset, val_dataset, test_dataset, batch_size=32):
        """Create data loaders"""
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=0, pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader, test_loader
    
    def setup_optimizer(self, learning_rate=2e-3):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-6
        )
        
        print(f"Optimizer: AdamW(lr={learning_rate})")
        print(f"Scheduler: CosineAnnealingLR")
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch+1}")
        
        for batch_idx, (audio, targets, vad) in enumerate(pbar):
            # Move to device
            audio = audio.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            vad = vad.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(audio)
                    loss_dict = self.criterion(predictions, targets, vad)
                    loss = loss_dict['total_loss']
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(audio)
                loss_dict = self.criterion(predictions, targets, vad)
                loss = loss_dict['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.6f}",
                'gpu': f"{gpu_mem:.1f}GB" if torch.cuda.is_available() else "N/A"
            })
        
        self.scheduler.step()
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for audio, targets, vad in tqdm(val_loader, desc="Validation"):
                audio = audio.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                vad = vad.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(audio)
                        loss_dict = self.criterion(predictions, targets, vad)
                        loss = loss_dict['total_loss']
                else:
                    predictions = self.model(audio)
                    loss_dict = self.criterion(predictions, targets, vad)
                    loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_model(self, path):
        """Save model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        torch.save(save_dict, path)
        print(f"Model saved to: {path}")
    
    def train(self, train_loader, val_loader, num_epochs=200):
        """Train the model"""
        print(f"\n🚀 Starting training for {num_epochs} epochs on {self.device}")
        print(f"Device type check: {self.device.type}")
        print(f"Model device: {next(self.model.parameters()).device}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            epoch_time = time.time() - start_time
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Epoch Time: {epoch_time:.1f}s")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model("models/best_gpu_simple_model.pth")
                print(f"  🎉 New best model! (Val Loss: {self.best_val_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f"models/checkpoint_epoch_{epoch+1}.pth")
                print(f"  💾 Checkpoint saved")
        
        print(f"\n🎉 Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple GPU training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="multi_video_features/combined_dataset", help="Data directory")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    
    args = parser.parse_args()
    
    print("=== SIMPLE GPU TRAINING ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Please check your PyTorch installation.")
        return
    
    # Create model
    model = create_model()
    
    # Create trainer (force GPU)
    trainer = SimpleGPUTrainer(model, force_gpu=True)
    
    # Load dataset
    train_dataset, val_dataset, test_dataset = trainer.load_dataset(args.data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=args.batch_size
    )
    
    # Setup optimizer
    trainer.setup_optimizer(learning_rate=args.lr)
    
    # Train
    trainer.train(train_loader, val_loader, num_epochs=args.epochs)

if __name__ == "__main__":
    main()