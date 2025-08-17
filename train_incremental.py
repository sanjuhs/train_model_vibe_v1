#!/usr/bin/env python3
"""
Incremental Training Script for TCN Audio-to-Blendshapes Model
1. Run sanity check first
2. Train base model (10 epochs, base loss only)
3. Add complexity gradually (temporal, silence, pose losses)
4. Continue training with full loss
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
from sanity_check import SanityChecker

class IncrementalTrainer:
    """Incremental trainer that adds complexity gradually"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        
        print(f"🚀 Incremental Trainer on device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Loss function - start simple
        self.criterion = AudioBlendshapeLoss(
            base_weight=1.0,
            temporal_weight=0.0,    # Will enable later
            silence_weight=0.0,     # Will enable later
            pose_clamp_weight=0.0   # Will enable later
        )
        
        # Mixed precision if GPU
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ Mixed precision enabled")
        
        # Training state
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.phase = "base"  # base -> temporal -> silence -> full
        
    def load_dataset(self, data_dir):
        """Load and normalize dataset"""
        data_dir = Path(data_dir)
        print(f"Loading dataset from: {data_dir}")
        
        # Load arrays
        audio_sequences = np.load(data_dir / "audio_sequences.npy")
        target_sequences = np.load(data_dir / "target_sequences.npy") 
        vad_sequences = np.load(data_dir / "vad_sequences.npy")
        
        print(f"Loaded: Audio {audio_sequences.shape}, Targets {target_sequences.shape}")
        
        # Load and apply scalers if they exist
        import joblib
        audio_scaler_path = data_dir / "audio_scaler.pkl"
        target_scaler_path = data_dir / "target_scaler.pkl"
        
        if audio_scaler_path.exists():
            print(f"🔧 Loading audio scaler from {audio_scaler_path}")
            audio_scaler = joblib.load(audio_scaler_path)
            # Reshape for scaler application
            original_shape = audio_sequences.shape
            audio_sequences = audio_sequences.reshape(-1, original_shape[-1])
            audio_sequences = audio_scaler.transform(audio_sequences)
            audio_sequences = audio_sequences.reshape(original_shape)
            print(f"✅ Audio sequences normalized using saved scaler")
        else:
            # Fallback normalization
            print(f"🔧 No audio scaler found, using z-score normalization...")
            audio_mean = audio_sequences.mean(axis=(0, 1), keepdims=True)
            audio_std = audio_sequences.std(axis=(0, 1), keepdims=True) + 1e-6
            audio_sequences = (audio_sequences - audio_mean) / audio_std
        
        # Skip target scaler for now - it seems to be corrupted/incorrect
        print(f"⚠️ Skipping target scaler (may be corrupted - creates huge values)")
        
        # Check target ranges and apply basic normalization if needed
        print(f"Raw target range: [{target_sequences.min():.3f}, {target_sequences.max():.3f}]")
        
        # If targets are way outside expected ranges, apply basic normalization
        if target_sequences.max() > 10 or target_sequences.min() < -10:
            print(f"🔧 Targets outside reasonable range, applying basic normalization...")
            # Simple per-feature normalization to [0,1] range for blendshapes, reasonable range for pose
            blendshapes = target_sequences[:, :, :52]
            pose = target_sequences[:, :, 52:]
            
            # Normalize blendshapes to [0,1]
            bs_min = blendshapes.min(axis=(0,1), keepdims=True)
            bs_max = blendshapes.max(axis=(0,1), keepdims=True)
            blendshapes = (blendshapes - bs_min) / (bs_max - bs_min + 1e-6)
            
            # Normalize pose to reasonable range
            pose_std = pose.std(axis=(0,1), keepdims=True) + 1e-6
            pose_mean = pose.mean(axis=(0,1), keepdims=True)
            pose = (pose - pose_mean) / pose_std * 0.1  # Scale to small range
            
            target_sequences = np.concatenate([blendshapes, pose], axis=2)
            print(f"✅ Applied basic normalization to targets")
        
        # CRITICAL FIX: Store audio normalization statistics for consistent training
        self.audio_mean = audio_sequences.mean(axis=(0,1), keepdims=True)
        self.audio_std = audio_sequences.std(axis=(0,1), keepdims=True) + 1e-6
        print(f"📊 Audio normalization stats computed:")
        print(f"  Mean: {self.audio_mean.mean():.3f}, Std: {self.audio_std.mean():.3f}")
        print(f"  🔧 Will normalize audio inputs during training for stable gradients")
        
        print(f"  Audio range after norm: [{audio_sequences.min():.3f}, {audio_sequences.max():.3f}]")
        
        # Check target ranges
        blendshapes = target_sequences[:, :, :52]
        pose = target_sequences[:, :, 52:]
        print(f"  Blendshapes range: [{blendshapes.min():.3f}, {blendshapes.max():.3f}]")
        print(f"  Pose range: [{pose.min():.3f}, {pose.max():.3f}]")
        
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
    
    def setup_optimizer(self, learning_rate=1e-3, phase="base"):
        """Setup optimizer based on training phase"""
        # CRITICAL FIX: Use stable learning rates based on diagnostic findings
        if phase == "base":
            # Start with proven stable LR
            lr = learning_rate
        elif phase == "temporal":
            # Slightly lower LR when adding temporal loss
            lr = learning_rate * 0.8
        elif phase == "silence":
            # Lower LR when adding silence loss
            lr = learning_rate * 0.7
        else:  # full
            # Conservative LR for full training
            lr = learning_rate * 0.6
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
        
        # Gentler scheduler based on diagnostic findings
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.7, verbose=True, min_lr=1e-6
        )
        
        print(f"Optimizer for {phase} phase: AdamW(lr={lr:.6f})")
        print(f"Scheduler: ReduceLROnPlateau(patience=5, factor=0.7)")
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Phase: {self.phase} | Epoch {self.epoch+1}")
        
        for batch_idx, (audio, targets, vad) in enumerate(pbar):
            # Move to device
            audio = audio.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            vad = vad.to(self.device, non_blocking=True)
            
            # CRITICAL FIX: Apply consistent audio normalization
            if hasattr(self, 'audio_mean') and hasattr(self, 'audio_std'):
                audio_mean = torch.FloatTensor(self.audio_mean).to(self.device)
                audio_std = torch.FloatTensor(self.audio_std).to(self.device)
                audio = (audio - audio_mean) / audio_std
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(audio)
                    loss_dict = self.criterion(predictions, targets, vad)
                    loss = loss_dict['total_loss']
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # CRITICAL FIX: Gradient clipping for stability
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(audio)
                loss_dict = self.criterion(predictions, targets, vad)
                loss = loss_dict['total_loss']
                
                loss.backward()
                # CRITICAL FIX: Gradient clipping for stability
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.6f}",
                'grad': f"{grad_norm:.3f}"
            })
            
            # Log detailed loss components every 200 batches
            if (batch_idx + 1) % 200 == 0:
                print(f"\nBatch {batch_idx+1}: "
                      f"base={loss_dict['base_loss']:.4f}, "
                      f"temp={loss_dict['temporal_loss']:.4f}, "
                      f"sil={loss_dict['silence_loss']:.4f}, "
                      f"pose={loss_dict['pose_loss']:.4f}")
        
        # Don't step scheduler here - we'll step it in train_phase after validation
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
                
                # CRITICAL FIX: Apply same audio normalization during validation
                if hasattr(self, 'audio_mean') and hasattr(self, 'audio_std'):
                    audio_mean = torch.FloatTensor(self.audio_mean).to(self.device)
                    audio_std = torch.FloatTensor(self.audio_std).to(self.device)
                    audio = (audio - audio_mean) / audio_std
                
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
    
    def save_model(self, path, phase="unknown"):
        """Save model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'epoch': self.epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'criterion_state': {
                'base_weight': self.criterion.base_weight,
                'temporal_weight': self.criterion.temporal_weight,
                'silence_weight': self.criterion.silence_weight,
                'pose_clamp_weight': self.criterion.pose_clamp_weight
            }
        }
        
        torch.save(save_dict, path)
        print(f"Model saved to: {path}")
    
    def train_phase(self, train_loader, val_loader, num_epochs, phase_name):
        """Train for a specific phase"""
        print(f"\n🎯 Starting {phase_name} phase ({num_epochs} epochs)")
        self.phase = phase_name
        
        phase_best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.epoch += 1
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            epoch_time = time.time() - start_time
            
            # Print epoch results
            print(f"\nPhase: {phase_name} | Epoch {self.epoch}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Epoch Time: {epoch_time:.1f}s")
            # Get current learning rate (different for ReduceLROnPlateau)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Update scheduler based on validation loss
            self.scheduler.step(val_loss)
            
            # Save best model for this phase
            if val_loss < phase_best_loss:
                phase_best_loss = val_loss
                self.save_model(f"models/best_{phase_name}_model.pth", phase_name)
                print(f"  🎉 New best for {phase_name} phase! (Val Loss: {phase_best_loss:.4f})")
            
            # Update global best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model("models/best_incremental_model.pth", phase_name)
                print(f"  🏆 New global best! (Val Loss: {self.best_val_loss:.4f})")
        
        print(f"✅ {phase_name} phase completed. Best loss: {phase_best_loss:.4f}")
        return phase_best_loss
    
    def run_incremental_training(self, train_loader, val_loader):
        """Run the full incremental training pipeline"""
        print(f"\n" + "="*60)
        print(f"🚀 INCREMENTAL TRAINING PIPELINE")
        print(f"="*60)
        
        # Phase 1: Base loss only (get the fundamentals working)
        print(f"\n📊 Phase 1: Base Loss Only")
        print(f"   Focusing on basic audio-to-blendshapes mapping")
        # CRITICAL FIX: Start with proven stable learning rate
        self.setup_optimizer(learning_rate=1e-3, phase="base")
        base_loss = self.train_phase(train_loader, val_loader, 15, "base")
        
        # Check if base loss is decreasing
        if base_loss > 0.1:
            print(f"⚠️  Base loss still high ({base_loss:.4f}). Consider more base epochs.")
        
        # Phase 2: Add temporal smoothness
        print(f"\n📊 Phase 2: Add Temporal Smoothness")
        print(f"   Adding temporal consistency to predictions")
        self.criterion.enable_temporal_loss(weight=0.1)
        self.setup_optimizer(learning_rate=1e-3, phase="temporal")
        temporal_loss = self.train_phase(train_loader, val_loader, 10, "temporal")
        
        # Phase 3: Add silence weighting
        print(f"\n📊 Phase 3: Add Silence Weighting")
        print(f"   Emphasizing mouth closure during silence")
        self.criterion.enable_silence_loss(weight=1.5)
        self.setup_optimizer(learning_rate=1e-3, phase="silence")
        silence_loss = self.train_phase(train_loader, val_loader, 10, "silence")
        
        # Phase 4: Add pose regularization and full training
        print(f"\n📊 Phase 4: Full Training")
        print(f"   Adding pose regularization and long training")
        self.criterion.enable_pose_loss(weight=0.05)
        self.setup_optimizer(learning_rate=1e-3, phase="full")
        final_loss = self.train_phase(train_loader, val_loader, 25, "full")
        
        # Summary
        print(f"\n" + "="*60)
        print(f"📋 INCREMENTAL TRAINING SUMMARY")
        print(f"="*60)
        print(f"Phase 1 (Base):     {base_loss:.4f}")
        print(f"Phase 2 (Temporal): {temporal_loss:.4f}")
        print(f"Phase 3 (Silence):  {silence_loss:.4f}")
        print(f"Phase 4 (Full):     {final_loss:.4f}")
        print(f"Best Overall:       {self.best_val_loss:.4f}")
        print(f"Total Epochs:       {self.epoch}")
        
        return self.best_val_loss

def main():
    """Main incremental training function"""
    parser = argparse.ArgumentParser(description="Incremental TCN training")
    parser.add_argument("--data-dir", type=str, default="multi_video_features/combined_dataset", help="Data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip sanity check")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧠 TCN INCREMENTAL TRAINING PIPELINE")
    print("="*60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Step 1: Sanity check (unless skipped)
    if not args.skip_sanity:
        print(f"\n🧪 Step 1: Sanity Check")
        checker = SanityChecker(args.data_dir)
        sanity_passed = checker.run_full_sanity_check()
        
        if not sanity_passed:
            print(f"❌ Sanity check failed! Fix issues before proceeding.")
            return False
        
        print(f"✅ Sanity check passed! Proceeding with incremental training.")
    else:
        print(f"⏭️  Skipping sanity check (--skip-sanity)")
    
    # Step 2: Create model and trainer
    print(f"\n🏗️  Step 2: Model Setup")
    model = create_model()
    trainer = IncrementalTrainer(model)
    
    # Step 3: Load data
    print(f"\n📊 Step 3: Data Loading")
    train_dataset, val_dataset, test_dataset = trainer.load_dataset(args.data_dir)
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=args.batch_size
    )
    
    # Step 4: Incremental training
    print(f"\n🎯 Step 4: Incremental Training")
    final_loss = trainer.run_incremental_training(train_loader, val_loader)
    
    # Step 5: Final evaluation
    print(f"\n📈 Step 5: Final Evaluation")
    final_val_loss = trainer.validate(val_loader)
    test_loss = trainer.validate(test_loader)
    
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    print(f"\n🎉 Incremental training completed successfully!")
    print(f"Best model saved as: models/best_incremental_model.pth")
    
    return final_loss < 0.3  # Success if final loss < 0.3

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ Training pipeline completed successfully!")
    else:
        print(f"\n❌ Training pipeline had issues.")