#!/usr/bin/env python3
"""
Pure Sanity Check - Strip ALL regularization to achieve perfect overfitting
Tests if model can drive loss to near 0 on tiny batch without any obstacles
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to import models
sys.path.append(str(Path(__file__).parent))
from models.tcn_model import create_model

class PureSanityChecker:
    """
    Pure sanity checker with ALL regularization removed
    """
    def __init__(self, data_dir="multi_video_features/combined_dataset"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔍 PURE Sanity Check on device: {self.device}")
        
        # Load tiny dataset
        self.load_tiny_dataset(data_dir)
        
        # Create model and STRIP ALL REGULARIZATION
        self.model = create_model().to(self.device)
        self.strip_regularization()
        
        print(f"✅ Pure sanity checker initialized (NO regularization)")
    
    def load_tiny_dataset(self, data_dir, tiny_size=128):
        """Load a tiny subset for pure overfitting test"""
        data_dir = Path(data_dir)
        
        print(f"Loading tiny dataset from: {data_dir}")
        
        # Load full arrays
        audio_sequences = np.load(data_dir / "audio_sequences.npy")
        target_sequences = np.load(data_dir / "target_sequences.npy") 
        vad_sequences = np.load(data_dir / "vad_sequences.npy")
        
        print(f"Full dataset: Audio {audio_sequences.shape}, Targets {target_sequences.shape}")
        
        # Load and apply scalers if they exist
        import joblib
        audio_scaler_path = data_dir / "audio_scaler.pkl"
        
        if audio_scaler_path.exists():
            print(f"🔧 Loading audio scaler from {audio_scaler_path}")
            audio_scaler = joblib.load(audio_scaler_path)
            # Reshape for scaler application
            original_shape = audio_sequences.shape
            audio_sequences = audio_sequences.reshape(-1, original_shape[-1])
            audio_sequences = audio_scaler.transform(audio_sequences)
            audio_sequences = audio_sequences.reshape(original_shape)
            print(f"✅ Audio sequences normalized using saved scaler")
        
        # Skip target scaler and apply basic normalization
        print(f"⚠️ Skipping target scaler (may be corrupted)")
        print(f"Raw target range: [{target_sequences.min():.3f}, {target_sequences.max():.3f}]")
        
        # If targets are way outside expected ranges, apply basic normalization
        if target_sequences.max() > 10 or target_sequences.min() < -10:
            print(f"🔧 Targets outside reasonable range, applying basic normalization...")
            blendshapes = target_sequences[:, :, :52]
            pose = target_sequences[:, :, 52:]
            
            # Normalize blendshapes to [0,1]
            bs_min = blendshapes.min(axis=(0,1), keepdims=True)
            bs_max = blendshapes.max(axis=(0,1), keepdims=True)
            blendshapes = (blendshapes - bs_min) / (bs_max - bs_min + 1e-6)
            
            # Normalize pose to reasonable range
            pose_std = pose.std(axis=(0,1), keepdims=True) + 1e-6
            pose_mean = pose.mean(axis=(0,1), keepdims=True)
            pose = (pose - pose_mean) / pose_std * 0.1
            
            target_sequences = np.concatenate([blendshapes, pose], axis=2)
            print(f"✅ Applied basic normalization to targets")
        
        # Take tiny subset
        tiny_indices = np.random.choice(len(audio_sequences), size=tiny_size, replace=False)
        audio_tiny = audio_sequences[tiny_indices]
        target_tiny = target_sequences[tiny_indices]
        vad_tiny = vad_sequences[tiny_indices]
        
        print(f"Tiny dataset: Audio {audio_tiny.shape}, Targets {target_tiny.shape}")
        print(f"  Audio range: [{audio_tiny.min():.3f}, {audio_tiny.max():.3f}]")
        print(f"  Target range: [{target_tiny.min():.3f}, {target_tiny.max():.3f}]")
        
        # Check target ranges
        blendshapes_tiny = target_tiny[:, :, :52]
        pose_tiny = target_tiny[:, :, 52:]
        print(f"  Blendshapes range: [{blendshapes_tiny.min():.3f}, {blendshapes_tiny.max():.3f}]")
        print(f"  Pose range: [{pose_tiny.min():.3f}, {pose_tiny.max():.3f}]")
        
        # Convert to tensors
        self.audio_tiny = torch.FloatTensor(audio_tiny).to(self.device)
        self.target_tiny = torch.FloatTensor(target_tiny).to(self.device)
        self.vad_tiny = torch.FloatTensor(vad_tiny).to(self.device)
        
        print(f"✅ Tiny dataset loaded: {len(audio_tiny)} samples")
    
    def strip_regularization(self):
        """Remove ALL regularization from the model"""
        print(f"\n🚫 STRIPPING ALL REGULARIZATION:")
        
        # 1. Replace output activations with Identity AND remove scaling
        print(f"   1. Removing output activations (Sigmoid/Tanh → Identity)")
        self.model.blendshape_activation = nn.Identity()
        self.model.pose_activation = nn.Identity()
        
        # Monkey-patch the forward method to remove pose scaling
        original_forward = self.model.forward
        def forward_no_scaling(x):
            # Copy the original forward but without pose scaling
            if x.dim() == 3 and x.size(-1) == self.model.input_dim:
                x = x.transpose(1, 2)
            
            assert x.shape[1] == self.model.input_dim, f"Expected input dim {self.model.input_dim}, got {x.shape[1]}"
            
            x = self.model.input_conv(x)
            
            for tcn_layer in self.model.tcn_layers:
                x = tcn_layer(x)
            
            x = self.model.output_layers(x)
            
            # Apply activations but NO SCALING
            blendshapes = self.model.blendshape_activation(x[:, :52, :])
            pose = self.model.pose_activation(x[:, 52:, :])  # NO * 0.2 scaling
            
            x = torch.cat([blendshapes, pose], dim=1)
            
            return x.transpose(1, 2)
        
        self.model.forward = forward_no_scaling
        
        # 1.5. Rescale the final layer weights to reasonable range
        print(f"   1.5. Rescaling final layer weights to match target range")
        with torch.no_grad():
            # Get the final conv layer
            final_conv = None
            for module in self.model.output_layers:
                if isinstance(module, nn.Conv1d):
                    final_conv = module
            
            if final_conv is not None:
                # Scale weights and bias to much smaller values
                final_conv.weight.data *= 0.01  # Scale down by 100x
                if final_conv.bias is not None:
                    final_conv.bias.data *= 0.01
                print(f"      - Final layer weights scaled down by 100x")
        
        # 2. Set all dropout to 0
        print(f"   2. Disabling all dropout (p → 0.0)")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
                print(f"      - {name}: dropout disabled")
        
        # 3. Put all BatchNorm in eval mode (so they don't update)
        print(f"   3. Freezing BatchNorm layers (eval mode)")
        bn_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()
                bn_count += 1
        print(f"      - {bn_count} BatchNorm layers frozen")
        
        # 4. Put model in train mode (except for frozen BN)
        self.model.train()
        
        # Re-freeze the BN layers after train()
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()
        
        print(f"✅ ALL regularization removed!")
    
    def test_pure_overfit(self, target_loss=1e-4, max_iterations=1000, learning_rate=1e-2):
        """
        Pure overfitting test - should drive loss to near 0
        """
        print(f"\n🎯 PURE OVERFITTING TEST")
        print(f"Target loss: {target_loss} (near perfect)")
        print(f"Max iterations: {max_iterations}")
        print(f"Learning rate: {learning_rate} (constant)")
        
        # Simple MSE loss (no fancy components)
        criterion = nn.MSELoss()
        
        # Simple optimizer with constant learning rate
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.0)
        
        losses = []
        
        print(f"\n📈 Starting pure overfitting...")
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(self.audio_tiny)
            loss = criterion(predictions, self.target_tiny)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (light)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            loss_val = loss.item()
            losses.append(loss_val)
            
            # Print progress
            if iteration % 50 == 0 or iteration < 10:
                print(f"Iter {iteration:4d}: Loss = {loss_val:.8f}, Grad Norm = {grad_norm:.6f}")
            
            # Check if target reached
            if loss_val < target_loss:
                print(f"🎉 SUCCESS! Reached target loss {loss_val:.8f} < {target_loss} in {iteration+1} iterations")
                break
            
            # Check for stalling
            if iteration > 100 and len(losses) > 50:
                recent_losses = losses[-50:]
                if max(recent_losses) - min(recent_losses) < 1e-8:
                    print(f"⚠️  Loss plateaued at {loss_val:.8f} (stalled for 50 iterations)")
                    break
        else:
            print(f"❌ Did not reach target loss. Final loss: {loss_val:.8f}")
        
        return losses, loss_val < target_loss
    
    def analyze_bottleneck(self):
        """Analyze what might be causing the bottleneck"""
        print(f"\n🔍 BOTTLENECK ANALYSIS:")
        
        # Test 1: Check output ranges
        with torch.no_grad():
            predictions = self.model(self.audio_tiny)
            
            print(f"   📊 Output analysis:")
            print(f"      - Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
            print(f"      - Target range: [{self.target_tiny.min():.6f}, {self.target_tiny.max():.6f}]")
            
            # Check individual components
            pred_bs = predictions[:, :, :52]
            pred_pose = predictions[:, :, 52:]
            target_bs = self.target_tiny[:, :, :52]
            target_pose = self.target_tiny[:, :, 52:]
            
            print(f"      - Pred blendshapes: [{pred_bs.min():.6f}, {pred_bs.max():.6f}]")
            print(f"      - Target blendshapes: [{target_bs.min():.6f}, {target_bs.max():.6f}]")
            print(f"      - Pred pose: [{pred_pose.min():.6f}, {pred_pose.max():.6f}]")
            print(f"      - Target pose: [{target_pose.min():.6f}, {target_pose.max():.6f}]")
        
        # Test 2: Check gradient flow
        print(f"   🔄 Gradient analysis:")
        predictions = self.model(self.audio_tiny)
        loss = nn.MSELoss()(predictions, self.target_tiny)
        loss.backward()
        
        total_grad_norm = 0.0
        zero_grad_layers = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_grad_norm ** 2
                if param_grad_norm < 1e-8:
                    zero_grad_layers += 1
            else:
                print(f"      ❌ No gradient: {name}")
        
        total_grad_norm = total_grad_norm ** 0.5
        print(f"      - Total gradient norm: {total_grad_norm:.6f}")
        print(f"      - Layers with ~zero gradients: {zero_grad_layers}")
        
        self.model.zero_grad()
    
    def plot_pure_overfit_curve(self, losses, save_path="pure_sanity_curve.png"):
        """Plot the pure overfitting curve"""
        plt.figure(figsize=(12, 8))
        
        # Main plot
        plt.subplot(2, 1, 1)
        plt.plot(losses)
        plt.title('Pure Sanity Check: Overfitting Tiny Batch (NO Regularization)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        
        # Zoomed plot (if we have enough data)
        if len(losses) > 100:
            plt.subplot(2, 1, 2)
            plt.plot(losses[50:])  # Skip first 50 iterations
            plt.title('Zoomed View (after iteration 50)')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📊 Pure overfitting curve saved to: {save_path}")
    
    def run_pure_sanity_check(self):
        """Run complete pure sanity check"""
        print(f"\n" + "="*60)
        print(f"🔥 PURE SANITY CHECK (NO REGULARIZATION)")
        print(f"="*60)
        
        # Analyze potential bottlenecks
        self.analyze_bottleneck()
        
        # Test pure overfitting
        losses, success = self.test_pure_overfit()
        
        # Plot results
        self.plot_pure_overfit_curve(losses)
        
        # Summary
        print(f"\n" + "="*60)
        print(f"📋 PURE SANITY CHECK SUMMARY")
        print(f"="*60)
        print(f"📊 Initial loss: {losses[0]:.8f}")
        print(f"📊 Final loss: {losses[-1]:.8f}")
        print(f"📊 Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
        print(f"📊 Min loss achieved: {min(losses):.8f}")
        
        if success:
            print(f"\n🎉 PURE SANITY CHECK PASSED!")
            print(f"✅ Model can memorize perfectly with no regularization")
            print(f"💡 Now we know the plateau was caused by regularization")
        else:
            final_loss = losses[-1]
            if final_loss < 1e-3:
                print(f"\n⚠️  CLOSE TO SUCCESS (loss: {final_loss:.8f})")
                print(f"💡 Model can nearly memorize - very minor bottleneck remains")
            elif final_loss < 1e-2:
                print(f"\n⚠️  PARTIAL SUCCESS (loss: {final_loss:.8f})")
                print(f"💡 Model learning but hitting some constraint")
            else:
                print(f"\n❌ PURE SANITY CHECK FAILED!")
                print(f"🔧 There's a fundamental issue with model/data/loss")
        
        return success, losses[-1]

def main():
    """Main pure sanity check function"""
    print(f"🔥 PURE TCN Model Sanity Check")
    print(f"This will test if the model can memorize perfectly with NO regularization")
    
    # Create pure sanity checker
    checker = PureSanityChecker()
    
    # Run pure sanity check
    success, final_loss = checker.run_pure_sanity_check()
    
    if success:
        print(f"\n✅ Perfect memorization achieved!")
        print(f"🔧 Next: Add back regularization one piece at a time")
    elif final_loss < 1e-3:
        print(f"\n⚠️  Very close to perfect memorization")
        print(f"🔧 Minor tweaks needed")
    else:
        print(f"\n❌ Cannot memorize even without regularization")
        print(f"🔧 Check model architecture, data, or loss function")
    
    return success

if __name__ == "__main__":
    main()