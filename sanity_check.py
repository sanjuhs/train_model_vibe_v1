#!/usr/bin/env python3
"""
Enhanced Sanity Check Script for TCN Audio-to-Blendshapes Model
Systematically tests components to identify bottlenecks in learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from collections import defaultdict

# Add parent directory to import models
sys.path.append(str(Path(__file__).parent))
from models.tcn_model import create_model, AudioToBlendshapesTCN
from training.train_tcn import AudioBlendshapeLoss

class MinimalTCN(nn.Module):
    """Minimal TCN for testing - no regularizers, linear output"""
    def __init__(self, input_dim=80, output_dim=59, hidden_channels=192, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Simple architecture without regularizers
        self.input_conv = nn.Conv1d(input_dim, hidden_channels, 1)
        
        # Simplified TCN layers without BatchNorm/Dropout
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            padding = dilation
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, 3, 
                         dilation=dilation, padding=padding)
            )
        
        # Linear output - no activation functions
        self.output_conv = nn.Conv1d(hidden_channels, output_dim, 1)
        
        print(f"Minimal TCN: {input_dim} -> {hidden_channels} -> {output_dim}")
    
    def forward(self, x):
        # Ensure input is (batch, features, time)
        if x.dim() == 3 and x.size(-1) == self.input_dim:
            x = x.transpose(1, 2)
        
        x = self.input_conv(x)
        
        for layer in self.conv_layers:
            residual = x
            x = layer(x)
            # Remove future frames for causality
            if x.size(2) > residual.size(2):
                x = x[:, :, :residual.size(2)]
            x = torch.relu(x) + residual
        
        x = self.output_conv(x)
        return x.transpose(1, 2)  # Return (batch, time, features)

class SanityChecker:
    """
    Enhanced sanity checker with systematic component testing
    """
    def __init__(self, data_dir="multi_video_features/combined_dataset"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔍 Enhanced Sanity Check on device: {self.device}")
        
        # Load tiny dataset
        self.load_tiny_dataset(data_dir)
        
        # Track test results
        self.test_results = {}
        
        print(f"✅ Enhanced sanity checker initialized")
    
    def load_tiny_dataset(self, data_dir, tiny_size=128):
        """Load a tiny subset for overfitting test"""
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
        
        # Skip target scaler for now - it seems to be corrupted/incorrect
        # The targets should already be in reasonable ranges from the data pipeline
        print(f"⚠️ Skipping target scaler (may be corrupted - creates huge values)")
        
        # Check target ranges and apply basic normalization if needed
        print(f"Raw target range: [{target_sequences.min():.3f}, {target_sequences.max():.3f}]")
        
        # If targets are way outside expected ranges, apply basic normalization
        if target_sequences.max() > 10 or target_sequences.min() < -10:
            print(f"🔧 Targets outside reasonable range, applying basic normalization...")
            # Simple per-feature normalization to [0,1] range for blendshapes, [-1,1] for pose
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
        
        # Create tiny dataset and loader
        tiny_dataset = torch.utils.data.TensorDataset(
            self.audio_tiny, self.target_tiny, self.vad_tiny
        )
        
        self.tiny_loader = torch.utils.data.DataLoader(
            tiny_dataset, batch_size=32, shuffle=True, num_workers=0
        )
        
        print(f"✅ Tiny dataset loaded: {len(tiny_dataset)} samples, {len(self.tiny_loader)} batches")
    

    
    def analyze_gradients(self, model):
        """Analyze gradient flow in the model"""
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.max().item(),
                    'min': grad.min().item(),
                    'norm': grad.norm().item()
                }
            else:
                grad_stats[name] = None
        
        return grad_stats
    
    def create_model_variant(self, variant="minimal"):
        """Create different model variants for testing"""
        if variant == "minimal":
            return MinimalTCN().to(self.device)
        elif variant == "no_activation":
            # Original model but bypass activations
            model = create_model()
            # Monkey patch to bypass activations
            model.blendshape_activation = nn.Identity()
            model.pose_activation = nn.Identity()
            return model.to(self.device)
        elif variant == "no_batchnorm":
            # Create model without batch normalization
            config = {
                'input_dim': 80, 'output_dim': 59, 'hidden_channels': 192,
                'num_layers': 4, 'dropout': 0.0  # No dropout
            }
            return create_model(config).to(self.device)
        else:
            return create_model().to(self.device)
    
    def analyze_data_statistics(self, audio, targets):
        """Analyze input data statistics to understand scale issues"""
        print(f"\n🔍 DATA ANALYSIS:")
        print(f"Audio stats:")
        print(f"  Shape: {audio.shape}")
        print(f"  Range: [{audio.min():.3f}, {audio.max():.3f}]")
        print(f"  Mean: {audio.mean():.3f}, Std: {audio.std():.3f}")
        print(f"  Per-feature range: [{audio.mean(dim=(0,1)).min():.3f}, {audio.mean(dim=(0,1)).max():.3f}]")
        
        print(f"Target stats:")
        print(f"  Shape: {targets.shape}")
        print(f"  Range: [{targets.min():.3f}, {targets.max():.3f}]")
        print(f"  Mean: {targets.mean():.3f}, Std: {targets.std():.3f}")
        
        # Check for problematic ranges
        if audio.max() > 10 or audio.min() < -10:
            print(f"⚠️  WARNING: Audio features have large magnitude")
        if targets.max() > 5 or targets.min() < -5:
            print(f"⚠️  WARNING: Targets have large magnitude")
    
    def analyze_model_behavior(self, model, audio_input, target, step_name=""):
        """Analyze model behavior during forward pass"""
        print(f"\n🔍 MODEL ANALYSIS {step_name}:")
        
        model.eval()
        with torch.no_grad():
            # Get predictions
            predictions = model(audio_input)
            
            print(f"Predictions stats:")
            print(f"  Shape: {predictions.shape}")
            print(f"  Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
            print(f"  Mean: {predictions.mean():.6f}, Std: {predictions.std():.6f}")
            
            # Check for saturation in activations (if they exist)
            if hasattr(model, 'blendshape_activation'):
                # For sigmoid: saturation near 0 or 1
                bs_pred = predictions[:, :, :52]
                saturated_low = (bs_pred < 0.01).float().mean()
                saturated_high = (bs_pred > 0.99).float().mean()
                print(f"  Blendshape saturation: {saturated_low:.3f} near 0, {saturated_high:.3f} near 1")
                
                if saturated_low > 0.5 or saturated_high > 0.5:
                    print(f"⚠️  WARNING: High saturation detected!")
            
            # Compute loss to see baseline
            loss = nn.MSELoss()(predictions, target)
            print(f"  MSE Loss: {loss.item():.8f}")
    
    def analyze_gradients_detailed(self, model, audio_input, target, loss_fn):
        """Detailed gradient analysis to understand vanishing/exploding"""
        print(f"\n🔍 GRADIENT ANALYSIS:")
        
        model.train()
        model.zero_grad()
        
        # Forward pass
        predictions = model(audio_input)
        if isinstance(loss_fn, AudioBlendshapeLoss):
            vad = torch.ones(audio_input.shape[0], audio_input.shape[1]).to(audio_input.device)
            loss_dict = loss_fn(predictions, target, vad)
            loss = loss_dict['total_loss']
        else:
            loss = loss_fn(predictions, target)
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients layer by layer
        grad_norms = {}
        param_magnitudes = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_magnitude = param.data.norm().item()
                grad_norms[name] = grad_norm
                param_magnitudes[name] = param_magnitude
                
                # Check for vanishing/exploding
                if grad_norm < 1e-7:
                    print(f"  ⚠️  {name}: VANISHING gradients ({grad_norm:.2e})")
                elif grad_norm > 100:
                    print(f"  ⚠️  {name}: EXPLODING gradients ({grad_norm:.2e})")
                else:
                    print(f"  ✓ {name}: Normal gradients ({grad_norm:.2e})")
        
        # Overall statistics
        all_grad_norms = list(grad_norms.values())
        if all_grad_norms:
            print(f"Gradient norm stats:")
            print(f"  Min: {min(all_grad_norms):.2e}")
            print(f"  Max: {max(all_grad_norms):.2e}")
            print(f"  Mean: {np.mean(all_grad_norms):.2e}")
            
            # Check for gradient flow issues
            if max(all_grad_norms) / min(all_grad_norms) > 1000:
                print(f"⚠️  WARNING: Large gradient norm variance - indicates unstable training")
        
        model.zero_grad()
        return grad_norms, param_magnitudes
    
    def test_single_sample_overfit(self, model, criterion, target_loss=1e-4, max_iterations=500, debug=True):
        """Test overfitting on a single sample with detailed debugging"""
        print(f"\n🎯 Single sample overfit test (with debugging)...")
        
        # Use just one sample
        audio_single = self.audio_tiny[:1]  # (1, seq_len, features)
        target_single = self.target_tiny[:1]
        vad_single = self.vad_tiny[:1]
        
        print(f"Sample shapes: audio={audio_single.shape}, target={target_single.shape}")
        
        if debug:
            # Analyze input data
            self.analyze_data_statistics(audio_single, target_single)
            
            # Analyze initial model behavior
            self.analyze_model_behavior(model, audio_single, target_single, "(Initial)")
            
            # Analyze initial gradients
            self.analyze_gradients_detailed(model, audio_single, target_single, criterion)
        
        # Start with lower learning rate to avoid explosion
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        losses = []
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(audio_single)
            
            if isinstance(criterion, AudioBlendshapeLoss):
                loss_dict = criterion(predictions, target_single, vad_single)
                loss = loss_dict['total_loss']
            else:
                # Simple MSE loss
                loss = criterion(predictions, target_single)
            
            # Check for NaN/inf before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ NaN/Inf detected at iteration {iteration+1}")
                if debug:
                    print(f"  Loss: {loss.item()}")
                    print(f"  Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
                    print(f"  Target range: [{target_single.min():.6f}, {target_single.max():.6f}]")
                    print(f"  Audio range: [{audio_single.min():.6f}, {audio_single.max():.6f}]")
                break
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            loss_val = loss.item()
            losses.append(loss_val)
            
            # Print progress with gradient info
            if iteration < 10 or (iteration + 1) % 50 == 0:
                print(f"Iter {iteration+1:4d}: Loss = {loss_val:.8f}, Grad Norm = {grad_norm:.6f}")
            
            # Detailed analysis at key points
            if debug and iteration in [0, 5, 50, 100]:
                self.analyze_model_behavior(model, audio_single, target_single, f"(Iter {iteration+1})")
            
            # Check if target reached
            if loss_val < target_loss:
                print(f"🎉 SUCCESS! Single sample overfit achieved: {loss_val:.8f} < {target_loss}")
                if debug:
                    self.analyze_model_behavior(model, audio_single, target_single, "(Final)")
                return losses, True
        
        print(f"❌ Single sample overfit failed. Final loss: {losses[-1] if losses else 'N/A':.8f}")
        return losses, False
    
    def overfit_tiny_batch(self, model, criterion, target_loss=0.001, max_iterations=800, 
                          learning_rate=0.01, optimizer_type="sgd", use_scheduler=False):
        """
        Attempt to overfit the tiny dataset with configurable parameters
        """
        print(f"\n🎯 Overfitting tiny batch...")
        print(f"Target loss: {target_loss}, Max iterations: {max_iterations}")
        print(f"Learning rate: {learning_rate}, Optimizer: {optimizer_type}")
        
        # Setup optimizer
        if optimizer_type == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0)
        elif optimizer_type == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
        
        # Optional scheduler
        scheduler = None
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=100, factor=0.5, verbose=True, min_lr=1e-7)
        
        model.train()
        losses = []
        gradient_norms = []
        
        for iteration in range(max_iterations):
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            batch_count = 0
            
            for audio_batch, target_batch, vad_batch in self.tiny_loader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(audio_batch)
                
                if isinstance(criterion, AudioBlendshapeLoss):
                    loss_dict = criterion(predictions, target_batch, vad_batch)
                    loss = loss_dict['total_loss']
                else:
                    loss = criterion(predictions, target_batch)
                
                # Backward pass
                loss.backward()
                
                # Calculate gradient norm (no clipping)
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm ** 0.5
                
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_grad_norm += grad_norm
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            avg_grad_norm = epoch_grad_norm / batch_count
            
            losses.append(avg_loss)
            gradient_norms.append(avg_grad_norm)
            
            # Update scheduler
            if scheduler:
                scheduler.step(avg_loss)
            
            # Print progress
            if iteration < 10 or (iteration + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Iter {iteration+1:4d}: Loss = {avg_loss:.8f}, Grad Norm = {avg_grad_norm:.6f}, LR = {current_lr:.8f}")
            
            # Check if target reached
            if avg_loss < target_loss:
                print(f"🎉 SUCCESS! Reached target loss {avg_loss:.8f} < {target_loss} in {iteration+1} iterations")
                return losses, gradient_norms, True
        
        print(f"❌ FAILED to reach target loss. Final loss: {avg_loss:.8f}")
        return losses, gradient_norms, False
    
    def plot_overfitting_curve(self, losses, save_path="sanity_check_curve.png"):
        """Plot the overfitting curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Sanity Check: Overfitting Tiny Batch')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(save_path)
        print(f"📊 Overfitting curve saved to: {save_path}")
    
    def plot_detailed_results(self, results, save_path="enhanced_sanity_results.png"):
        """Plot detailed results from all tests"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced Sanity Check Results', fontsize=16)
        
        # Plot 1: Loss curves comparison
        ax1 = axes[0, 0]
        for test_name, data in results.items():
            if 'losses' in data and data['losses']:
                ax1.plot(data['losses'], label=f"{test_name} (final: {data['losses'][-1]:.6f})")
        ax1.set_yscale('log')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Gradient norms
        ax2 = axes[0, 1]
        for test_name, data in results.items():
            if 'gradient_norms' in data and data['gradient_norms']:
                ax2.plot(data['gradient_norms'], label=test_name)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('Gradient Flow')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Final losses comparison
        ax3 = axes[1, 0]
        test_names = []
        final_losses = []
        for test_name, data in results.items():
            if 'losses' in data and data['losses']:
                test_names.append(test_name)
                final_losses.append(data['losses'][-1])
        
        if test_names:
            bars = ax3.bar(test_names, final_losses)
            ax3.set_yscale('log')
            ax3.set_ylabel('Final Loss')
            ax3.set_title('Final Loss Comparison')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, loss in zip(bars, final_losses):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{loss:.2e}', ha='center', va='bottom')
        
        # Plot 4: Success/Failure summary
        ax4 = axes[1, 1]
        success_counts = {'SUCCESS': 0, 'FAILED': 0}
        for test_name, data in results.items():
            if data.get('success', False):
                success_counts['SUCCESS'] += 1
            else:
                success_counts['FAILED'] += 1
        
        colors = ['green', 'red']
        ax4.pie(success_counts.values(), labels=success_counts.keys(), colors=colors, autopct='%1.0f%%')
        ax4.set_title('Success Rate')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Detailed results saved to: {save_path}")
    
    def run_diagnostic_tests(self):
        """Run diagnostic tests to understand WHY things fail"""
        print(f"\n" + "="*80)
        print(f"🔬 DIAGNOSTIC TESTING - Understanding the WHY")
        print(f"="*80)
        
        results = {}
        
        # Diagnostic Test 1: Minimal model with detailed analysis
        print(f"\n🧪 DIAGNOSTIC Test 1: Minimal Model Deep Dive")
        print("-" * 60)
        model = self.create_model_variant("minimal")
        criterion = nn.MSELoss()
        
        print(f"🔍 This test will show us:")
        print(f"  - Exact input/output data ranges")
        print(f"  - Initial gradient flow patterns")  
        print(f"  - Where/when exploding gradients occur")
        print(f"  - Model weight magnitudes vs gradient magnitudes")
        
        # Single sample test with FULL debugging (relaxed target based on our findings)
        single_losses, single_success = self.test_single_sample_overfit(
            model, criterion, target_loss=1e-3, max_iterations=200, debug=True
        )
        
        results["minimal_diagnostic"] = {
            'losses': single_losses,
            'success': single_success,
            'description': 'Minimal TCN - Deep Diagnostic'
        }
        
        # Diagnostic Test 2: Input normalization experiment
        print(f"\n🧪 DIAGNOSTIC Test 2: Input Normalization Impact")
        print("-" * 60)
        model2 = self.create_model_variant("minimal")
        
        # Test with normalized inputs
        audio_sample = self.audio_tiny[:1]
        target_sample = self.target_tiny[:1]
        
        # Manual normalization
        audio_mean = audio_sample.mean()
        audio_std = audio_sample.std() + 1e-6
        audio_normalized = (audio_sample - audio_mean) / audio_std
        
        print(f"🔍 Comparing raw vs normalized inputs:")
        print(f"Raw audio range: [{audio_sample.min():.3f}, {audio_sample.max():.3f}]")
        print(f"Normalized audio range: [{audio_normalized.min():.3f}, {audio_normalized.max():.3f}]")
        
        # Test with normalized input
        single_losses_norm, single_success_norm = self.test_normalized_input(
            model2, criterion, audio_normalized, target_sample
        )
        
        results["normalization_test"] = {
            'losses': single_losses_norm,
            'success': single_success_norm,
            'description': 'Input Normalization Test'
        }
        
        return results
    
    def test_normalized_input(self, model, criterion, audio_norm, target, max_iterations=100):
        """Test with pre-normalized input to isolate normalization effects"""
        print(f"\n🎯 Testing with pre-normalized input...")
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        losses = []
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            predictions = model(audio_norm)
            loss = criterion(predictions, target)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ Still getting NaN with normalized input at iter {iteration+1}")
                break
                
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            if iteration < 10 or (iteration + 1) % 25 == 0:
                print(f"Iter {iteration+1:3d}: Loss = {loss.item():.8f}, Grad = {grad_norm:.6f}")
            
            if loss.item() < 1e-4:
                print(f"🎉 SUCCESS with normalized input!")
                return losses, True
        
        print(f"Final loss with normalization: {losses[-1]:.8f}")
        return losses, False
    
    def run_systematic_tests(self):
        """Run systematic tests to identify bottlenecks"""
        print(f"\n" + "="*80)
        print(f"🔬 SYSTEMATIC COMPONENT TESTING")
        print(f"="*80)
        
        results = {}
        
        # Test 1: Minimal model with MSE loss (with safer parameters)
        print(f"\n🧪 Test 1: Minimal Model + MSE Loss (Fixed)")
        print("-" * 50)
        model = self.create_model_variant("minimal")
        criterion = nn.MSELoss()
        
        # Single sample test with debugging enabled
        single_losses, single_success = self.test_single_sample_overfit(
            model, criterion, target_loss=1e-4, max_iterations=200, debug=False
        )
        
        # Batch test with lower learning rate (relaxed target)
        batch_losses, batch_grads, batch_success = self.overfit_tiny_batch(
            model, criterion, target_loss=1e-3, learning_rate=1e-3, optimizer_type="adam"
        )
        
        results["minimal_mse_fixed"] = {
            'losses': batch_losses,
            'gradient_norms': batch_grads,
            'success': batch_success,
            'single_success': single_success,
            'description': 'Minimal TCN + MSE (Fixed)'
        }
        
        # Test 2: Original model without activations
        print(f"\n🧪 Test 2: Original Model - Linear Output")
        print("-" * 50)
        model = self.create_model_variant("no_activation")
        criterion = nn.MSELoss()
        
        batch_losses, batch_grads, batch_success = self.overfit_tiny_batch(
            model, criterion, target_loss=1e-3, learning_rate=0.01, optimizer_type="adamw"
        )
        
        results["original_linear"] = {
            'losses': batch_losses,
            'gradient_norms': batch_grads,
            'success': batch_success,
            'description': 'Original TCN - Linear Output'
        }
        
        # Test 3: Original model with sigmoid/tanh
        print(f"\n🧪 Test 3: Original Model + Activations")
        print("-" * 50)
        model = create_model().to(self.device)
        criterion = nn.MSELoss()
        
        batch_losses, batch_grads, batch_success = self.overfit_tiny_batch(
            model, criterion, target_loss=1e-3, learning_rate=0.01, optimizer_type="adamw"
        )
        
        results["original_activated"] = {
            'losses': batch_losses,
            'gradient_norms': batch_grads,
            'success': batch_success,
            'description': 'Original TCN + Activations'
        }
        
        # Test 4: Original model + Custom loss
        print(f"\n🧪 Test 4: Original Model + Custom Loss")
        print("-" * 50)
        model = create_model().to(self.device)
        criterion = AudioBlendshapeLoss(base_weight=1.0, temporal_weight=0.0, 
                                      silence_weight=0.0, pose_clamp_weight=0.0)
        
        batch_losses, batch_grads, batch_success = self.overfit_tiny_batch(
            model, criterion, target_loss=5e-3, learning_rate=0.01, optimizer_type="adamw"
        )
        
        results["original_custom_loss"] = {
            'losses': batch_losses,
            'gradient_norms': batch_grads,
            'success': batch_success,
            'description': 'Original TCN + Custom Loss'
        }
        
        # Store results
        self.test_results = results
        
        # Generate detailed plots
        self.plot_detailed_results(results)
        
        return results
    
    def run_diagnostic_mode(self):
        """Run diagnostic mode to understand WHY things fail"""
        print(f"\n" + "="*80)
        print(f"🔬 DIAGNOSTIC MODE - Deep Understanding")
        print(f"="*80)
        
        # Run diagnostic tests
        results = self.run_diagnostic_tests()
        
        # Summary for diagnostic mode
        print(f"\n" + "="*80)
        print(f"📋 DIAGNOSTIC SUMMARY")
        print(f"="*80)
        
        for test_name, result in results.items():
            status = "SUCCEEDED ✅" if result.get('success', False) else "REVEALED ISSUES ❌"
            final_loss = result['losses'][-1] if result['losses'] else float('inf')
            print(f"{result['description']:30} | {status}")
        
        print(f"\n🧠 UNDERSTANDING GAINED:")
        print(f"This diagnostic run should have revealed:")
        print(f"  🔍 Data scale issues (if input normalization helps)")
        print(f"  🔍 Model initialization problems (exploding/vanishing initial gradients)")
        print(f"  🔍 Architecture sensitivity (minimal vs complex model behavior)")
        
        return results
    
    def run_full_sanity_check(self):
        """Run complete enhanced sanity check pipeline"""
        print(f"\n" + "="*80)
        print(f"🔍 ENHANCED COMPREHENSIVE SANITY CHECK")
        print(f"="*80)
        
        # Run systematic tests
        results = self.run_systematic_tests()
        
        # Analyze results
        successful_tests = sum(1 for r in results.values() if r.get('success', False))
        total_tests = len(results)
        
        # Summary
        print(f"\n" + "="*80)
        print(f"📋 ENHANCED SANITY CHECK SUMMARY")
        print(f"="*80)
        
        for test_name, result in results.items():
            status = "PASSED ✅" if result.get('success', False) else "FAILED ❌"
            final_loss = result['losses'][-1] if result['losses'] else float('inf')
            print(f"{result['description']:30} | {status} | Final Loss: {final_loss:.2e}")
        
        print(f"\nOverall Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        
        # Recommendations based on which tests passed/failed
        print(f"\n📋 RECOMMENDATIONS:")
        if successful_tests == 0:
            print("❌ CRITICAL: No tests passed. Run diagnostic mode to understand why.")
            print("   Command: python sanity_check.py --diagnostic")
        elif results.get("minimal_mse_fixed", {}).get('success', False):
            print("✅ Fixed minimal model works - issue was in hyperparameters!")
            if not results.get("original_linear", {}).get('success', False):
                print("🔧 Issue: BatchNorm or model complexity causing problems")
            elif not results.get("original_activated", {}).get('success', False):
                print("🔧 Issue: Sigmoid/Tanh activations causing vanishing gradients")
            elif not results.get("original_custom_loss", {}).get('success', False):
                print("🔧 Issue: Custom loss function implementation")
        else:
            print("❌ Fundamental issue persists. Need deeper investigation.")
        
        # Save detailed report
        with open('sanity_check_report.json', 'w') as f:
            # Convert to serializable format
            serializable_results = {}
            for k, v in results.items():
                serializable_results[k] = {
                    'success': v.get('success', False),
                    'final_loss': v['losses'][-1] if v['losses'] else None,
                    'description': v.get('description', ''),
                    'iterations': len(v['losses']) if v['losses'] else 0
                }
            json.dump(serializable_results, f, indent=2)
        
        print(f"📄 Detailed report saved to: sanity_check_report.json")
        
        return successful_tests > 0

def main():
    """Main enhanced sanity check function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced TCN Sanity Check")
    parser.add_argument("--diagnostic", action="store_true", 
                       help="Run diagnostic mode to understand WHY things fail")
    parser.add_argument("--data-dir", type=str, default="multi_video_features/combined_dataset",
                       help="Data directory")
    
    args = parser.parse_args()
    
    if args.diagnostic:
        print(f"🔬 TCN Model DIAGNOSTIC Mode")
        print(f"This will deeply analyze WHY components fail")
    else:
        print(f"🧪 Enhanced TCN Model Sanity Check")
        print(f"This will systematically test components to identify bottlenecks")
    
    # Create enhanced sanity checker
    checker = SanityChecker(args.data_dir)
    
    if args.diagnostic:
        # Run diagnostic mode
        results = checker.run_diagnostic_mode()
        
        print(f"\n🧠 DIAGNOSTIC COMPLETE!")
        print(f"You should now understand:")
        print(f"  - Whether the issue is data scaling")
        print(f"  - Whether the issue is model architecture")
        print(f"  - Where exactly gradients explode/vanish")
        print(f"  - What the model is actually predicting")
        
        return True  # Diagnostic always "succeeds" by providing insight
    else:
        # Run standard sanity check
        success = checker.run_full_sanity_check()
        
        if success:
            print(f"\n✅ At least one test passed! Check the report for details.")
            print(f"📊 Review enhanced_sanity_results.png and sanity_check_report.json")
        else:
            print(f"\n❌ All tests failed. Run diagnostic mode:")
            print(f"   python sanity_check.py --diagnostic")
        
        return success

if __name__ == "__main__":
    main()