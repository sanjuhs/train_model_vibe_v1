# """
# Quick Fix Training Script
# Improves mouth movement by fixing normalization and adding mouth-focused loss
# """

# import torch
# import torch.nn as nn
# import numpy as np
# from pathlib import Path

# class MouthFocusedLoss(nn.Module):
#     """Loss function that prioritizes mouth movements."""
    
#     def __init__(self):
#         super().__init__()
#         # Mouth-related indices in MediaPipe blendshapes
#         self.mouth_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
#         self.jaw_open_idx = 25  # jawOpen is most important for speech
        
#     def forward(self, predictions, targets, vad=None):
#         # Apply sigmoid to predictions (convert logits to [0,1])
#         pred_sigmoid = torch.sigmoid(predictions)
        
#         # Standard MSELoss loss
#         base_loss = nn.MSELoss()(pred_sigmoid, targets)
        
#         # Extra emphasis on mouth region (5x weight)
#         mouth_pred = pred_sigmoid[:, :, self.mouth_indices]
#         mouth_target = targets[:, :, self.mouth_indices]
#         mouth_loss = nn.MSELoss()(mouth_pred, mouth_target) * 5.0
        
#         # Extra emphasis on jaw opening (correlates most with audio)
#         jaw_pred = pred_sigmoid[:, :, self.jaw_open_idx]
#         jaw_target = targets[:, :, self.jaw_open_idx]
#         jaw_loss = nn.MSELoss()(jaw_pred, jaw_target) * 10.0
        
#         total_loss = base_loss + mouth_loss + jaw_loss
        
#         return {
#             'total_loss': total_loss,
#             'base_loss': base_loss,
#             'mouth_loss': mouth_loss,
#             'jaw_loss': jaw_loss,
#             'temporal_loss': torch.tensor(0.0),
#             'silence_loss': torch.tensor(0.0),
#             'pose_loss': torch.tensor(0.0)
#         }

# def fix_target_normalization(target_sequences):
#     """
#     Fix the target normalization to preserve natural blendshape ranges.
#     Instead of min-max normalization, use expected physiological ranges.
#     """
#     print("🔧 Applying improved target normalization...")
    
#     # Copy data
#     fixed_targets = target_sequences.copy()
    
#     # Split blendshapes and pose
#     blendshapes = fixed_targets[:, :, :52]
#     pose = fixed_targets[:, :, 52:]
    
#     # Check current ranges
#     print(f"Original blendshapes range: [{blendshapes.min():.3f}, {blendshapes.max():.3f}]")
#     print(f"Original pose range: [{pose.min():.3f}, {pose.max():.3f}]")
    
#     # Apply gentle normalization that preserves natural ranges
#     # MediaPipe blendshapes naturally range from 0 to ~0.8 for most movements
    
#     # Method 1: Clip outliers and scale gently
#     blendshapes_clipped = np.clip(blendshapes, 0.0, 1.0)
    
#     # If values are too small (compressed), scale them up
#     if blendshapes_clipped.max() < 0.1:
#         print("⚠️ Blendshapes seem compressed, scaling up...")
#         blendshapes_clipped *= 5.0
#         blendshapes_clipped = np.clip(blendshapes_clipped, 0.0, 0.8)
    
#     # Ensure mouth region has good dynamic range
#     mouth_indices = list(range(23, 52))
#     mouth_values = blendshapes_clipped[:, :, mouth_indices]
#     mouth_range = mouth_values.max() - mouth_values.min()
    
#     if mouth_range < 0.1:
#         print("⚠️ Mouth range too small, boosting mouth movements...")
#         # Boost mouth movements specifically
#         mouth_boost = 0.3 / (mouth_range + 1e-6)
#         blendshapes_clipped[:, :, mouth_indices] *= mouth_boost
#         blendshapes_clipped[:, :, mouth_indices] = np.clip(
#             blendshapes_clipped[:, :, mouth_indices], 0.0, 0.6
#         )
    
#     # Fix pose normalization (keep small values)
#     pose_normalized = pose / (np.abs(pose).max() + 1e-6) * 0.1
    
#     # Recombine
#     fixed_targets[:, :, :52] = blendshapes_clipped
#     fixed_targets[:, :, 52:] = pose_normalized
    
#     print(f"Fixed blendshapes range: [{blendshapes_clipped.min():.3f}, {blendshapes_clipped.max():.3f}]")
#     print(f"Fixed pose range: [{pose_normalized.min():.3f}, {pose_normalized.max():.3f}]")
    
#     # Analyze mouth movement specifically
#     mouth_values_fixed = blendshapes_clipped[:, :, mouth_indices]
#     jaw_values_fixed = blendshapes_clipped[:, :, 25]  # jawOpen
    
#     print(f"Mouth movement range: [{mouth_values_fixed.min():.3f}, {mouth_values_fixed.max():.3f}]")
#     print(f"Jaw open range: [{jaw_values_fixed.min():.3f}, {jaw_values_fixed.max():.3f}]")
#     print(f"Jaw variation (std): {jaw_values_fixed.std():.3f}")
    
#     if jaw_values_fixed.std() < 0.05:
#         print("⚠️ Still low jaw variation! Your training data might lack diverse mouth movements.")
    
#     return fixed_targets

# def quick_retrain_mouth_focused(data_dir="multi_video_features/combined_dataset"):
#     """
#     Quick retraining script focused on mouth movements.
#     """
#     print("🚀 Quick Mouth-Focused Retraining")
    
#     # Load your existing model
#     from models.tcn_model import create_model
#     model = create_model()
    
#     # Load existing weights if available
#     try:
#         checkpoint = torch.load("models/best_base_model.pth", map_location='cpu')
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print("✅ Loaded existing model weights")
#     except:
#         print("⚠️ No existing weights found, starting fresh")
    
#     # Use mouth-focused loss
#     criterion = MouthFocusedLoss()
    
#     # Lower learning rate for fine-tuning
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
#     # Load and fix your data
#     print("📊 Loading and fixing dataset...")
#     data_dir = Path(data_dir)
    
#     audio_sequences = np.load(data_dir / "audio_sequences.npy")
#     target_sequences = np.load(data_dir / "target_sequences.npy")
#     vad_sequences = np.load(data_dir / "vad_sequences.npy")
    
#     # Apply improved normalization
#     target_sequences_fixed = fix_target_normalization(target_sequences)
    
#     # Convert to tensors
#     audio_tensor = torch.FloatTensor(audio_sequences)
#     target_tensor = torch.FloatTensor(target_sequences_fixed)
#     vad_tensor = torch.FloatTensor(vad_sequences)
    
#     # Create dataset
#     dataset = torch.utils.data.TensorDataset(audio_tensor, target_tensor, vad_tensor)
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
#     # Training loop
#     print("🎯 Training with mouth focus...")
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
    
#     for epoch in range(50):  # Quick training
#         model.train()
#         total_loss = 0.0
        
#         for batch_idx, (audio, targets, vad) in enumerate(train_loader):
#             audio, targets, vad = audio.to(device), targets.to(device), vad.to(device)
            
#             optimizer.zero_grad()
#             predictions = model(audio)
            
#             loss_dict = criterion(predictions, targets, vad)
#             loss = loss_dict['total_loss']
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             total_loss += loss.item()
            
#             if batch_idx % 10 == 0:
#                 print(f"Epoch {epoch}, Batch {batch_idx}: "
#                       f"Total={loss.item():.4f}, "
#                       f"Mouth={loss_dict['mouth_loss'].item():.4f}, "
#                       f"Jaw={loss_dict['jaw_loss'].item():.4f}")
        
#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
#         # Save every 10 epochs
#         if epoch % 10 == 0:
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': avg_loss
#             }, f"models/mouth_focused_epoch_{epoch}.pth")
    
#     # Save final model
#     torch.save({
#         'epoch': 50,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': avg_loss
#     }, "models/mouth_focused_final.pth")
    
#     print("✅ Mouth-focused training completed!")
#     print("Test the new model with your inference script.")

# if __name__ == "__main__":
#     quick_retrain_mouth_focused()

"""
Quick Fix Training Script
Improves mouth movement by fixing normalization and adding mouth-focused loss
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class MouthFocusedLoss(nn.Module):
    """Loss function that prioritizes mouth movements."""
    
    def __init__(self):
        super().__init__()
        # Mouth-related indices in MediaPipe blendshapes
        self.mouth_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
        self.jaw_open_idx = 25  # jawOpen is most important for speech
        
    def forward(self, predictions, targets, vad=None):
        # Apply sigmoid to predictions (convert logits to [0,1])
        pred_sigmoid = torch.sigmoid(predictions)
        
        # Standard MSE loss
        base_loss = nn.MSELoss()(pred_sigmoid, targets)
        
        # Extra emphasis on mouth region (5x weight)
        mouth_pred = pred_sigmoid[:, :, self.mouth_indices]
        mouth_target = targets[:, :, self.mouth_indices]
        mouth_loss = nn.MSELoss()(mouth_pred, mouth_target) * 5.0
        
        # Extra emphasis on jaw opening (correlates most with audio)
        jaw_pred = pred_sigmoid[:, :, self.jaw_open_idx]
        jaw_target = targets[:, :, self.jaw_open_idx]
        jaw_loss = nn.MSELoss()(jaw_pred, jaw_target) * 10.0
        
        total_loss = base_loss + mouth_loss + jaw_loss
        
        return {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'mouth_loss': mouth_loss,
            'jaw_loss': jaw_loss,
            'temporal_loss': torch.tensor(0.0),
            'silence_loss': torch.tensor(0.0),
            'pose_loss': torch.tensor(0.0)
        }

def fix_target_normalization(target_sequences):
    """
    Fix the target normalization to preserve natural blendshape ranges.
    Instead of min-max normalization, use expected physiological ranges.
    """
    print("🔧 Applying improved target normalization...")
    
    # Copy data
    fixed_targets = target_sequences.copy()
    
    # Split blendshapes and pose
    blendshapes = fixed_targets[:, :, :52]
    pose = fixed_targets[:, :, 52:]
    
    # Check current ranges
    print(f"Original blendshapes range: [{blendshapes.min():.3f}, {blendshapes.max():.3f}]")
    print(f"Original pose range: [{pose.min():.3f}, {pose.max():.3f}]")
    
    # Apply gentle normalization that preserves natural ranges
    # MediaPipe blendshapes naturally range from 0 to ~0.8 for most movements
    
    # Method 1: Clip outliers and scale gently
    blendshapes_clipped = np.clip(blendshapes, 0.0, 1.0)
    
    # If values are too small (compressed), scale them up
    if blendshapes_clipped.max() < 0.1:
        print("⚠️ Blendshapes seem compressed, scaling up...")
        blendshapes_clipped *= 5.0
        blendshapes_clipped = np.clip(blendshapes_clipped, 0.0, 0.8)
    
    # Ensure mouth region has good dynamic range
    mouth_indices = list(range(23, 52))
    mouth_values = blendshapes_clipped[:, :, mouth_indices]
    mouth_range = mouth_values.max() - mouth_values.min()
    
    if mouth_range < 0.1:
        print("⚠️ Mouth range too small, boosting mouth movements...")
        # Boost mouth movements specifically
        mouth_boost = 0.3 / (mouth_range + 1e-6)
        blendshapes_clipped[:, :, mouth_indices] *= mouth_boost
        blendshapes_clipped[:, :, mouth_indices] = np.clip(
            blendshapes_clipped[:, :, mouth_indices], 0.0, 0.6
        )
    
    # Fix pose normalization (keep small values)
    pose_normalized = pose / (np.abs(pose).max() + 1e-6) * 0.1
    
    # Recombine
    fixed_targets[:, :, :52] = blendshapes_clipped
    fixed_targets[:, :, 52:] = pose_normalized
    
    print(f"Fixed blendshapes range: [{blendshapes_clipped.min():.3f}, {blendshapes_clipped.max():.3f}]")
    print(f"Fixed pose range: [{pose_normalized.min():.3f}, {pose_normalized.max():.3f}]")
    
    # Analyze mouth movement specifically
    mouth_values_fixed = blendshapes_clipped[:, :, mouth_indices]
    jaw_values_fixed = blendshapes_clipped[:, :, 25]  # jawOpen
    
    print(f"Mouth movement range: [{mouth_values_fixed.min():.3f}, {mouth_values_fixed.max():.3f}]")
    print(f"Jaw open range: [{jaw_values_fixed.min():.3f}, {jaw_values_fixed.max():.3f}]")
    print(f"Jaw variation (std): {jaw_values_fixed.std():.3f}")
    
    if jaw_values_fixed.std() < 0.05:
        print("⚠️ Still low jaw variation! Your training data might lack diverse mouth movements.")
    
    return fixed_targets

def quick_retrain_mouth_focused(data_dir="multi_video_features_fixed/combined_dataset"):
    """
    Quick retraining script focused on mouth movements.
    """
    print("🚀 Quick Mouth-Focused Retraining")
    
    # Force GPU usage
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This script requires GPU.")
        print("Please ensure CUDA is properly installed.")
        return False
    
    device = torch.device('cuda')
    print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load your existing model
    from models.tcn_model import create_model
    model = create_model()
    
    # Load existing weights if available
    try:
        checkpoint = torch.load("models/best_base_model.pth", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded existing model weights")
    except:
        print("⚠️ No existing weights found, starting fresh")
    
    # Use mouth-focused loss
    criterion = MouthFocusedLoss()
    
    # Lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # Load and fix your data
    print("📊 Loading and fixing dataset...")
    data_dir = Path(data_dir)
    
    audio_sequences = np.load(data_dir / "audio_sequences.npy")
    target_sequences = np.load(data_dir / "target_sequences.npy")
    vad_sequences = np.load(data_dir / "vad_sequences.npy")
    
    # Apply improved normalization
    target_sequences_fixed = fix_target_normalization(target_sequences)
    
    # Convert to tensors
    audio_tensor = torch.FloatTensor(audio_sequences)
    target_tensor = torch.FloatTensor(target_sequences_fixed)
    vad_tensor = torch.FloatTensor(vad_sequences)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(audio_tensor, target_tensor, vad_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Training loop
    print("🎯 Training with mouth focus...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(50):  # Quick training
        model.train()
        total_loss = 0.0
        
        for batch_idx, (audio, targets, vad) in enumerate(train_loader):
            audio, targets, vad = audio.to(device), targets.to(device), vad.to(device)
            
            optimizer.zero_grad()
            predictions = model(audio)
            
            loss_dict = criterion(predictions, targets, vad)
            loss = loss_dict['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Total={loss.item():.4f}, "
                      f"Mouth={loss_dict['mouth_loss'].item():.4f}, "
                      f"Jaw={loss_dict['jaw_loss'].item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Save every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, f"models/mouth_focused_epoch_{epoch}.pth")
    
    # Save final model
    torch.save({
        'epoch': 50,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, "models/mouth_focused_final.pth")
    
    print("✅ Mouth-focused training completed!")
    print("Test the new model with your inference script.")

if __name__ == "__main__":
    quick_retrain_mouth_focused()