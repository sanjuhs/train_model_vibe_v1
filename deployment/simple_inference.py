#!/usr/bin/env python3
"""
Simple inference example for the TCN model
"""

import torch
import numpy as np
import joblib
from pathlib import Path

def load_model_and_scalers():
    """Load model and normalization scalers"""
    # Load model
    model_path = "best_tcn_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Create model (you'll need to import this from your model file)
    # For now, we'll load the state dict directly
    model = checkpoint['model_state_dict']  # This is just the weights
    
    # Load scalers
    audio_scaler = joblib.load("audio_scaler.pkl")
    target_scaler = joblib.load("target_scaler.pkl")
    
    return model, audio_scaler, target_scaler

def predict_blendshapes(mel_features, model, audio_scaler, target_scaler):
    """
    Predict blendshapes from mel features
    
    Args:
        mel_features: (24, 80) mel spectrogram features
        model: Loaded model
        audio_scaler: Audio feature scaler
        target_scaler: Target feature scaler
    
    Returns:
        blendshapes: (52,) blendshape values
        head_pose: (7,) head pose values
    """
    
    # Normalize input
    mel_flat = mel_features.reshape(-1, mel_features.shape[-1])
    mel_normalized = audio_scaler.transform(mel_flat)
    mel_normalized = mel_normalized.reshape(mel_features.shape)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.FloatTensor(mel_normalized).unsqueeze(0)
    
    # Run inference (note: this is simplified, you need the full model)
    # output = model(input_tensor)
    # For demo, return random values
    output = torch.randn(1, 24, 59)
    
    # Get latest prediction and denormalize
    latest_pred = output[0, -1].numpy()
    denormalized = target_scaler.inverse_transform(latest_pred.reshape(1, -1))[0]
    
    # Split into blendshapes and pose
    blendshapes = denormalized[:52]
    head_pose = denormalized[52:]
    
    return blendshapes, head_pose

def main():
    """Demo inference"""
    print("Loading model and scalers...")
    
    try:
        model, audio_scaler, target_scaler = load_model_and_scalers()
        print("[OK] Model loaded successfully")
        
        # Create dummy input
        mel_features = np.random.randn(24, 80)
        print(f"Input shape: {mel_features.shape}")
        
        # Predict
        blendshapes, head_pose = predict_blendshapes(
            mel_features, model, audio_scaler, target_scaler
        )
        
        print(f"Output blendshapes: {blendshapes.shape}")
        print(f"Output head pose: {head_pose.shape}")
        print("[OK] Inference completed")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        print("Make sure all model files are in the same directory")

if __name__ == "__main__":
    main()
