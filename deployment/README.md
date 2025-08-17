# Audio-to-Blendshapes TCN Model

Real-time temporal convolutional network for audio-driven facial animation.

## Model Specifications
- **Architecture**: TCN with 8 layers, 345,467 parameters
- **Model Size**: 4.1 MB
- **Input**: 24 frames × 80 mel features (240ms context)
- **Output**: 52 blendshapes + 7 head pose values
- **Performance**: 365 FPS capability

## Quick Start

```python
import torch
import numpy as np

# Load model
model = torch.load('best_tcn_model.pth', map_location='cpu')
model.eval()

# Prepare input (example)
audio_features = np.random.randn(1, 24, 80)  # (batch, time, features)
input_tensor = torch.FloatTensor(audio_features)

# Run inference
with torch.no_grad():
    output = model(input_tensor)  # (batch, time, 59)

# Extract latest predictions
latest_frame = output[0, -1]  # Last time step
blendshapes = latest_frame[:52]  # First 52 values
head_pose = latest_frame[52:]   # Last 7 values
```

## Files Included
- best_tcn_model.pth
- audio_scaler.pkl
- target_scaler.pkl
- dataset_metadata.json

## Performance Metrics
- **Inference Time**: 2.74ms
- **Real-time Capable**: True
- **Overall MAE**: 0.7104867696762085

## Requirements
- Python 3.8+
- PyTorch 2.0+
- librosa (for audio processing)
- scikit-learn (for normalization)

## Usage Notes
- Model expects normalized mel features (use provided audio_scaler.pkl)
- Apply EMA smoothing (alpha=0.85) for stable output
- Target output range: 0-1 for blendshapes, varies for pose
- Best performance with 16kHz audio input

See `deployment_info.json` for complete technical specifications.
