# Audio-to-Blendshapes TCN Model

Real-time Temporal Convolutional Network (TCN) for converting audio to facial blendshapes and head pose. This project provides tools for training, inference, and deployment across multiple platforms including PyTorch, ONNX Runtime, and JavaScript.

## 🚀 Quick Start

### Training the Model

The model is already trained, but if you want to retrain or continue training:

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows PowerShell
# .venv\Scripts\activate.bat  # Windows Command Prompt
# source .venv/bin/activate  # Linux/Mac

# Quick mouth-focused training (latest approach - recommended for better mouth movements)
python quick_fix_training.py

# Train with incremental approach (standard approach)
python train_incremental.py --skip-sanity

# Alternative: Train with specific configuration
python train_incremental.py --data-dir multi_video_features/combined_dataset --batch-size 32 --skip-sanity
```

**Training Options:**

- `--skip-sanity`: Skip the sanity check (use if you've already verified your data)
- `--data-dir`: Specify the dataset directory (default: `multi_video_features/combined_dataset`)
- `--batch-size`: Set batch size (default: 32)

#### 🔥 Quick Mouth-Focused Training (Latest - Recommended)

The `quick_fix_training.py` script represents our latest training approach that significantly improves mouth movement quality:

**Features:**
- **Mouth-Focused Loss**: 5x weight on mouth region, 10x weight on jaw movements
- **Improved Normalization**: Better target normalization preserving natural blendshape ranges
- **Fine-tuning Approach**: Uses lower learning rate (5e-4) for refined training
- **GPU Optimized**: Requires CUDA for optimal performance

**What it fixes:**
- ✅ Poor mouth movement synchronization with audio
- ✅ Insufficient jaw opening during speech
- ✅ Over-compressed blendshape values
- ✅ Lack of dynamic range in facial movements

**Output Models:**
- `mouth_focused_final.pth`: Final model with enhanced mouth movements
- `mouth_focused_epoch_X.pth`: Intermediate checkpoints every 10 epochs

This script loads existing model weights (if available) and fine-tunes them with mouth-specific improvements, making it the preferred method for creating production-ready models.

#### Standard Incremental Training

The training script uses an incremental approach:

1. **Base Phase** (15 epochs): Learn basic audio-to-blendshapes mapping
2. **Temporal Phase** (10 epochs): Add temporal smoothness
3. **Silence Phase** (10 epochs): Improve mouth closure during silence
4. **Full Phase** (25 epochs): Complete training with all loss components

### Model Conversion and Inference

#### 1. Convert PyTorch Models to ONNX

```bash
cd deployment

# Convert best_full_model.pth to ONNX
python convert_to_onnx.py --input ../models/best_full_model.pth --output best_full_model.onnx

# Convert best_base_model.pth to ONNX
python convert_to_onnx.py --input ../models/best_base_model.pth --output best_base_model.onnx

# Convert with custom settings
python convert_to_onnx.py --input ../models/best_incremental_model.pth --batch-size 1 --sequence-length 24
```

#### 2. Run PyTorch Inference Demo

```bash
cd inference

# Run with trained model (uses synthetic audio if no audio file provided)
python pytorch_inference_demo.py --model ../models/best_full_model.pth

# Run with your own audio file
python pytorch_inference_demo.py --model ../models/best_full_model.pth --audio your_audio.wav

# Specify all components
python pytorch_inference_demo.py \
    --model ../models/best_full_model.pth \
    --audio your_audio.wav \
    --audio-scaler ../deployment/audio_scaler.pkl \
    --target-scaler ../deployment/target_scaler.pkl \
    --output pytorch_results
```

#### 3. Run ONNX Runtime Inference Demo

```bash
cd deployment

# First convert model to ONNX (if not done)
python convert_to_onnx.py --input ../models/best_full_model.pth

# Run ONNX inference demo
python onnx_inference_demo.py --model best_full_model.onnx

# With custom audio
python onnx_inference_demo.py --model best_full_model.onnx --audio your_audio.wav

# Full configuration
python onnx_inference_demo.py \
    --model best_full_model.onnx \
    --audio your_audio.wav \
    --audio-scaler audio_scaler.pkl \
    --target-scaler target_scaler.pkl \
    --output onnx_results
```

## 📁 Project Structure

```
train_LSTM/
├── models/                     # Trained model files
│   ├── best_full_model.pth     # Best full training model
│   ├── best_base_model.pth     # Best base model
│   ├── best_incremental_model.pth  # Best incremental training
│   └── tcn_model.py           # Model architecture
├── deployment/                 # Deployment tools and ONNX models
│   ├── convert_to_onnx.py     # PyTorch to ONNX converter
│   ├── onnx_inference_demo.py # ONNX Runtime inference demo
│   ├── audio_scaler.pkl       # Audio feature normalization
│   └── target_scaler.pkl      # Target normalization
├── inference/                  # Inference scripts
│   ├── pytorch_inference_demo.py  # PyTorch inference demo
│   └── real_time_inference.py     # Real-time audio processing
├── training/                   # Training scripts
│   ├── train_tcn.py           # Main training script
│   └── train_tcn_gpu_enhanced.py
├── data_preparation_scripts/   # Data preprocessing
│   └── ...
├── multi_video_features/       # Training datasets
│   └── combined_dataset/       # Main training data
├── quick_fix_training.py      # Latest mouth-focused training (recommended)
└── train_incremental.py       # Incremental training script
```

## 🎯 Model Architecture

- **Type**: Temporal Convolutional Network (TCN)
- **Input**: 80-dimensional mel spectrogram features
- **Context**: 24 frames (240ms) sliding window
- **Output**: 59 features (52 blendshapes + 7 head pose)
- **Real-time**: < 3ms inference time on modern hardware

### Input Format

- **Audio**: 16kHz sample rate
- **Features**: 80 mel spectrogram coefficients
- **Window**: 24 frames (240ms context)
- **Frame rate**: 100 Hz (10ms hop length)

### Output Format

- **Blendshapes**: 52 MediaPipe facial blendshapes [0-1]
- **Head pose**: 7 values [x, y, z, qw, qx, qy, qz]
  - Translation: x, y, z coordinates
  - Rotation: quaternion (w, x, y, z)

## 🛠️ Installation

### Requirements

```bash
# Install Python dependencies
pip install -r requirements.txt

# Additional requirements for ONNX
pip install onnx onnxruntime

# For visualization
pip install matplotlib seaborn
```

### Virtual Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows Command Prompt)
.venv\Scripts\activate.bat

# Activate (Linux/Mac)
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## 🎮 Usage Examples

### 1. Basic Model Training

```bash
# Use virtual environment
.venv\Scripts\Activate.ps1  # PowerShell

# Latest mouth-focused training (recommended)
python quick_fix_training.py

# Train with incremental approach (standard)
python train_incremental.py --skip-sanity
```

### 2. Convert Models for Deployment

```bash
cd deployment

# Convert your preferred model
python convert_to_onnx.py --input ../models/best_full_model.pth
python convert_to_onnx.py --input ../models/best_base_model.pth
```

### 3. Test PyTorch Inference

```bash
cd inference
# Use latest mouth-focused model (recommended)
python pytorch_inference_demo.py --model ../models/mouth_focused_final.pth

# Or use previous models
python pytorch_inference_demo.py --model ../models/best_full_model.pth
```

### 4. Test ONNX Inference (JavaScript Ready)

```bash
cd deployment
# Convert and test latest model
python convert_to_onnx.py --input ../models/mouth_focused_final.pth
python onnx_inference_demo.py --model mouth_focused_final.onnx

# Or use previous models
python onnx_inference_demo.py --model best_full_model.onnx
```

## 🌐 JavaScript Deployment

The ONNX models are ready for JavaScript deployment via ONNX.js:

```javascript
import { InferenceSession, Tensor } from "onnxjs";

// Load model
const session = new InferenceSession();
await session.loadModel("path/to/best_full_model.onnx");

// Prepare input (mel features: Float32Array of shape [1, 24, 80])
const inputTensor = new Tensor(melFeatures, "float32", [1, 24, 80]);

// Run inference
const results = await session.run({ audio_features: inputTensor });
const blendshapes = results["blendshapes"];
```

See the generated `javascript_example.js` in your output directory for complete examples.

## 📊 Model Performance

### Available Models

1. **mouth_focused_final.pth**: Latest mouth-focused model with enhanced facial movements (recommended)
2. **best_full_model.pth**: Complete training with all loss components
3. **best_base_model.pth**: Basic model focusing on core mapping
4. **best_incremental_model.pth**: Best model from incremental training

### Performance Metrics

- **Inference Speed**: < 3ms per frame
- **Real-time Capability**: 300+ FPS
- **Memory Usage**: ~4MB model size
- **Accuracy**: Suitable for real-time facial animation

## 🔧 Configuration

### Audio Parameters (Must Match Training)

```python
SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 160
WIN_LENGTH = 400
N_FFT = 512
CONTEXT_FRAMES = 24  # 240ms
```

### Training Parameters

```python
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 60 (incremental: 15+10+10+25)
```

## 🚨 Important Notes

1. **Virtual Environment**: Always use the `.venv` virtual environment when running Python scripts
2. **Quick Fix Training**: The `quick_fix_training.py` requires CUDA/GPU for optimal performance
3. **Sanity Check**: The `--skip-sanity` flag skips data validation (use only if data is verified)
4. **Model Paths**: Ensure correct relative paths when running scripts from different directories
5. **Audio Format**: Input audio must be 16kHz for proper inference
6. **ONNX Compatibility**: Models are exported with ONNX opset 11 for maximum compatibility

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the virtual environment

```bash
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

2. **Model Not Found**: Check file paths relative to your current directory

```bash
ls models/  # Check available models
ls deployment/  # Check scalers and ONNX files
```

3. **ONNX Runtime Issues**: Install the correct ONNX Runtime version

```bash
pip install onnxruntime  # CPU version
# OR
pip install onnxruntime-gpu  # GPU version
```

4. **Audio Processing Errors**: Ensure librosa is properly installed

```bash
pip install librosa soundfile
```

## 📚 Additional Resources

- **Model Architecture**: See `models/tcn_model.py` for detailed implementation
- **Training Details**: Check `train_incremental.py` for training pipeline
- **Real-time Processing**: Use `inference/real_time_inference.py` for live audio
- **Data Preparation**: Scripts in `data_preparation_scripts/` for custom datasets

## 🎉 Success Indicators

After running the scripts successfully, you should see:

✅ **Training**: Model files saved in `models/` directory  
✅ **Conversion**: ONNX files generated in `deployment/`  
✅ **PyTorch Inference**: Results and plots in `pytorch_inference_results/`  
✅ **ONNX Inference**: Results and JavaScript example in `onnx_inference_results/`

---

**Happy facial animation! 🎭**
