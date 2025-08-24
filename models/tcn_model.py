#!/usr/bin/env python3
"""
TCN (Temporal Convolutional Network) for Audio-to-Blendshapes
Real-time causal model for audio to facial animation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution for efficiency
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=in_channels,  # Key: groups=in_channels makes it depthwise
            bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x = self.bn1(self.depthwise(x))
        x = self.bn2(self.pointwise(x))
        return x

class TCNBlock(nn.Module):
    """
    Single TCN block with dilated causal convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding for causal convolution (no future information)
        self.padding = (kernel_size - 1) * dilation
        
        # Depthwise separable convolution
        self.conv = DepthwiseSeparableConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        # Activation and normalization
        self.activation = nn.GELU()  # GELU as recommended
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        """
        Forward pass with causal convolution
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Output tensor of shape (batch, channels, time)
        """
        residual = x
        
        # Apply convolution with causal padding
        out = self.conv(x)
        
        # Remove future information (causal)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        # Ensure same sequence length for residual connection
        if residual.size(2) != out.size(2):
            min_len = min(residual.size(2), out.size(2))
            residual = residual[:, :, :min_len]
            out = out[:, :, :min_len]
        
        return out + residual

class AudioToBlendshapesTCN(nn.Module):
    """
    TCN model for real-time audio to blendshapes + head pose
    """
    def __init__(self, 
                 input_dim=80,           # Mel features
                 output_dim=59,          # 52 blendshapes + 7 head pose
                 hidden_channels=192,    # Hidden dimension
                 num_layers=8,           # Number of TCN layers
                 kernel_size=3,          # Convolution kernel size
                 dropout=0.1,            # Dropout rate
                 max_dilation=64):       # Maximum dilation
        """
        Initialize TCN model
        
        Args:
            input_dim: Input feature dimension (80 mel features)
            output_dim: Output dimension (59: 52 blendshapes + 7 pose)
            hidden_channels: Hidden channels in TCN blocks
            num_layers: Number of TCN layers
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            max_dilation: Maximum dilation factor
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Calculate dilations (exponentially increasing)
        self.dilations = [min(2**i, max_dilation) for i in range(num_layers)]
        
        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field(kernel_size, self.dilations)
        
        print(f"TCN Model Architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Hidden channels: {hidden_channels}")
        print(f"  Layers: {num_layers}")
        print(f"  Dilations: {self.dilations}")
        print(f"  Receptive field: {self.receptive_field} frames (~{self.receptive_field*10:.0f}ms)")
        
        # Input projection
        self.input_conv = nn.Conv1d(input_dim, hidden_channels, 1)
        
        # TCN layers
        self.tcn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = hidden_channels
            out_ch = hidden_channels
            dilation = self.dilations[i]
            
            self.tcn_layers.append(
                TCNBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Output layers with proper bounding for blendshapes [0,1] + pose
        self.output_layers = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels // 2, output_dim, 1)
        )
        
        # Separate output activation: sigmoid for blendshapes [0,1], tanh for pose [-1,1]
        self.blendshape_activation = nn.Sigmoid()  # For indices 0-51 (blendshapes)
        self.pose_activation = nn.Tanh()           # For indices 52-58 (pose, scaled to reasonable range)
        
        # Initialize weights
        self._initialize_weights()
        
        # Scale final layer weights for better initial range
        with torch.no_grad():
            for module in self.output_layers:
                if isinstance(module, nn.Conv1d) and hasattr(module, 'weight'):
                    module.weight.data *= 0.1  # Scale down initial weights
                    if module.bias is not None:
                        module.bias.data *= 0.1
        
        # Calculate model size
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size_mb = self.num_parameters * 4 / (1024 * 1024)  # 4 bytes per float32
        print(f"  Parameters: {self.num_parameters:,} ({size_mb:.1f} MB)")
    
    def _calculate_receptive_field(self, kernel_size, dilations):
        """Calculate the receptive field of the network"""
        receptive_field = 1
        for dilation in dilations:
            receptive_field += (kernel_size - 1) * dilation
        return receptive_field
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, time, features) or (batch, features, time)
        
        Returns:
            Output tensor of shape (batch, time, output_dim)
        """
        # Ensure input is (batch, features, time)
        if x.dim() == 3 and x.size(-1) == self.input_dim:
            x = x.transpose(1, 2)  # (batch, time, features) -> (batch, features, time)
        
        # Add shape assertion for debugging
        assert x.shape[1] == self.input_dim, f"Expected input dim {self.input_dim}, got {x.shape[1]}"
        
        # Input projection
        x = self.input_conv(x)  # (batch, hidden_channels, time)
        
        # Pass through TCN layers
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)
        
        # Output projection
        x = self.output_layers(x)  # (batch, output_dim, time)
        
        # Apply appropriate activations for different output components
        # Blendshapes (0-51): sigmoid to [0,1]
        blendshapes = self.blendshape_activation(x[:, :52, :])
        # Pose (52-58): tanh to [-1,1] for reasonable pose range
        pose = self.pose_activation(x[:, 52:, :]) * 0.2  # Scale to [-0.2, 0.2] for pose
        
        # Combine outputs
        x = torch.cat([blendshapes, pose], dim=1)
        
        # Return as (batch, time, output_dim) for convenience
        return x.transpose(1, 2)
    
    def get_model_info(self):
        """Get model information"""
        return {
            'architecture': 'TCN',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'dilations': self.dilations,
            'receptive_field_frames': self.receptive_field,
            'receptive_field_ms': self.receptive_field * 10,  # Assuming 100Hz
            'num_parameters': self.num_parameters,
            'model_size_mb': self.num_parameters * 4 / (1024 * 1024)
        }

def create_model(config=None):
    """
    Create TCN model with default or custom configuration
    
    Args:
        config: Optional model configuration dict
    
    Returns:
        TCN model instance
    """
    if config is None:
        # Default configuration for real-time performance
        config = {
            'input_dim': 80,        # 80 mel features
            'output_dim': 59,       # 52 blendshapes + 7 head pose
            'hidden_channels': 192, # Balanced size for performance
            'num_layers': 6,        # Reduced from 8 for better receptive field control
            'kernel_size': 3,       # Standard kernel size
            'dropout': 0.1,         # Light dropout
            'max_dilation': 16      # Reduced from 64 for ~300ms receptive field
        }
    
    model = AudioToBlendshapesTCN(**config)
    return model

def test_model():
    """Test the model with dummy data"""
    print("Testing TCN model...")
    
    # Create model
    model = create_model()
    model.eval()
    
    # Test with different input shapes
    batch_size = 8
    seq_length = 24  # 240ms at 100Hz
    input_dim = 80
    
    # Test input (batch, time, features)
    x = torch.randn(batch_size, seq_length, input_dim)
    print(f"\\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_length}, 59)")
    
    # Test causal property (output should not depend on future)
    x_truncated = x[:, :-5, :]  # Remove last 5 frames
    with torch.no_grad():
        output_truncated = model(x_truncated)
    
    # First part should be identical
    if torch.allclose(output[:, :-5, :], output_truncated, atol=1e-6):
        print("✅ Causal property verified: model doesn't use future information")
    else:
        print("❌ Causal property failed")
    
    # Print model info
    info = model.get_model_info()
    print(f"\\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_model()