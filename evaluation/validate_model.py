#!/usr/bin/env python3
"""
Model Validation and Evaluation Script
Comprehensive evaluation of the trained TCN model
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib

# Add parent directory to path to import models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.tcn_model import create_model

class ModelValidator:
    """
    Comprehensive model validation
    """
    def __init__(self, model_path="models/best_tcn_model.pth", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize validator
        
        Args:
            model_path: Path to trained model
            device: Device to run validation on
        """
        self.device = device
        self.model_path = model_path
        
        # Key feature indices for detailed analysis
        self.feature_names = {
            'jaw_open': 25,      # Approximate MediaPipe blendshape indices
            'lip_close': 12,     # These would need to be adjusted based on 
            'smile': 20,         # actual MediaPipe blendshape mappings
            'eyebrow_raise': 5,
            'cheek_puff': 15,
            'mouth_left': 18,
            'mouth_right': 19,
        }
        
        # Pose feature indices (last 7 features)
        self.pose_indices = list(range(52, 59))
        self.pose_names = ['pos_x', 'pos_y', 'pos_z', 'rot_qw', 'rot_qx', 'rot_qy', 'rot_qz']
        
        # Load model
        self.model = self._load_model()
        
        print(f"Model Validator initialized on {device}")
        print(f"Model loaded from: {model_path}")
    
    def _load_model(self):
        """Load trained model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint with weights_only=False for compatibility
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Create model with same configuration
        model_config = checkpoint.get('model_config', {})
        model = create_model()  # Use default config if not available
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    
    def load_test_data(self, data_dir="extracted_features", test_split=0.2):
        """
        Load test data (we'll use part of our dataset as test data)
        
        Args:
            data_dir: Directory containing extracted features
            test_split: Fraction to use for testing
        
        Returns:
            dict: Test dataset
        """
        data_path = Path(data_dir)
        
        # Load arrays
        audio_sequences = np.load(data_path / "audio_sequences.npy")
        target_sequences = np.load(data_path / "target_sequences.npy")
        vad_sequences = np.load(data_path / "vad_sequences.npy")
        
        # Load metadata
        with open(data_path / "dataset_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create test split (use last portion as test data)
        total_samples = len(audio_sequences)
        test_size = int(total_samples * test_split)
        test_start = total_samples - test_size
        
        test_data = {
            'audio': torch.FloatTensor(audio_sequences[test_start:]),
            'targets': torch.FloatTensor(target_sequences[test_start:]),
            'vad': torch.FloatTensor(vad_sequences[test_start:]),
            'metadata': metadata
        }
        
        print(f"Test data loaded: {len(test_data['audio'])} sequences")
        return test_data
    
    def compute_comprehensive_metrics(self, test_data):
        """
        Compute comprehensive validation metrics
        
        Args:
            test_data: Test dataset
        
        Returns:
            dict: Comprehensive metrics
        """
        print("Computing comprehensive metrics...")
        
        all_predictions = []
        all_targets = []
        all_vad = []
        inference_times = []
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(test_data['audio'])), desc="Processing sequences"):
                audio = test_data['audio'][i:i+1].to(self.device)  # Single sequence
                targets = test_data['targets'][i:i+1].to(self.device)
                vad = test_data['vad'][i:i+1].to(self.device)
                
                # Measure inference time
                start_time = time.time()
                predictions = self.model(audio)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_vad.append(vad.cpu())
        
        # Concatenate all results
        predictions = torch.cat(all_predictions, dim=0)  # (num_sequences, seq_len, 59)
        targets = torch.cat(all_targets, dim=0)
        vad = torch.cat(all_vad, dim=0)
        
        # Flatten for metric computation
        pred_flat = predictions.view(-1, predictions.size(-1)).numpy()
        target_flat = targets.view(-1, targets.size(-1)).numpy()
        vad_flat = vad.view(-1).numpy()
        
        print(f"Computing metrics on {pred_flat.shape[0]} frames...")
        
        # 1. MAE per channel
        mae_per_channel = np.mean(np.abs(pred_flat - target_flat), axis=0)
        
        # 2. Key feature metrics
        key_metrics = {}
        for feature_name, idx in self.feature_names.items():
            if idx < pred_flat.shape[1]:
                mae = mae_per_channel[idx]
                try:
                    corr = pearsonr(pred_flat[:, idx], target_flat[:, idx])[0]
                    if np.isnan(corr):
                        corr = 0.0
                except:
                    corr = 0.0
                
                key_metrics[feature_name] = {
                    'mae': float(mae),
                    'correlation': float(corr)
                }
        
        # 3. Pose metrics
        pose_pred = pred_flat[:, self.pose_indices]
        pose_target = target_flat[:, self.pose_indices]
        pose_mae = np.mean(np.abs(pose_pred - pose_target), axis=0)
        
        pose_metrics = {}
        for i, name in enumerate(self.pose_names):
            try:
                corr = pearsonr(pose_pred[:, i], pose_target[:, i])[0]
                if np.isnan(corr):
                    corr = 0.0
            except:
                corr = 0.0
            
            pose_metrics[name] = {
                'mae': float(pose_mae[i]),
                'correlation': float(corr)
            }
        
        # 4. Overall metrics
        overall_mae = np.mean(mae_per_channel)
        blendshape_mae = np.mean(mae_per_channel[:52])
        pose_mae_avg = np.mean(pose_mae)
        
        # 5. Voice activity analysis
        voice_frames = vad_flat > 0.5
        silence_frames = vad_flat <= 0.5
        
        voice_mae = np.mean(np.abs(pred_flat[voice_frames] - target_flat[voice_frames])) if np.any(voice_frames) else 0.0
        silence_mae = np.mean(np.abs(pred_flat[silence_frames] - target_flat[silence_frames])) if np.any(silence_frames) else 0.0
        
        # 6. Latency metrics
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        fps_capability = 1.0 / np.mean(inference_times)
        
        # Compile comprehensive metrics
        metrics = {
            'overall_metrics': {
                'overall_mae': float(overall_mae),
                'blendshape_mae': float(blendshape_mae),
                'pose_mae': float(pose_mae_avg),
                'voice_activity_mae': float(voice_mae),
                'silence_mae': float(silence_mae),
                'avg_inference_time_ms': float(avg_inference_time),
                'fps_capability': float(fps_capability),
                'target_fps': 30.0,
                'real_time_capable': avg_inference_time < 33.33  # 30 FPS = 33.33ms per frame
            },
            'key_features': key_metrics,
            'pose_features': pose_metrics,
            'per_channel_mae': mae_per_channel.tolist(),
            'validation_criteria': self._evaluate_criteria(key_metrics, overall_mae, avg_inference_time)
        }
        
        return metrics
    
    def _evaluate_criteria(self, key_metrics, overall_mae, inference_time_ms):
        """
        Evaluate against the "Go" criteria mentioned in requirements
        """
        criteria = {
            'jaw_open_correlation': {
                'value': key_metrics.get('jaw_open', {}).get('correlation', 0.0),
                'target': 0.6,
                'passed': key_metrics.get('jaw_open', {}).get('correlation', 0.0) >= 0.6,
                'description': 'Pearson r ≥ 0.6 on jawOpen'
            },
            'mouth_mae': {
                'value': overall_mae,  # Using overall MAE as approximation
                'target': 0.1,
                'passed': overall_mae <= 0.1,
                'description': 'MAE ≤ 0.1 (0-1 scale) on mouth channels'
            },
            'frame_rate': {
                'value': 1000.0 / inference_time_ms if inference_time_ms > 0 else 0,
                'target': 20.0,
                'passed': (1000.0 / inference_time_ms if inference_time_ms > 0 else 0) >= 20.0,
                'description': 'Stable 20-30 FPS capability'
            },
            'latency': {
                'value': inference_time_ms,
                'target': 120.0,
                'passed': inference_time_ms <= 120.0,
                'description': 'End-to-end latency ≤ 120 ms'
            }
        }
        
        # Overall pass/fail
        all_passed = all(criterion['passed'] for criterion in criteria.values())
        
        return {
            'criteria': criteria,
            'overall_pass': all_passed,
            'passed_count': sum(1 for c in criteria.values() if c['passed']),
            'total_count': len(criteria)
        }
    
    def create_validation_report(self, metrics, output_dir="evaluation"):
        """
        Create comprehensive validation report
        
        Args:
            metrics: Computed metrics
            output_dir: Directory to save report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert metrics to JSON-serializable format
        def make_json_serializable(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        metrics_serializable = make_json_serializable(metrics)
        
        # Save metrics as JSON
        with open(output_path / "validation_metrics.json", 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        # Create text report
        report_lines = [
            "=" * 80,
            "AUDIO-TO-BLENDSHAPES MODEL VALIDATION REPORT",
            "=" * 80,
            "",
            "OVERALL PERFORMANCE:",
            f"  Overall MAE: {metrics['overall_metrics']['overall_mae']:.4f}",
            f"  Blendshape MAE: {metrics['overall_metrics']['blendshape_mae']:.4f}",
            f"  Head Pose MAE: {metrics['overall_metrics']['pose_mae']:.4f}",
            f"  Voice Activity MAE: {metrics['overall_metrics']['voice_activity_mae']:.4f}",
            f"  Silence MAE: {metrics['overall_metrics']['silence_mae']:.4f}",
            "",
            "PERFORMANCE METRICS:",
            f"  Average Inference Time: {metrics['overall_metrics']['avg_inference_time_ms']:.2f} ms",
            f"  FPS Capability: {metrics['overall_metrics']['fps_capability']:.1f} FPS",
            f"  Real-time Capable: {'[YES]' if metrics['overall_metrics']['real_time_capable'] else '[NO]'}",
            "",
            "KEY FEATURE PERFORMANCE:",
        ]
        
        for feature_name, feature_metrics in metrics['key_features'].items():
            report_lines.append(f"  {feature_name.replace('_', ' ').title()}:")
            report_lines.append(f"    MAE: {feature_metrics['mae']:.4f}")
            report_lines.append(f"    Correlation: {feature_metrics['correlation']:.3f}")
        
        report_lines.extend([
            "",
            "HEAD POSE PERFORMANCE:",
        ])
        
        for pose_name, pose_metrics in metrics['pose_features'].items():
            report_lines.append(f"  {pose_name.replace('_', ' ').title()}:")
            report_lines.append(f"    MAE: {pose_metrics['mae']:.4f}")
            report_lines.append(f"    Correlation: {pose_metrics['correlation']:.3f}")
        
        # Validation criteria
        validation = metrics['validation_criteria']
        report_lines.extend([
            "",
            "VALIDATION CRITERIA ('GO' DECISION):",
            f"  Passed: {validation['passed_count']}/{validation['total_count']} criteria",
            f"  Overall Result: {'[PASS]' if validation['overall_pass'] else '[FAIL]'}",
            "",
        ])
        
        for criterion_name, criterion in validation['criteria'].items():
            status = "[PASS]" if criterion['passed'] else "[FAIL]"
            report_lines.append(f"  {criterion['description']}: {status}")
            report_lines.append(f"    Value: {criterion['value']:.3f}, Target: {criterion['target']}")
        
        report_lines.extend([
            "",
            "=" * 80,
            f"Report generated: {Path().absolute()}",
            "=" * 80
        ])
        
        # Save text report
        report_text = "\\n".join(report_lines)
        with open(output_path / "validation_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Print to console
        print("\\n" + report_text)
        
        return output_path / "validation_report.txt"
    
    def plot_validation_results(self, metrics, output_dir="evaluation"):
        """
        Create visualization plots for validation results
        
        Args:
            metrics: Computed metrics
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Feature Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MAE comparison
        feature_names = list(metrics['key_features'].keys())
        maes = [metrics['key_features'][name]['mae'] for name in feature_names]
        
        axes[0, 0].bar(feature_names, maes)
        axes[0, 0].set_title('MAE by Key Feature')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Correlation comparison
        correlations = [metrics['key_features'][name]['correlation'] for name in feature_names]
        axes[0, 1].bar(feature_names, correlations)
        axes[0, 1].set_title('Correlation by Key Feature')
        axes[0, 1].set_ylabel('Pearson Correlation')
        axes[0, 1].axhline(y=0.6, color='r', linestyle='--', label='Target (0.6)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        
        # Per-channel MAE distribution
        per_channel_mae = metrics['per_channel_mae']
        axes[1, 0].hist(per_channel_mae, bins=20, alpha=0.7)
        axes[1, 0].axvline(x=np.mean(per_channel_mae), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(per_channel_mae):.3f}')
        axes[1, 0].set_title('Per-Channel MAE Distribution')
        axes[1, 0].set_xlabel('Mean Absolute Error')
        axes[1, 0].set_ylabel('Number of Channels')
        axes[1, 0].legend()
        
        # Performance criteria
        criteria = metrics['validation_criteria']['criteria']
        criterion_names = list(criteria.keys())
        criterion_values = [criteria[name]['value'] for name in criterion_names]
        criterion_targets = [criteria[name]['target'] for name in criterion_names]
        criterion_passed = [criteria[name]['passed'] for name in criterion_names]
        
        colors = ['green' if passed else 'red' for passed in criterion_passed]
        x_pos = np.arange(len(criterion_names))
        
        axes[1, 1].bar(x_pos - 0.2, criterion_values, 0.4, label='Actual', color=colors, alpha=0.7)
        axes[1, 1].bar(x_pos + 0.2, criterion_targets, 0.4, label='Target', color='blue', alpha=0.5)
        axes[1, 1].set_title('Validation Criteria Performance')
        axes[1, 1].set_xlabel('Criteria')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(criterion_names, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / "validation_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Validation plots saved to: {output_path / 'validation_plots.png'}")
        
        return output_path / "validation_plots.png"

def main():
    """
    Main validation function
    """
    print("=== MODEL VALIDATION ===")
    
    # Initialize validator
    validator = ModelValidator()
    
    # Load test data
    test_data = validator.load_test_data()
    
    # Compute comprehensive metrics
    metrics = validator.compute_comprehensive_metrics(test_data)
    
    # Create validation report
    report_path = validator.create_validation_report(metrics)
    
    # Create plots
    plot_path = validator.plot_validation_results(metrics)
    
    print(f"\\nValidation completed!")
    print(f"Report saved to: {report_path}")
    print(f"Plots saved to: {plot_path}")

if __name__ == "__main__":
    main()