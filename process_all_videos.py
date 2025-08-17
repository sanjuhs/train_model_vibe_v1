#!/usr/bin/env python3
"""
Simple script to process all videos for training
"""

import os
import sys
import json
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

def process_single_video(video_path, output_dir, video_index):
    """Process a single video using existing scripts"""
    
    video_name = Path(video_path).stem
    video_output_dir = Path(output_dir) / f"video_{video_index:02d}_{video_name}"
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\\nProcessing video {video_index + 1}: {video_name}")
    print(f"Output directory: {video_output_dir}")
    
    try:
        # Step 1: Extract blendshapes (modify existing script to accept parameters)
        print("  Step 1: Extracting blendshapes...")
        
        # Create a temporary script for blendshape extraction
        blendshape_script = f'''
import sys
sys.path.append("data_preparation_scripts")
exec(open("data_preparation_scripts/02_extract_blendshapes.py", encoding="utf-8").read())

# Override the main function
def custom_main():
    video_path = r"{video_path}"
    output_dir = r"{video_output_dir}"
    
    print("Initializing Face Blendshape Extractor...")
    extractor = FaceBlendshapeExtractor()
    
    print("Starting feature extraction...")
    extraction_data = extractor.extract_from_video(
        video_path, 
        output_dir=output_dir,
        max_frames=None  # Process full video
    )
    
    print("Feature extraction completed!")
    return extraction_data

if __name__ == "__main__":
    custom_main()
'''
        
        # Write and execute blendshape extraction
        temp_script = video_output_dir / "temp_blendshape.py"
        with open(temp_script, 'w') as f:
            f.write(blendshape_script)
        
        import subprocess
        result = subprocess.run([
            sys.executable, str(temp_script)
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            print(f"    Error in blendshape extraction: {result.stderr}")
            return None
        
        # Clean up temp script
        temp_script.unlink()
        
        # Step 2: Extract audio features
        print("  Step 2: Extracting audio...")
        
        audio_script = f'''
import sys
sys.path.append("data_preparation_scripts")
exec(open("data_preparation_scripts/03_extract_audio_features.py", encoding="utf-8").read())

def custom_main():
    video_path = r"{video_path}"
    output_dir = r"{video_output_dir}"
    
    print("Initializing Audio Feature Extractor...")
    extractor = AudioFeatureExtractor()
    
    print("Starting audio feature extraction...")
    features = extractor.extract_from_video(
        video_path,
        output_dir=output_dir,
        max_duration=None  # Process full audio
    )
    
    print("Audio feature extraction completed!")
    return features

if __name__ == "__main__":
    custom_main()
'''
        
        temp_script = video_output_dir / "temp_audio.py"
        with open(temp_script, 'w') as f:
            f.write(audio_script)
        
        result = subprocess.run([
            sys.executable, str(temp_script)
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            print(f"    Error in audio extraction: {result.stderr}")
            return None
        
        temp_script.unlink()
        
        # Step 3: Create dataset
        print("  Step 3: Creating dataset...")
        
        dataset_script = f'''
import sys
sys.path.append("data_preparation_scripts")
exec(open("data_preparation_scripts/04_create_dataset.py", encoding="utf-8").read())

def custom_main():
    features_dir = r"{video_output_dir}"
    
    print("Creating synchronized training dataset...")
    creator = DatasetCreator(sequence_length_ms=240, overlap_ms=120)
    
    print("Loading extracted features...")
    audio_data, visual_data = creator.load_features(features_dir)
    
    print("Synchronizing audio and visual features...")
    synchronized_data = creator.synchronize_features(audio_data, visual_data)
    
    print("Creating training sequences...")
    sequences_data = creator.create_sequences(synchronized_data)
    
    print("Normalizing features...")
    final_dataset = creator.normalize_features(sequences_data, features_dir)
    
    print("Saving dataset...")
    output_path = creator.save_dataset(final_dataset, features_dir)
    
    print(f"Dataset creation completed! Sequences: {{sequences_data['metadata']['num_sequences']}}")
    return sequences_data

if __name__ == "__main__":
    custom_main()
'''
        
        temp_script = video_output_dir / "temp_dataset.py"
        with open(temp_script, 'w') as f:
            f.write(dataset_script)
        
        result = subprocess.run([
            sys.executable, str(temp_script)
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            print(f"    Error in dataset creation: {result.stderr}")
            return None
        
        temp_script.unlink()
        
        # Load the final results for summary
        try:
            with open(video_output_dir / "dataset_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Create processing summary
            processing_summary = {
                'video_path': str(video_path),
                'video_name': video_name,
                'video_index': video_index,
                'num_sequences': metadata['dataset_info']['num_sequences'],
                'processing_status': 'success'
            }
            
            with open(video_output_dir / "processing_summary.json", 'w') as f:
                json.dump(processing_summary, f, indent=2)
            
            print(f"  ✅ Video {video_index + 1} processed successfully!")
            print(f"     Sequences created: {metadata['dataset_info']['num_sequences']}")
            
            return processing_summary
            
        except Exception as e:
            print(f"    Warning: Could not load final metadata: {e}")
            return {'video_path': str(video_path), 'video_name': video_name, 
                   'video_index': video_index, 'processing_status': 'partial'}
    
    except Exception as e:
        print(f"  ❌ Error processing video {video_index + 1}: {e}")
        return {'video_path': str(video_path), 'video_name': video_name, 
               'video_index': video_index, 'processing_status': 'failed', 'error': str(e)}

def combine_all_datasets(output_dir):
    """Combine datasets from all processed videos"""
    
    output_path = Path(output_dir)
    video_dirs = [d for d in output_path.iterdir() 
                 if d.is_dir() and d.name.startswith("video_")]
    
    if not video_dirs:
        print("No processed video directories found!")
        return None
    
    print(f"\\nCombining datasets from {len(video_dirs)} videos...")
    
    all_audio_sequences = []
    all_target_sequences = []
    all_vad_sequences = []
    
    combined_metadata = {
        'num_videos': len(video_dirs),
        'video_sources': [],
        'total_sequences': 0
    }
    
    for video_dir in sorted(video_dirs):
        try:
            print(f"  Loading {video_dir.name}...")
            
            audio_seq = np.load(video_dir / "audio_sequences.npy")
            target_seq = np.load(video_dir / "target_sequences.npy")
            vad_seq = np.load(video_dir / "vad_sequences.npy")
            
            all_audio_sequences.append(audio_seq)
            all_target_sequences.append(target_seq)
            all_vad_sequences.append(vad_seq)
            
            combined_metadata['video_sources'].append({
                'directory': video_dir.name,
                'num_sequences': len(audio_seq)
            })
            combined_metadata['total_sequences'] += len(audio_seq)
            
            print(f"    Loaded {len(audio_seq)} sequences")
            
        except Exception as e:
            print(f"    ⚠️ Warning: Could not load {video_dir.name}: {e}")
    
    if not all_audio_sequences:
        print("No valid sequences found!")
        return None
    
    # Concatenate all sequences
    combined_audio = np.concatenate(all_audio_sequences, axis=0)
    combined_targets = np.concatenate(all_target_sequences, axis=0)
    combined_vad = np.concatenate(all_vad_sequences, axis=0)
    
    print(f"\\nCombined dataset:")
    print(f"  Audio sequences: {combined_audio.shape}")
    print(f"  Target sequences: {combined_targets.shape}")
    print(f"  VAD sequences: {combined_vad.shape}")
    print(f"  Total sequences: {combined_metadata['total_sequences']}")
    
    # Save combined dataset
    combined_dir = output_path / "combined_dataset"
    combined_dir.mkdir(exist_ok=True)
    
    print(f"\\nSaving combined dataset to: {combined_dir}")
    
    np.save(combined_dir / "audio_sequences.npy", combined_audio)
    np.save(combined_dir / "target_sequences.npy", combined_targets)
    np.save(combined_dir / "vad_sequences.npy", combined_vad)
    
    # Copy scalers from first video
    first_video_dir = video_dirs[0]
    for scaler_file in ["audio_scaler.pkl", "target_scaler.pkl"]:
        if (first_video_dir / scaler_file).exists():
            shutil.copy(first_video_dir / scaler_file, combined_dir / scaler_file)
    
    # Save metadata
    combined_metadata['dataset_shape'] = {
        'audio_sequences': list(combined_audio.shape),
        'target_sequences': list(combined_targets.shape),
        'vad_sequences': list(combined_vad.shape)
    }
    
    with open(combined_dir / "dataset_metadata.json", 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    
    print(f"✅ Combined dataset ready!")
    return combined_dir

def main():
    parser = argparse.ArgumentParser(description="Process multiple videos for training")
    parser.add_argument("--video-dir", type=str, default="videodata", 
                       help="Directory containing video files")
    parser.add_argument("--output-dir", type=str, default="multi_video_features",
                       help="Output directory for processed features")
    
    args = parser.parse_args()
    
    print("=== PROCESSING ALL VIDEOS FOR TRAINING ===")
    
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    
    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        print(f"To use a custom path:")
        print(f"python process_all_videos.py --video-dir C:\\Users\\USER\\Videos\\mlVideoData")
        return
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(f"*{ext}")))
        video_files.extend(list(video_dir.glob(f"*{ext.upper()}")))
    
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return
    
    print(f"\\nFound {len(video_files)} video files:")
    for i, video_file in enumerate(video_files, 1):
        size_gb = video_file.stat().st_size / (1024**3)
        print(f"  {i}. {video_file.name} ({size_gb:.1f} GB)")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    results = []
    total_sequences = 0
    
    for i, video_file in enumerate(video_files):
        result = process_single_video(video_file, output_dir, i)
        results.append(result)
        
        if result and result.get('processing_status') == 'success':
            total_sequences += result.get('num_sequences', 0)
    
    # Combine all datasets
    combined_dir = combine_all_datasets(output_dir)
    
    # Print final summary
    successful = len([r for r in results if r and r.get('processing_status') == 'success'])
    failed = len(results) - successful
    
    print(f"\\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos: {len(video_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total sequences: {total_sequences}")
    
    if combined_dir:
        print(f"Combined dataset: {combined_dir}")
        print(f"\\n🎉 Ready for GPU training!")
        print(f"\\nNext step:")
        print(f"python training/train_tcn_gpu_enhanced.py --epochs 100 --batch-size 32 --data-dir {combined_dir}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()