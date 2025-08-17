#!/usr/bin/env python3
"""
Check video processing progress
"""

import json
import time
from pathlib import Path

def check_progress():
    """Check processing progress"""
    
    multi_video_dir = Path("multi_video_features")
    
    if not multi_video_dir.exists():
        print("Multi-video directory not found yet...")
        return False
    
    video_dirs = [d for d in multi_video_dir.iterdir() if d.is_dir() and d.name.startswith("video_")]
    
    print(f"\\nVideo Processing Progress:")
    print(f"{'='*50}")
    
    completed = 0
    total_sequences = 0
    
    for video_dir in sorted(video_dirs):
        video_name = video_dir.name
        
        # Check for completion markers
        has_blendshapes = (video_dir / "blendshapes_and_pose.json").exists()
        has_audio = (video_dir / "audio_features.json").exists()
        has_dataset = (video_dir / "audio_sequences.npy").exists()
        has_summary = (video_dir / "processing_summary.json").exists()
        
        status = "🟢 Complete" if has_dataset else "🟡 Processing" if has_blendshapes or has_audio else "🔴 Starting"
        
        print(f"{video_name}: {status}")
        
        if has_blendshapes:
            print(f"  ✓ Blendshapes extracted")
        if has_audio:
            print(f"  ✓ Audio features extracted") 
        if has_dataset:
            print(f"  ✓ Dataset created")
            completed += 1
            
            # Get sequence count
            try:
                with open(video_dir / "dataset_metadata.json", 'r') as f:
                    metadata = json.load(f)
                    sequences = metadata.get('dataset_info', {}).get('num_sequences', 0)
                    total_sequences += sequences
                    print(f"    Sequences: {sequences}")
            except:
                pass
        
        print()
    
    # Check for combined dataset
    combined_dir = multi_video_dir / "combined_dataset"
    if combined_dir.exists():
        print(f"🎉 Combined dataset ready!")
        try:
            with open(combined_dir / "dataset_metadata.json", 'r') as f:
                metadata = json.load(f)
                total_seq = metadata.get('total_sequences', 0)
                num_videos = metadata.get('num_videos', 0)
                print(f"   Videos: {num_videos}")
                print(f"   Total sequences: {total_seq}")
        except:
            pass
        return True
    
    print(f"Progress: {completed}/{len(video_dirs)} videos completed")
    if total_sequences > 0:
        print(f"Total sequences so far: {total_sequences}")
    
    return completed == len(video_dirs)

def main():
    """Monitor progress"""
    print("Monitoring video processing progress...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            all_complete = check_progress()
            
            if all_complete:
                print("\\n🎉 All videos processed! Ready for GPU training!")
                print("\\nTo start enhanced GPU training:")
                print("python training/train_tcn_gpu_enhanced.py --epochs 100 --batch-size 32 --data-dir multi_video_features/combined_dataset")
                break
            
            print(f"Checking again in 30 seconds...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\\nMonitoring stopped.")

if __name__ == "__main__":
    main()