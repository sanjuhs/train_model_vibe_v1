#!/usr/bin/env python3
"""
Restart Video Processing
Clean up failed runs and restart the multi-video processing pipeline
"""

import shutil
from pathlib import Path
import json
import sys

def cleanup_failed_processing():
    """Clean up failed processing artifacts"""
    
    multi_video_dir = Path("multi_video_features")
    
    if not multi_video_dir.exists():
        print("✅ No multi_video_features directory found - starting fresh")
        return
    
    print("🧹 Cleaning up previous processing artifacts...")
    
    # Find video directories
    video_dirs = [d for d in multi_video_dir.iterdir() if d.is_dir() and d.name.startswith("video_")]
    
    cleaned_dirs = 0
    for video_dir in video_dirs:
        # Check if this directory has only temp files or is incomplete
        files = list(video_dir.iterdir())
        
        # If only temp files exist, remove the whole directory
        temp_files_only = all(f.name.startswith("temp_") for f in files if f.is_file())
        
        # Check for completion
        has_complete_dataset = all([
            (video_dir / "audio_sequences.npy").exists(),
            (video_dir / "target_sequences.npy").exists(),
            (video_dir / "vad_sequences.npy").exists(),
            (video_dir / "processing_summary.json").exists()
        ])
        
        if temp_files_only or not has_complete_dataset:
            print(f"  🗑️ Removing incomplete directory: {video_dir.name}")
            shutil.rmtree(video_dir)
            cleaned_dirs += 1
        else:
            print(f"  ✅ Keeping complete directory: {video_dir.name}")
    
    # Remove combined dataset if it exists (will be recreated)
    combined_dir = multi_video_dir / "combined_dataset"
    if combined_dir.exists():
        print("  🗑️ Removing old combined dataset")
        shutil.rmtree(combined_dir)
    
    print(f"🧹 Cleanup complete! Removed {cleaned_dirs} incomplete directories")

def main():
    """Main function"""
    print("🚀 RESTARTING VIDEO PROCESSING PIPELINE")
    print("=" * 50)
    
    # Clean up failed processing
    cleanup_failed_processing()
    
    # Import and run the multi-video processor
    print("\n📹 Starting fresh multi-video processing...")
    
    try:
        sys.path.append("data_preparation_scripts")
        from data_preparation_scripts.multi_video_processor_05 import MultiVideoProcessor
        
        # Initialize processor
        processor = MultiVideoProcessor(
            video_directory="videodata",
            output_directory="multi_video_features",
            max_workers=2  # Conservative for stability
        )
        
        # Process all videos sequentially for maximum stability
        results = processor.process_all_videos(parallel=False)
        
        # If all successful, combine datasets
        successful_videos = len([r for r in results if r['processing_status'] == 'success'])
        if successful_videos > 0:
            print("\n🔄 Combining datasets from all successful videos...")
            combined_dataset_dir = processor.combine_datasets()
            print(f"✅ Combined dataset created at: {combined_dataset_dir}")
            
            print("\n🎉 PROCESSING COMPLETE!")
            print("🚀 Ready for GPU training with:")
            print(f"python training/train_tcn_gpu_enhanced.py --epochs 100 --batch-size 32 --data-dir {combined_dataset_dir}")
        else:
            print("\n❌ No videos processed successfully. Check error logs.")
            
    except ImportError:
        print("\n❌ Could not import multi-video processor. Running directly...")
        import subprocess
        result = subprocess.run([
            sys.executable, 
            "data_preparation_scripts/05_multi_video_processor.py",
            "--video-dir", "videodata",
            "--output-dir", "multi_video_features"
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
    except Exception as e:
        print(f"\n❌ Error in video processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()