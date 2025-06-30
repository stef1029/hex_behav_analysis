#!/usr/bin/env python3
"""
Cleanup utility for DeepLabCut split processing pipeline.
Removes partial files, failed outputs, and temporary directories.
"""

import sys
from pathlib import Path
import shutil
import argparse


def clean_video_directory(video_dir, video_stem=None, dry_run=False):
    """
    Clean up all temporary and partial files in a video directory.
    
    Args:
        video_dir: Directory containing the video files
        video_stem: Specific video name to clean (None for all)
        dry_run: If True, only show what would be deleted
        
    Returns:
        int: Number of files/directories cleaned
    """
    video_dir = Path(video_dir)
    cleaned_count = 0
    
    # Define patterns to clean
    if video_stem:
        patterns = [
            # Split files
            f"{video_stem}_split*of*",
            f"{video_stem}_frames*_split*of*",
            # Temporary files
            f"{video_stem}*_temp*",
            f"{video_stem}*_partial*",
            f".{video_stem}_*_in_progress",
            # Failed/incomplete markers
            f"{video_stem}*_FAILED*",
            f"{video_stem}*_merged_temp*",
            # Temporary split videos
            f"temp_splits/{video_stem}_split*.avi"
        ]
    else:
        patterns = [
            # All split files
            "*_split*of*",
            "*_frames*_split*of*",
            # All temporary files
            "*_temp*",
            "*_partial*",
            ".*_in_progress",
            # All failed/incomplete markers
            "*_FAILED*",
            "*_merged_temp*",
            # All temporary directories
            "temp_splits"
        ]
    
    print(f"Scanning directory: {video_dir}")
    if dry_run:
        print("DRY RUN - No files will be deleted")
    
    # Process each pattern
    for pattern in patterns:
        for path in video_dir.glob(pattern):
            try:
                if path.is_file():
                    if dry_run:
                        print(f"  Would remove file: {path.name}")
                    else:
                        path.unlink()
                        print(f"  Removed file: {path.name}")
                    cleaned_count += 1
                elif path.is_dir():
                    # Count files in directory
                    file_count = sum(1 for _ in path.rglob('*'))
                    if dry_run:
                        print(f"  Would remove directory: {path.name} ({file_count} files)")
                    else:
                        shutil.rmtree(path)
                        print(f"  Removed directory: {path.name} ({file_count} files)")
                    cleaned_count += 1
            except Exception as e:
                print(f"  ERROR: Could not remove {path.name}: {e}")
    
    # Also check for nested temp_splits directories
    if not video_stem:  # Only do this for full directory cleanup
        for subdir in video_dir.iterdir():
            if subdir.is_dir():
                temp_splits = subdir / "temp_splits"
                if temp_splits.exists():
                    try:
                        file_count = sum(1 for _ in temp_splits.rglob('*'))
                        if dry_run:
                            print(f"  Would remove nested temp directory: {temp_splits} ({file_count} files)")
                        else:
                            shutil.rmtree(temp_splits)
                            print(f"  Removed nested temp directory: {temp_splits} ({file_count} files)")
                        cleaned_count += 1
                    except Exception as e:
                        print(f"  ERROR: Could not remove {temp_splits}: {e}")
    
    return cleaned_count


def find_incomplete_videos(cohort_dir, splits_per_video=8):
    """
    Find videos with incomplete split processing.
    
    Args:
        cohort_dir: Directory containing video folders
        splits_per_video: Expected number of splits per video
        
    Returns:
        list: List of (video_path, found_splits, missing_splits) tuples
    """
    cohort_dir = Path(cohort_dir)
    incomplete_videos = []
    
    print(f"Scanning for incomplete videos in: {cohort_dir}")
    print(f"Expected splits per video: {splits_per_video}")
    
    # Look for video files
    for video_path in cohort_dir.rglob("*_raw_MP.avi"):
        video_stem = video_path.stem
        video_dir = video_path.parent
        
        # Look for split output files
        split_pattern = f"{video_stem}_frames*_split*of{splits_per_video}*.csv"
        split_files = list(video_dir.glob(split_pattern))
        
        # Extract split indices
        found_splits = set()
        for split_file in split_files:
            match = re.search(rf'split(\d+)of{splits_per_video}', split_file.name)
            if match:
                found_splits.add(int(match.group(1)))
        
        # Check if incomplete
        expected_splits = set(range(1, splits_per_video + 1))
        if found_splits and found_splits != expected_splits:
            missing_splits = expected_splits - found_splits
            incomplete_videos.append((video_path, found_splits, missing_splits))
    
    return incomplete_videos


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description='Clean up temporary files from DeepLabCut split processing'
    )
    parser.add_argument(
        'path',
        help='Path to cohort directory or specific video directory'
    )
    parser.add_argument(
        '--video',
        help='Specific video name to clean (without extension)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--find-incomplete',
        action='store_true',
        help='Find videos with incomplete split processing'
    )
    parser.add_argument(
        '--splits',
        type=int,
        default=8,
        help='Number of splits per video (default: 8)'
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"ERROR: Path does not exist: {path}")
        sys.exit(1)
    
    if args.find_incomplete:
        # Find incomplete videos
        incomplete = find_incomplete_videos(path, args.splits)
        
        if incomplete:
            print(f"\nFound {len(incomplete)} videos with incomplete processing:")
            for video_path, found, missing in incomplete:
                print(f"\n  {video_path.name}")
                print(f"    Found splits: {sorted(found)}")
                print(f"    Missing splits: {sorted(missing)}")
        else:
            print("\nNo incomplete videos found")
    
    else:
        # Perform cleanup
        print("=== DeepLabCut Split Cleanup ===")
        
        total_cleaned = 0
        
        if path.is_file():
            # If a video file was specified
            video_dir = path.parent
            video_stem = path.stem
            cleaned = clean_video_directory(video_dir, video_stem, args.dry_run)
            total_cleaned += cleaned
            
        elif args.video:
            # Clean specific video in directory
            cleaned = clean_video_directory(path, args.video, args.dry_run)
            total_cleaned += cleaned
            
        else:
            # Clean entire directory/cohort
            if path.is_dir():
                # Check if this is a video directory (contains .avi files)
                avi_files = list(path.glob("*.avi"))
                if avi_files:
                    # This is a video directory
                    cleaned = clean_video_directory(path, None, args.dry_run)
                    total_cleaned += cleaned
                else:
                    # This might be a cohort directory, check subdirectories
                    for subdir in path.iterdir():
                        if subdir.is_dir():
                            avi_files = list(subdir.glob("*.avi"))
                            if avi_files:
                                print(f"\nCleaning video directory: {subdir.name}")
                                cleaned = clean_video_directory(subdir, None, args.dry_run)
                                total_cleaned += cleaned
        
        print(f"\n=== Cleanup Summary ===")
        if args.dry_run:
            print(f"Would remove {total_cleaned} files/directories")
        else:
            print(f"Removed {total_cleaned} files/directories")


if __name__ == "__main__":
    import re  # Import here to avoid issues if used as module
    main()